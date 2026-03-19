"""Async parallel evaluation for V3 using Chi's AsyncVectorEnv.

Runs N environments in parallel processes. Observations from all envs are
batched into a single GPU forward pass. MuJoCo physics runs concurrently
across processes (no GIL).

Usage:
    from training.eval_v3_async import evaluate_v3_async
    success_rate, results = evaluate_v3_async(
        policy, norm_stats, num_episodes=50, n_envs=4, device="cuda",
    )
"""

import logging
import math

import numpy as np
import torch

log = logging.getLogger(__name__)


def evaluate_v3_async(
    policy,
    norm_stats: dict,
    num_episodes: int = 50,
    n_envs: int = 4,
    task: str = "lift",
    max_steps: int = 400,
    seed_start: int = 100000,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    image_size: int = 84,
    device: str = "cuda",
    use_rot6d: bool = True,
    ema_model=None,
) -> tuple:
    """Run parallel evaluation with AsyncVectorEnv + batched inference.

    Args:
        policy:       PolicyDiTv3 instance (supports batch > 1).
        norm_stats:   dict with 'actions' and 'proprio' {min, max}.
        num_episodes: Total episodes to evaluate.
        n_envs:       Number of parallel environment processes.
        task:         Robosuite task name.
        max_steps:    Max steps per episode.
        seed_start:   First episode seed.
        n_obs_steps:  Observation history length.
        n_action_steps: Actions per chunk (executed before re-planning).
        image_size:   Env render size.
        device:       GPU device for policy inference.
        use_rot6d:    Convert 10D rot6d → 7D axis-angle for env.
        ema_model:    diffusers.EMAModel (optional). If provided, uses EMA weights.

    Returns:
        (success_rate, list of per-episode result dicts)
    """
    import dill
    from data_pipeline.gym_util.async_vector_env import AsyncVectorEnv
    from data_pipeline.gym_util.multistep_wrapper import MultiStepWrapper
    from data_pipeline.envs.robomimic_gym_wrapper import RobomimicGymWrapper
    from data_pipeline.utils.rotation import convert_actions_from_rot6d

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    device = torch.device(device)
    policy = policy.to(device)
    policy.eval()

    # Apply EMA weights if provided
    if ema_model is not None:
        ema_model.store(policy.parameters())
        ema_model.copy_to(policy.parameters())

    try:
        results = _run_episodes(
            policy=policy,
            num_episodes=num_episodes,
            n_envs=n_envs,
            task=task,
            max_steps=max_steps,
            seed_start=seed_start,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            image_size=image_size,
            device=device,
            use_rot6d=use_rot6d,
            action_stats=action_stats,
            proprio_stats=proprio_stats,
        )
    finally:
        if ema_model is not None:
            ema_model.restore(policy.parameters())

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / max(len(results), 1)
    log.info("Async eval: %d/%d (%.1f%%) with %d envs",
             n_success, len(results), success_rate * 100, n_envs)

    return success_rate, results


def _run_episodes(
    policy, num_episodes, n_envs, task, max_steps, seed_start,
    n_obs_steps, n_action_steps, image_size, device, use_rot6d,
    action_stats, proprio_stats,
):
    """Run episodes in chunks of n_envs."""
    from data_pipeline.gym_util.async_vector_env import AsyncVectorEnv
    from data_pipeline.gym_util.multistep_wrapper import MultiStepWrapper
    from data_pipeline.envs.robomimic_gym_wrapper import RobomimicGymWrapper
    from data_pipeline.utils.rotation import convert_actions_from_rot6d

    def env_fn():
        env = RobomimicGymWrapper(task=task, abs_action=True, image_size=image_size)
        env = MultiStepWrapper(
            env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps,
        )
        return env

    # Process episodes in chunks of n_envs
    n_chunks = math.ceil(num_episodes / n_envs)
    all_results = []

    for chunk_idx in range(n_chunks):
        ep_start = chunk_idx * n_envs
        ep_end = min(ep_start + n_envs, num_episodes)
        chunk_size = ep_end - ep_start
        seeds = [seed_start + i for i in range(ep_start, ep_end)]

        # Create vectorized env for this chunk
        env = AsyncVectorEnv([env_fn] * chunk_size, shared_memory=False)

        try:
            # Seed each env
            env.call_each("seed", [(s,) for s in seeds])

            # Reset all envs
            obs = env.reset()  # dict of (chunk_size, n_obs_steps, ...)

            done = np.zeros(chunk_size, dtype=bool)
            rewards = np.zeros(chunk_size, dtype=np.float64)
            step_counts = np.zeros(chunk_size, dtype=int)

            while not np.all(done):
                # Convert obs dict to policy input
                obs_dict = _obs_to_policy_input(
                    obs, proprio_stats, n_obs_steps, device
                )

                # Batched policy forward pass
                with torch.no_grad():
                    actions_norm = policy.predict_action(obs_dict)
                    # (chunk_size, n_action_steps, ac_dim) normalized [-1, 1]

                # Denormalize actions
                actions_raw = _denormalize_actions(
                    actions_norm.cpu().numpy(), action_stats
                )

                # Convert rot6d → axis_angle if needed
                if use_rot6d:
                    B, T_a, D = actions_raw.shape
                    actions_raw = actions_raw.reshape(B * T_a, D)
                    actions_raw = convert_actions_from_rot6d(actions_raw)
                    actions_raw = actions_raw.reshape(B, T_a, -1)

                # Step all envs (MultiStepWrapper handles action chunking)
                obs, reward, done_step, info = env.step(actions_raw)

                rewards += reward
                done = done | done_step
                step_counts += n_action_steps

            # Collect results
            # Check success via info dict
            for i in range(chunk_size):
                # MultiStepWrapper aggregates info; check last info
                ep_info = info[i] if isinstance(info, list) else {}
                success = bool(ep_info.get("success", False))

                # Also try checking via env method
                if not success:
                    try:
                        successes = env.call("is_success")
                        success = bool(successes[i])
                    except Exception:
                        pass

                all_results.append({
                    "success": success,
                    "steps": int(step_counts[i]),
                    "reward": float(rewards[i]),
                })

                status = "SUCCESS" if success else "FAIL"
                ep_num = ep_start + i + 1
                n_succ = sum(1 for r in all_results if r["success"])
                log.info("Episode %d/%d: %s (%d steps) | Running: %d/%d (%.0f%%)",
                         ep_num, num_episodes, status, step_counts[i],
                         n_succ, len(all_results),
                         100 * n_succ / len(all_results))

        finally:
            env.close()

    return all_results


def _obs_to_policy_input(obs, proprio_stats, n_obs_steps, device):
    """Convert vectorized env obs dict to PolicyDiTv3 input format.

    AsyncVectorEnv returns obs dict where each value has shape (N, n_obs_steps, ...).
    PolicyDiTv3 expects:
      images_enc: (B, T_o, K, 3, H, W) float [0,1]
      proprio:    (B, T_o, proprio_dim) normalized
      view_present: (B, K) bool

    Args:
        obs: dict from AsyncVectorEnv with keys matching RobomimicGymWrapper.
        proprio_stats: dict with 'min', 'max' for proprio normalization.
        n_obs_steps: T_o.
        device: torch device.
    """
    N = obs["agentview_image"].shape[0]  # batch size

    # Stack camera images into (B, T_o, K=4, 3, H, W)
    # Slots 0 and 3 are real cameras, 1 and 2 are zeros
    agentview = obs["agentview_image"]           # (N, T_o, 3, H, W)
    wrist = obs["robot0_eye_in_hand_image"]      # (N, T_o, 3, H, W)
    H, W = agentview.shape[-2:]
    zeros = np.zeros_like(agentview)

    images = np.stack([agentview, zeros, zeros, wrist], axis=2)  # (N, T_o, 4, 3, H, W)

    # Proprio: concatenate eef_pos + eef_quat + gripper_qpos → (N, T_o, 9)
    proprio_raw = np.concatenate([
        obs["robot0_eef_pos"],       # (N, T_o, 3)
        obs["robot0_eef_quat"],      # (N, T_o, 4)
        obs["robot0_gripper_qpos"],  # (N, T_o, 2)
    ], axis=-1)  # (N, T_o, 9)

    # Normalize proprio with minmax
    p_min = proprio_stats["min"]
    p_max = proprio_stats["max"]
    p_range = np.clip(p_max - p_min, 1e-6, None)
    proprio_norm = 2.0 * (proprio_raw - p_min) / p_range - 1.0

    # View present: slots 0 and 3 are real
    view_present = np.tile(
        np.array([True, False, False, True]), (N, 1)
    )

    return {
        "images_enc": torch.from_numpy(images).float().to(device),
        "proprio": torch.from_numpy(proprio_norm).float().to(device),
        "view_present": torch.from_numpy(view_present).to(device),
    }


def _denormalize_actions(actions_norm, action_stats):
    """Denormalize actions from [-1, 1] to raw scale."""
    a_min = action_stats["min"]
    a_max = action_stats["max"]
    a_range = a_max - a_min
    return (actions_norm + 1.0) / 2.0 * a_range + a_min
