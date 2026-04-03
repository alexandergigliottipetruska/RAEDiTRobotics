"""Receding-horizon evaluation rollout loop.

Benchmark-agnostic: only the environment wrapper changes per benchmark.

Protocol (Chi et al. 2023, Dasari et al. 2024):
  1. Maintain rolling buffer of T_o past observations
  2. On replan: stack T_o frames, normalize proprio,
     pass view_present flags, run diffusion -> [T_p, action_dim]
  3. Denormalize actions (zscore or minmax)
  4. Execute first T_a actions from the chunk
  5. Re-observe and repeat until success or max_steps
"""

from collections import deque

import numpy as np
import torch


def evaluate_policy(
    policy,
    env,
    num_episodes: int = 25,
    max_steps: int = 300,
    norm_mode: str = "zscore",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
    proprio_mean: np.ndarray | None = None,
    proprio_std: np.ndarray | None = None,
    proprio_min: np.ndarray | None = None,
    proprio_max: np.ndarray | None = None,
    exec_horizon: int = 8,
    obs_horizon: int = 2,
    rot6d: bool = False,
) -> tuple[float, list[dict]]:
    """Closed-loop evaluation with receding horizon control.

    Args:
        policy: Model with .predict(images, proprio, view_present)
                -> [T_p, action_dim] normalized actions (torch tensor).
        env: BaseManipulationEnv wrapper with reset(), step(),
             get_multiview_images(), get_proprio(), get_view_present().
        num_episodes: Number of evaluation rollouts.
        max_steps: Maximum timesteps per episode.
        norm_mode: "zscore" or "minmax".
        action_mean, action_std: For zscore denormalization.
        action_min, action_max: For minmax denormalization.
        proprio_mean, proprio_std: For zscore proprio normalization.
        proprio_min, proprio_max: For minmax proprio normalization.
        exec_horizon: T_a = 8 (actions executed before re-planning).
        obs_horizon: T_o = 2 (past observation frames to condition on).

    Returns:
        success_rate: float in [0, 1]
        per_episode_results: list of dicts with 'success', 'steps', 'reward'
    """
    import logging
    _log = logging.getLogger(__name__)

    results = []

    for ep in range(num_episodes):
        env.reset()
        done = False
        step_count = 0
        action_queue = []
        total_reward = 0.0

        # Rolling observation buffer (T_o frames)
        # On reset, duplicate initial frame to fill the buffer
        init_images = env.get_multiview_images()  # [1, K, 3, H, W]
        init_proprio = env.get_proprio()            # [1, D_prop]
        img_buffer = deque(
            [init_images] * obs_horizon, maxlen=obs_horizon
        )
        proprio_buffer = deque(
            [init_proprio] * obs_horizon, maxlen=obs_horizon
        )
        view_present = env.get_view_present()  # [K] bool

        while not done and step_count < max_steps:
            if len(action_queue) == 0:
                # Stack T_o frames: [T_o, K, 3, H, W]
                images_seq = np.concatenate(list(img_buffer), axis=0)
                proprio_seq = np.concatenate(list(proprio_buffer), axis=0)

                # Normalize proprio (matches training)
                if norm_mode == "minmax" and proprio_min is not None and proprio_max is not None:
                    p_range = np.clip(proprio_max - proprio_min, 1e-6, None)
                    proprio_norm = 2.0 * (proprio_seq - proprio_min) / p_range - 1.0
                elif proprio_mean is not None and proprio_std is not None:
                    proprio_norm = (proprio_seq - proprio_mean) / proprio_std
                else:
                    proprio_norm = proprio_seq

                with torch.no_grad():
                    pred = policy.predict(
                        torch.from_numpy(images_seq).unsqueeze(0),   # [1, T_o, K, 3, H, W]
                        torch.from_numpy(proprio_norm).unsqueeze(0), # [1, T_o, D_prop]
                        torch.from_numpy(view_present).unsqueeze(0), # [1, K]
                    )  # [T_p, 7] normalized

                # Denormalize to robot action space
                raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
                if norm_mode == "minmax" and action_min is not None and action_max is not None:
                    a_range = action_max - action_min
                    raw = (raw + 1.0) / 2.0 * a_range + action_min
                elif action_mean is not None and action_std is not None:
                    raw = raw * action_std + action_mean

                # Project rot6d through quaternion space to enforce valid rotation
                if rot6d:
                    from models.policy_v3 import PolicyDiTv3
                    raw_t = torch.from_numpy(raw).float()
                    raw = PolicyDiTv3.project_rot6d_via_quaternion(raw_t).numpy()

                # Convert 10D rot6d actions → 7D axis-angle for robosuite
                if rot6d:
                    from data_pipeline.utils.rotation import convert_actions_from_rot6d
                    raw = convert_actions_from_rot6d(raw)

                action_queue = list(raw[:exec_horizon])
                if ep == 0 and step_count < 16:
                    _log.info("  step=%d action[0]: %s", step_count,
                              np.array2string(action_queue[0], precision=4, suppress_small=True))

            action = action_queue.pop(0)
            obs, reward, done, info = env.step(action)
            step_count += 1
            total_reward += reward

            # Update observation buffer every step
            img_buffer.append(env.get_multiview_images())
            proprio_buffer.append(env.get_proprio())

        success = info.get("success", False) if isinstance(info, dict) else False
        results.append({
            "success": success,
            "steps": step_count,
            "reward": total_reward,
        })
        n_success = sum(1 for r in results if r["success"])
        _log.info("Episode %d/%d: %s (%d steps) | Running: %d/%d (%.0f%%)",
                  ep + 1, num_episodes, "SUCCESS" if success else "FAIL",
                  step_count, n_success, len(results),
                  100 * n_success / len(results))

    successes = [r["success"] for r in results]
    return float(np.mean(successes)), results
