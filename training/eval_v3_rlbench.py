"""V3 evaluation for RLBench tasks.

Standalone eval loop (does NOT use rollout.py) because:
  - T_a=1: predict chunk, execute only first action, re-observe
  - Action conversion: 10D rot6d → 8D quaternion (different from robomimic's 10D → 7D aa)
  - Keeping rollout.py clean for robomimic avoids cross-benchmark bugs

Eval protocol (PerAct/CoA standard):
  - 25 episodes, variation 0, task.reset() between episodes
  - Success from RLBench's built-in success conditions
  - Per-episode timeout to prevent OMPL planning hangs

Usage:
    from training.eval_v3_rlbench import evaluate_v3_rlbench
    sr, results, env = evaluate_v3_rlbench(wrapper, norm_stats, task="close_jar")
"""

import logging
import threading
from collections import deque

import numpy as np
import torch

from data_pipeline.utils.rotation import convert_actions_rot6d_to_quat

log = logging.getLogger(__name__)

# Default max steps per RLBench task (episode length + margin)
TASK_MAX_STEPS = {
    "close_jar": 250,
    "open_drawer": 150,
    "slide_block_to_color_target": 200,
    "meat_off_grill": 200,
    "place_wine_at_rack_location": 250,
    "push_buttons": 250,
    "sweep_to_dustpan_of_size": 250,
    "turn_tap": 200,
}


def _denorm_and_convert(pred_10d: np.ndarray, action_stats: dict) -> np.ndarray:
    """Denormalize 10D rot6d actions (chi mode) and convert to 8D quaternion.

    Chi denorm: position [0:3] inverse minmax, rest identity.
    Then rot6d [3:9] → quaternion [3:7], producing 8D for OMPL.

    Args:
        pred_10d: (T_p, 10) normalized actions from policy.
        action_stats: dict with 'min' and 'max' (8D from HDF5).

    Returns:
        (T_p, 8) denormalized actions [pos(3), quat_xyzw(4), grip(1)].
    """
    raw = pred_10d.copy()

    # Inverse chi normalization: only position [0:3] was minmax'd
    pos_min = action_stats["min"][:3]
    pos_max = action_stats["max"][:3]
    raw[:, :3] = (raw[:, :3] + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    # rot6d [3:9] and grip [9] were identity-normalized, so they're already raw

    # Convert 10D rot6d → 8D quaternion
    actions_8d = convert_actions_rot6d_to_quat(raw)  # (T_p, 8)
    return actions_8d


def _run_episode(
    policy,
    env,
    action_stats: dict,
    proprio_stats: dict,
    max_steps: int,
    obs_horizon: int,
) -> dict:
    """Run a single RLBench episode with T_a=1.

    Returns dict with 'success', 'steps', 'reward'.
    """
    env.reset()

    # Initialize observation buffer (duplicate first frame for start padding)
    init_images = env.get_multiview_images()  # [1, K, 3, H, W]
    init_proprio = env.get_proprio()            # [1, D_prop]
    img_buffer = deque([init_images] * obs_horizon, maxlen=obs_horizon)
    proprio_buffer = deque([init_proprio] * obs_horizon, maxlen=obs_horizon)
    view_present = env.get_view_present()       # [K] bool

    # Proprio normalization stats (chi mode: pos minmax, quat identity, grip minmax)
    p_min, p_max = proprio_stats["min"], proprio_stats["max"]

    step_count = 0
    total_reward = 0.0
    info = {}

    while step_count < max_steps:
        # Stack T_o frames
        images_seq = np.concatenate(list(img_buffer), axis=0)    # [T_o, K, 3, H, W]
        proprio_seq = np.concatenate(list(proprio_buffer), axis=0)  # [T_o, D_prop]

        # Normalize proprio (chi mode: pos[0:3] minmax, quat[3:7] identity, grip[7:] minmax)
        # Must match stage3_dataset._normalize_proprio exactly
        proprio_norm = proprio_seq.copy()
        pos_range = np.clip(p_max[:3] - p_min[:3], 1e-6, None)
        proprio_norm[..., :3] = 2.0 * (proprio_seq[..., :3] - p_min[:3]) / pos_range - 1.0
        # [3:7] quat: identity (no change)
        g_min, g_max = p_min[7:], p_max[7:]
        g_range = np.clip(g_max - g_min, 1e-6, None)
        proprio_norm[..., 7:] = 2.0 * (proprio_seq[..., 7:] - g_min) / g_range - 1.0

        # Run policy inference
        with torch.no_grad():
            pred = policy.predict(
                torch.from_numpy(images_seq).unsqueeze(0),     # [1, T_o, K, 3, H, W]
                torch.from_numpy(proprio_norm).unsqueeze(0),   # [1, T_o, D_prop]
                torch.from_numpy(view_present).unsqueeze(0),   # [1, K]
            )  # [T_p, 10] normalized

        # Denormalize + convert 10D rot6d → 8D quaternion
        raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
        actions_8d = _denorm_and_convert(raw, action_stats)

        # T_a=1: execute only the first action
        action = actions_8d[0]

        if step_count == 0:
            log.debug("  step=0 action: %s",
                       np.array2string(action, precision=4, suppress_small=True))

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            log.warning("  OMPL step failed at step %d: %s", step_count, e)
            return {"success": False, "steps": step_count, "reward": total_reward}

        step_count += 1
        total_reward += reward

        # Update observation buffer
        img_buffer.append(env.get_multiview_images())
        proprio_buffer.append(env.get_proprio())

        if done:
            break

    success = info.get("success", False) if isinstance(info, dict) else False
    return {"success": success, "steps": step_count, "reward": total_reward}


def evaluate_v3_rlbench(
    wrapper,  # V3PolicyWrapper
    norm_stats: dict,
    task: str = "close_jar",
    num_episodes: int = 25,
    max_steps: int = 0,
    seed_start: int = 0,
    obs_horizon: int = 2,
    image_size: int = 224,
    headless: bool = True,
    episode_timeout: int = 180,
    _cached_env=None,
) -> tuple:
    """Run V3 evaluation on an RLBench task.

    Args:
        wrapper:         V3PolicyWrapper wrapping PolicyDiTv3.
        norm_stats:      dict with 'actions' and 'proprio' stats (from HDF5).
        task:            RLBench task name.
        num_episodes:    Number of eval episodes (PerAct standard: 25).
        max_steps:       Max steps per episode (0 = auto from TASK_MAX_STEPS).
        seed_start:      First episode seed (0 = no explicit seeding).
        obs_horizon:     T_o = 2 (past frames to condition on).
        image_size:      RLBench render size (224).
        headless:        Run CoppeliaSim headless.
        episode_timeout: Per-episode timeout in seconds (prevents OMPL hangs).
        _cached_env:     Pre-created RLBenchWrapper to reuse across eval calls.

    Returns:
        (success_rate, per_episode_results, env)
        env is returned so callers can cache it for reuse.
    """
    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

    if max_steps == 0:
        max_steps = TASK_MAX_STEPS.get(task, 200)

    # Reuse cached env or create new one
    env = _cached_env
    if env is None:
        log.info("Creating RLBenchWrapper for %s (headless=%s)...", task, headless)
        env = RLBenchWrapper(
            task_name=task,
            image_size=image_size,
            headless=headless,
        )

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    results = []
    for ep in range(num_episodes):
        if seed_start > 0:
            env.seed(seed_start + ep)

        # Run episode with timeout (prevents OMPL planning hangs)
        ep_result = None
        def _run():
            nonlocal ep_result
            ep_result = _run_episode(
                wrapper, env, action_stats, proprio_stats,
                max_steps, obs_horizon,
            )

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join(timeout=episode_timeout)

        if thread.is_alive():
            log.warning("Episode %d timed out after %ds — recreating env", ep, episode_timeout)
            results.append({"success": False, "steps": 0, "reward": 0.0})
            # Can't kill the thread; recreate env to get a clean CoppeliaSim state
            try:
                env.close()
            except Exception:
                pass
            env = RLBenchWrapper(
                task_name=task, image_size=image_size, headless=headless,
            )
        elif ep_result is not None:
            results.append(ep_result)
        else:
            log.warning("Episode %d failed with no result", ep)
            results.append({"success": False, "steps": 0, "reward": 0.0})

        n_success = sum(1 for r in results if r["success"])
        log.info("Episode %d/%d: %s (%d steps) | Running: %d/%d (%.0f%%)",
                 ep + 1, num_episodes,
                 "SUCCESS" if results[-1]["success"] else "FAIL",
                 results[-1]["steps"],
                 n_success, len(results),
                 100 * n_success / len(results))

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / num_episodes  # denominator is always num_episodes
    log.info("RLBench eval (%s): %.1f%% success (%d/%d episodes)",
             task, success_rate * 100, n_success, num_episodes)

    return success_rate, results, env
