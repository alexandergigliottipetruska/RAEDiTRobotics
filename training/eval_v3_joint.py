"""V3 evaluation for robomimic tasks with joint-space actions.

Uses JOINT_POSITION controller instead of OSC_POSE. Actions are 8D:
[joint_positions(7), gripper(1)]. No rotation conversion needed.

Eval protocol matches robomimic standard: 50 episodes, seeded, T_a=8.

Usage:
    from training.eval_v3_joint import evaluate_v3_joint
    sr, results = evaluate_v3_joint(wrapper, norm_stats, task="lift")
"""

import logging
from collections import deque

import numpy as np
import torch

log = logging.getLogger(__name__)


def _denorm_joint_actions(pred: np.ndarray, action_stats: dict) -> np.ndarray:
    """Denormalize 8D joint-space actions (minmax).

    Args:
        pred: (T_p, 8) normalized actions [-1, 1]
        action_stats: dict with 'min' and 'max' (8D)

    Returns:
        (T_p, 8) denormalized [joint_pos(7), gripper(1)]
    """
    a_min = action_stats["min"]
    a_max = action_stats["max"]
    a_range = np.clip(a_max - a_min, 1e-6, None)
    return (pred + 1.0) / 2.0 * a_range + a_min


def evaluate_v3_joint(
    wrapper,
    norm_stats: dict,
    task: str = "lift",
    num_episodes: int = 50,
    max_steps: int = 400,
    seed_start: int = 100000,
    obs_horizon: int = 2,
    exec_horizon: int = 8,
    image_size: int = 224,
    save_video: bool = False,
    video_dir: str = "",
) -> tuple:
    """Run V3 joint-space evaluation on a robomimic task.

    Args:
        wrapper:      V3PolicyWrapper wrapping PolicyDiTv3.
        norm_stats:   dict with 'actions' (8D) and 'proprio' (9D) stats.
        task:         Robosuite task name.
        num_episodes: Number of eval episodes.
        max_steps:    Max steps per episode.
        seed_start:   First episode seed.
        obs_horizon:  T_o (past frames to condition on).
        exec_horizon: T_a (actions to execute per prediction).
        image_size:   Render size.
        save_video:   If True, save MP4 videos.
        video_dir:    Directory for videos.

    Returns:
        (success_rate, per_episode_results)
    """
    import os
    from data_pipeline.envs.robomimic_joint_wrapper import RobomimicJointWrapper

    env = RobomimicJointWrapper(task=task, image_size=image_size)

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]
    p_min, p_max = proprio_stats["min"], proprio_stats["max"]

    if save_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)

    results = []

    for ep in range(num_episodes):
        seed = seed_start + ep
        env.seed(seed)
        env.reset()

        img_buffer = deque([env.get_multiview_images()] * obs_horizon, maxlen=obs_horizon)
        proprio_buffer = deque([env.get_proprio()] * obs_horizon, maxlen=obs_horizon)
        view_present = env.get_view_present()

        frames = []
        step_count = 0
        max_reward = 0.0
        pending_actions = []

        while step_count < max_steps:
            if pending_actions:
                action = pending_actions.pop(0)
            else:
                images_seq = np.concatenate(list(img_buffer), axis=0)
                proprio_seq = np.concatenate(list(proprio_buffer), axis=0)

                # Normalize proprio (minmax on all dims for joint space)
                p_range = np.clip(p_max - p_min, 1e-6, None)
                proprio_norm = 2.0 * (proprio_seq - p_min) / p_range - 1.0

                with torch.no_grad():
                    pred = wrapper.predict(
                        torch.from_numpy(images_seq).unsqueeze(0),
                        torch.from_numpy(proprio_norm).unsqueeze(0),
                        torch.from_numpy(view_present).unsqueeze(0),
                    )

                raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
                actions_8d = _denorm_joint_actions(raw, action_stats)

                action = actions_8d[0]
                for j in range(1, min(exec_horizon, len(actions_8d))):
                    pending_actions.append(actions_8d[j])

            try:
                obs, reward, done, info = env.step(action)
            except Exception as e:
                log.warning("  Step failed at step %d: %s", step_count, e)
                break

            step_count += 1
            max_reward = max(max_reward, reward)

            img_buffer.append(env.get_multiview_images())
            proprio_buffer.append(env.get_proprio())

            if save_video:
                img = img_buffer[-1][0]
                views = [img[i].transpose(1, 2, 0) for i in range(img.shape[0])]
                frame = (np.concatenate(views, axis=1) * 255).astype(np.uint8)
                frames.append(frame)

            # Terminate on success or done
            success_now = info.get("success", False) if isinstance(info, dict) else False
            if done or success_now:
                break

        success = max_reward > 0.5
        ep_result = {"success": success, "steps": step_count, "reward": max_reward}
        results.append(ep_result)

        if save_video and frames:
            import imageio
            tag = "success" if success else "fail"
            path = os.path.join(video_dir, f"ep{ep:03d}_seed{seed}_{tag}.mp4")
            imageio.mimwrite(path, frames, fps=10)

        n_success = sum(1 for r in results if r["success"])
        if (ep + 1) % 10 == 0 or ep == 0:
            log.info("Eval ep %d/%d: seed=%d %s (running %d/%d = %.0f%%)",
                     ep + 1, num_episodes, seed,
                     "SUCCESS" if success else "FAIL",
                     n_success, len(results),
                     100 * n_success / len(results))

    env.close()

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / num_episodes
    log.info("Joint-space eval (%s): %.1f%% (%d/%d)",
             task, success_rate * 100, n_success, num_episodes)

    return success_rate, results
