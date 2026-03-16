"""Evaluate Stage 3 with video recording to visually diagnose failures.

Records camera views + predicted actions for each episode.
Saves MP4 videos and action trajectory plots.

Usage:
  python training/eval_stage3_video.py \
    --checkpoint checkpoints/stage3/best.pt \
    --stage1_checkpoint checkpoints/epoch_024.pt \
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \
    --task lift \
    --num_episodes 3 \
    --output_dir eval_videos/
"""

import argparse
import logging
import os
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore", module="robosuite")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.conversion.compute_norm_stats import load_norm_stats
from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
from data_pipeline.evaluation.stage3_eval import Stage3PolicyWrapper
from data_pipeline.evaluation.visualization import (
    plot_action_trajectory,
    save_rollout_video,
)
from training.eval_stage3 import load_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ImageNet stats for normalizing [0,1] images to match save_rollout_video expectations
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def run_episode_with_recording(
    policy_wrapper,
    env,
    norm_mode,
    action_stats,
    proprio_stats,
    max_steps=400,
    exec_horizon=8,
    obs_horizon=2,
):
    """Run one episode, recording frames and actions."""
    env.reset()
    step_count = 0
    action_queue = []
    total_reward = 0.0

    frames = []       # list of [K, 3, H, W] ImageNet-normalized
    all_actions = []   # list of [ac_dim] raw actions executed

    init_images = env.get_multiview_images()  # [1, K, 3, H, W] float [0,1]
    init_proprio = env.get_proprio()
    img_buffer = deque([init_images] * obs_horizon, maxlen=obs_horizon)
    proprio_buffer = deque([init_proprio] * obs_horizon, maxlen=obs_horizon)
    view_present = env.get_view_present()

    # Record initial frame (convert [0,1] to ImageNet-normalized for video)
    frame_01 = init_images[0]  # [K, 3, H, W]
    frame_inet = (frame_01 - _MEAN[np.newaxis]) / _STD[np.newaxis]
    frames.append(frame_inet)

    done = False
    info = {}
    while not done and step_count < max_steps:
        if len(action_queue) == 0:
            images_seq = np.concatenate(list(img_buffer), axis=0)
            proprio_seq = np.concatenate(list(proprio_buffer), axis=0)

            # Normalize proprio
            if norm_mode == "minmax" and proprio_stats.get("min") is not None:
                p_range = np.clip(proprio_stats["max"] - proprio_stats["min"], 1e-6, None)
                proprio_norm = 2.0 * (proprio_seq - proprio_stats["min"]) / p_range - 1.0
            else:
                proprio_norm = (proprio_seq - proprio_stats["mean"]) / np.clip(proprio_stats["std"], 1e-6, None)

            with torch.no_grad():
                pred = policy_wrapper.predict(
                    torch.from_numpy(images_seq).unsqueeze(0),
                    torch.from_numpy(proprio_norm).unsqueeze(0),
                    torch.from_numpy(view_present).unsqueeze(0),
                )

            # Denormalize actions
            raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
            if norm_mode == "minmax" and action_stats.get("min") is not None:
                a_range = action_stats["max"] - action_stats["min"]
                raw = (raw + 1.0) / 2.0 * a_range + action_stats["min"]
            else:
                raw = raw * action_stats["std"] + action_stats["mean"]

            action_queue = list(raw[:exec_horizon])

        action = action_queue.pop(0)
        all_actions.append(action.copy())

        obs, reward, done, info = env.step(action)
        step_count += 1
        total_reward += reward

        new_images = env.get_multiview_images()  # only call ONCE per step
        img_buffer.append(new_images)
        proprio_buffer.append(env.get_proprio())

        # Record frame for video (reuse same images, no second render call)
        cur_inet = (new_images[0] - _MEAN[np.newaxis]) / _STD[np.newaxis]
        frames.append(cur_inet)

    success = info.get("success", False) if isinstance(info, dict) else False
    actions_array = np.stack(all_actions)  # [T, ac_dim]

    return {
        "success": success,
        "steps": step_count,
        "reward": total_reward,
        "frames": frames,
        "actions": actions_array,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--hdf5", required=True, help="Unified HDF5 for norm stats")
    parser.add_argument("--task", default="lift")
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--exec_horizon", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=10,
                        help="DDIM denoising steps (try 100 for diagnostic)")
    parser.add_argument("--norm_mode", default="minmax")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="eval_videos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_type", default="ddpm", choices=["ddpm", "flow_matching"])
    parser.add_argument("--num_flow_steps", type=int, default=10,
                        help="Euler integration steps for flow matching inference")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load policy
    log.info("Loading policy from %s", args.checkpoint)
    policy = load_policy(
        args.checkpoint, args.stage1_checkpoint, device,
        policy_type=args.policy_type,
        num_flow_steps=args.num_flow_steps,
        eval_diffusion_steps=args.eval_steps,
    )

    wrapper = Stage3PolicyWrapper(policy, ema=None, device=device)

    # Load norm stats
    norm = load_norm_stats(args.hdf5)
    action_stats = norm["actions"]
    proprio_stats = norm["proprio"]

    # Create env
    log.info("Creating %s environment", args.task)
    env = RobomimicWrapper(task=args.task, seed=args.seed)

    # Run episodes with recording
    for ep in range(args.num_episodes):
        log.info("Episode %d/%d...", ep + 1, args.num_episodes)
        result = run_episode_with_recording(
            wrapper, env, args.norm_mode, action_stats, proprio_stats,
            max_steps=args.max_steps, exec_horizon=args.exec_horizon,
        )

        status = "SUCCESS" if result["success"] else "FAIL"
        log.info("  %s (%d steps, reward=%.3f)", status, result["steps"], result["reward"])

        # Save video
        video_path = output_dir / f"ep{ep:02d}_{status.lower()}.mp4"
        save_rollout_video(result["frames"], video_path, fps=20)
        log.info("  Video saved: %s (%d frames)", video_path, len(result["frames"]))

        # Save action trajectory plot
        plot_path = output_dir / f"ep{ep:02d}_{status.lower()}_actions.png"
        plot_action_trajectory(
            result["actions"],
            action_labels=["dx", "dy", "dz", "drx", "dry", "drz", "grip"],
            title=f"Episode {ep} — {status} ({result['steps']} steps)",
            output_path=plot_path,
        )
        log.info("  Action plot saved: %s", plot_path)

    env.close()
    log.info("Done. Videos saved to %s", output_dir)


if __name__ == "__main__":
    main()
