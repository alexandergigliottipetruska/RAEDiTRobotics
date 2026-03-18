"""Evaluate a Stage 3 policy checkpoint on robomimic tasks.

Usage:
  python training/eval_stage3.py \
    --checkpoint checkpoints/stage3_lift/epoch_090.pt \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \
    --task lift \
    --num_episodes 25
"""

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import torch

# Suppress robosuite controller warnings
warnings.filterwarnings("ignore", module="robosuite")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.conversion.compute_norm_stats import load_norm_stats
from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
from data_pipeline.evaluation.rollout import evaluate_policy
from data_pipeline.evaluation.stage3_eval import Stage3PolicyWrapper
from models.ema import EMA
from models.policy_dit import PolicyDiT
from models.stage1_bridge import Stage1Bridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_policy(
    checkpoint_path: str,
    stage1_checkpoint: str,
    device: str = "cuda",
    policy_type: str = "ddpm",
    num_flow_steps: int = 10,
    eval_diffusion_steps: int = 100,
    use_ema: bool = True,
    ac_dim: int = 7,
):
    """Load a trained PolicyDiT from checkpoint.

    Returns:
        (policy, ema) tuple. ema is None if use_ema=False or checkpoint has no EMA.
    """
    # Load Stage 1 bridge (encoder + adapter)
    bridge = Stage1Bridge(
        checkpoint_path=stage1_checkpoint,
        pretrained_encoder=True,
        load_decoder=False,
    )

    # Create policy with default architecture
    policy = PolicyDiT(
        bridge=bridge,
        ac_dim=ac_dim,
        proprio_dim=9,
        hidden_dim=256,
        T_obs=2,
        T_pred=16,
        num_blocks=4,
        nhead=8,
        num_views=4,
        train_diffusion_steps=100,
        eval_diffusion_steps=eval_diffusion_steps,
        policy_type=policy_type,
        num_flow_steps=num_flow_steps,
    )

    # Load Stage 3 checkpoint (noise_net + adapter weights)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    def _strip(sd):
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            return {k.removeprefix(prefix): v for k, v in sd.items()}
        return sd

    policy.noise_net.load_state_dict(_strip(ckpt["noise_net"]))
    if "adapter" in ckpt:
        policy.bridge.adapter.load_state_dict(_strip(ckpt["adapter"]))
    if "obs_proj" in ckpt and hasattr(policy, "obs_proj"):
        policy.obs_proj.load_state_dict(_strip(ckpt["obs_proj"]))

    policy.to(device)
    policy.eval()

    # Load EMA if available and requested
    # NOTE: EMA is on policy.noise_net only (not the full policy)
    ema = None
    if use_ema and "ema" in ckpt:
        log.info("Loading EMA weights (decay=%.4f, step=%d)",
                 ckpt["ema"].get("decay", 0), ckpt["ema"].get("_step", 0))
        ema = EMA(policy.noise_net, decay=ckpt["ema"].get("decay", 0.9999))
        ema.load_state_dict(ckpt["ema"])
    elif use_ema:
        log.warning("EMA requested but not found in checkpoint — using raw weights")

    return policy, ema


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 3 policy")
    parser.add_argument("--checkpoint", required=True, help="Stage 3 checkpoint")
    parser.add_argument("--stage1_checkpoint", required=True, help="Stage 1 checkpoint")
    parser.add_argument("--hdf5", required=True, help="Unified HDF5 (for norm stats)")
    parser.add_argument("--task", default="lift", choices=["lift", "can", "square", "tool_hang"])
    parser.add_argument("--num_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--exec_horizon", type=int, default=8)
    parser.add_argument("--norm_mode", default="minmax", choices=["zscore", "minmax"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_type", default="ddpm", choices=["ddpm", "flow_matching"])
    parser.add_argument("--num_flow_steps", type=int, default=10,
                        help="Euler integration steps for flow matching inference")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="DDIM denoising steps (Chi uses 100)")
    parser.add_argument("--no_ema", action="store_true",
                        help="Use raw weights instead of EMA")
    parser.add_argument("--abs_action", default="",
                        help="Path to raw HDF5 with env_args for absolute action eval")
    parser.add_argument("--ac_dim", type=int, default=7,
                        help="Action dimension (7 for delta/abs, 10 for abs+rot6d)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Load policy
    log.info("Loading policy from %s", args.checkpoint)
    policy, ema = load_policy(
        args.checkpoint, args.stage1_checkpoint, device,
        policy_type=args.policy_type,
        num_flow_steps=args.num_flow_steps,
        eval_diffusion_steps=args.eval_steps,
        use_ema=not args.no_ema,
        ac_dim=args.ac_dim,
    )

    # Wrap for eval harness
    wrapper = Stage3PolicyWrapper(policy, ema=ema, device=device)

    # Load norm stats from HDF5
    norm = load_norm_stats(args.hdf5)
    action_stats = norm["actions"]
    proprio_stats = norm["proprio"]

    # Create environment
    log.info("Creating %s environment", args.task)
    env = RobomimicWrapper(task=args.task, seed=args.seed,
                           abs_action=args.abs_action or None)

    # Run evaluation
    log.info("Running %d episodes...", args.num_episodes)
    success_rate, results = evaluate_policy(
        wrapper,
        env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        norm_mode=args.norm_mode,
        action_mean=action_stats["mean"],
        action_std=action_stats["std"],
        action_min=action_stats.get("min"),
        action_max=action_stats.get("max"),
        proprio_mean=proprio_stats["mean"],
        proprio_std=proprio_stats["std"],
        proprio_min=proprio_stats.get("min"),
        proprio_max=proprio_stats.get("max"),
        exec_horizon=args.exec_horizon,
        obs_horizon=2,
        rot6d=bool(args.abs_action) and args.ac_dim == 10,
    )

    # Report
    n_success = sum(1 for r in results if r["success"])
    log.info("=" * 50)
    log.info("Task: %s", args.task)
    log.info("Checkpoint: %s", args.checkpoint)
    log.info("Success rate: %d/%d (%.1f%%)", n_success, args.num_episodes, success_rate * 100)
    log.info("Per-episode: %s", [r["success"] for r in results])
    log.info("Avg steps: %.1f", np.mean([r["steps"] for r in results]))
    log.info("Avg reward: %.3f", np.mean([r["reward"] for r in results]))
    log.info("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
