"""CLI entry point for Stage 3 diffusion policy training.

Usage:
  # Single task (lift)
  python training/train_stage3_script.py \
    --stage1_checkpoint checkpoints/stage1_rtx5090/epoch_024.pt \
    --hdf5 data/unified/robomimic/lift/ph.hdf5

  # Multi-task
  python training/train_stage3_script.py \
    --stage1_checkpoint checkpoints/stage1_full/epoch_007.pt \
    --hdf5 data/unified/robomimic/lift/ph.hdf5 data/unified/rlbench/close_jar.hdf5

  # Multi-GPU (2 GPUs)
  torchrun --nproc_per_node=2 training/train_stage3_script.py \
    --stage1_checkpoint checkpoints/stage1_rtx5090/epoch_024.pt \
    --hdf5 data/unified/robomimic/lift/ph.hdf5
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist

from training.train_stage3 import Stage3Config, train_stage3


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Diffusion Policy Training")

    # Required
    parser.add_argument("--stage1_checkpoint", required=True,
                        help="Path to Stage 1 checkpoint (.pt)")
    parser.add_argument("--hdf5", required=True, nargs="+",
                        help="Path(s) to unified HDF5 file(s)")

    # Architecture
    parser.add_argument("--ac_dim", type=int, default=7)
    parser.add_argument("--proprio_dim", type=int, default=9)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--T_obs", type=int, default=2)
    parser.add_argument("--T_pred", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_adapter", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--norm_mode", default="minmax", choices=["minmax", "zscore"])

    # Diffusion
    parser.add_argument("--train_diffusion_steps", type=int, default=100)
    parser.add_argument("--eval_diffusion_steps", type=int, default=10)

    # Co-training & regularization
    parser.add_argument("--lambda_recon", type=float, default=0.0)
    parser.add_argument("--p_view_drop", type=float, default=0.15)
    parser.add_argument("--ema_decay", type=float, default=0.9999)

    # Checkpointing
    parser.add_argument("--save_dir", default="checkpoints/stage3")
    parser.add_argument("--save_every_epoch", type=int, default=10)
    parser.add_argument("--resume", default=None, help="Stage 3 checkpoint to resume from")

    # Inline eval video
    parser.add_argument("--eval_video_task", default="",
                        help="Task for eval video (e.g. 'lift'). Empty = disabled.")
    parser.add_argument("--eval_video_hdf5", default="",
                        help="Unified HDF5 for eval norm stats (NOT the tokens file)")
    parser.add_argument("--eval_video_episodes", type=int, default=1)
    parser.add_argument("--eval_video_steps", type=int, default=100,
                        help="DDIM steps for eval video")
    parser.add_argument("--eval_video_dir", default="eval_videos")

    args = parser.parse_args()

    # Initialize distributed if launched with torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )

    config = Stage3Config(
        stage1_checkpoint=args.stage1_checkpoint,
        hdf5_paths=args.hdf5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.num_epochs,
        norm_mode=args.norm_mode,
        ac_dim=args.ac_dim,
        proprio_dim=args.proprio_dim,
        num_views=args.num_views,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        nhead=args.nhead,
        train_diffusion_steps=args.train_diffusion_steps,
        eval_diffusion_steps=args.eval_diffusion_steps,
        lr=args.lr,
        lr_adapter=args.lr_adapter,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        lambda_recon=args.lambda_recon,
        p=args.p_view_drop,
        ema_decay=args.ema_decay,
        save_dir=args.save_dir,
        save_every_epoch=args.save_every_epoch,
        eval_video_task=args.eval_video_task,
        eval_video_hdf5=args.eval_video_hdf5,
        eval_video_episodes=args.eval_video_episodes,
        eval_video_steps=args.eval_video_steps,
        eval_video_dir=args.eval_video_dir,
    )

    train_stage3(config, resume_from=args.resume)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
