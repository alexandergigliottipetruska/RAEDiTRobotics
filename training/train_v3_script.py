"""CLI entry point for V3 training.

Usage (single GPU):
    python training/train_v3_script.py \
        --hdf5 data/unified/robomimic/lift/ph_abs_v15.hdf5 \
        --stage1_checkpoint checkpoints/stage1/epoch_024.pt \
        --eval_hdf5 data/unified/robomimic/lift/ph_abs_v15.hdf5 \
        --eval_task lift

Usage (multi-GPU DDP):
    torchrun --nproc_per_node=2 training/train_v3_script.py \
        --hdf5 data/unified/robomimic/lift/ph_abs_v15.hdf5 \
        --stage1_checkpoint checkpoints/stage1/epoch_024.pt

Resume:
    python training/train_v3_script.py \
        --hdf5 ... --stage1_checkpoint ... \
        --resume checkpoints/v3/epoch_049.pt
"""

import argparse
import logging
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="V3 Diffusion Policy Training")

    # Data
    parser.add_argument("--hdf5", nargs="+", required=True,
                        help="Unified HDF5 file(s)")
    parser.add_argument("--stage1_checkpoint", type=str, default="",
                        help="Stage 1 checkpoint path")

    # Architecture
    parser.add_argument("--ac_dim", type=int, default=10)
    parser.add_argument("--proprio_dim", type=int, default=9)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--T_obs", type=int, default=2)
    parser.add_argument("--T_pred", type=int, default=10)
    parser.add_argument("--T_act", type=int, default=8)
    parser.add_argument("--pad_before", type=int, default=1,
                        help="Chi uses 1: allow windows starting before episode")
    parser.add_argument("--pad_after", type=int, default=7)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--n_active_cams", type=int, default=2,
                        help="Active cameras (2 for robomimic, 4 for RLBench)")

    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_rot6d", action="store_true",
                        help="Disable rot6d conversion (for RLBench 8D actions)")
    parser.add_argument("--weight_decay_denoiser", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=1e-6)
    parser.add_argument("--p_drop_attn", type=float, default=0.3)
    parser.add_argument("--p_drop_emb", type=float, default=0.0)
    parser.add_argument("--preload_ram", action="store_true",
                        help="Pre-load all data into RAM (needs ~30GB for fp32-none tokens)")

    # Diffusion
    parser.add_argument("--train_diffusion_steps", type=int, default=100)
    parser.add_argument("--eval_diffusion_steps", type=int, default=100)

    # Eval
    parser.add_argument("--eval_task", type=str, default="lift")
    parser.add_argument("--eval_hdf5", type=str, default="",
                        help="HDF5 for eval norm stats (defaults to first --hdf5)")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Quick eval episodes (every eval_every_epoch)")
    parser.add_argument("--eval_full_episodes", type=int, default=50,
                        help="Full eval episodes with video (every eval_full_every_epoch)")
    parser.add_argument("--eval_every_epoch", type=int, default=10)
    parser.add_argument("--eval_full_every_epoch", type=int, default=50,
                        help="Full eval + video interval")
    parser.add_argument("--eval_n_envs", type=int, default=10,
                        help="Max parallel envs for eval (default: 10)")
    parser.add_argument("--eval_mode", type=str, default="custom",
                        choices=["custom", "robomimic"],
                        help="'custom'=our RobomimicWrapper, 'robomimic'=Chi's pipeline")

    # Normalization mode
    parser.add_argument("--norm_mode", type=str, default="minmax",
                        choices=["minmax", "zscore", "chi"],
                        help="'minmax'=all dims [-1,1], 'chi'=pos minmax + rot6d/grip identity")

    # Val split override (Chi uses val_ratio=0.02)
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Random val split ratio (0=use HDF5 mask, 0.02=Chi's split)")
    parser.add_argument("--val_seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints/v3")
    parser.add_argument("--save_every_epoch", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    # DDP auto-detect
    if torch.distributed.is_available():
        local_rank = int(args.__dict__.get("local_rank", -1))
        if local_rank == -1:
            import os
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if local_rank >= 0:
            torch.distributed.init_process_group(backend="nccl")

    # Build config
    from training.train_v3 import V3Config, train_v3

    config = V3Config(
        stage1_checkpoint=args.stage1_checkpoint,
        hdf5_paths=args.hdf5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm_mode=args.norm_mode,
        use_rot6d=not args.no_rot6d,
        preload_to_ram=args.preload_ram,
        ac_dim=args.ac_dim,
        proprio_dim=args.proprio_dim,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        T_obs=args.T_obs,
        T_pred=args.T_pred,
        T_act=args.T_act,
        pad_before=args.pad_before,
        pad_after=args.pad_after,
        num_views=args.num_views,
        n_active_cams=args.n_active_cams,
        train_diffusion_steps=args.train_diffusion_steps,
        eval_diffusion_steps=args.eval_diffusion_steps,
        lr=args.lr,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        weight_decay_denoiser=args.weight_decay_denoiser,
        weight_decay_encoder=args.weight_decay_encoder,
        p_drop_attn=args.p_drop_attn,
        p_drop_emb=args.p_drop_emb,
        save_dir=args.save_dir,
        save_every_epoch=args.save_every_epoch,
        eval_every_epoch=args.eval_every_epoch,
        eval_full_every_epoch=args.eval_full_every_epoch,
        eval_n_envs=args.eval_n_envs,
        eval_task=args.eval_task,
        eval_hdf5=args.eval_hdf5 or args.hdf5[0],
        eval_episodes=args.eval_episodes,
        eval_full_episodes=args.eval_full_episodes,
        eval_mode=args.eval_mode,
        val_ratio=args.val_ratio,
        val_seed=args.val_seed,
    )

    train_v3(config, device=args.device, resume_from=args.resume)


if __name__ == "__main__":
    main()
