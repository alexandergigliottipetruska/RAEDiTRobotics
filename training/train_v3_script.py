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
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Gradient clipping (0=disabled, matching Chi)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accum)")
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "constant"],
                        help="LR schedule: cosine (default) or constant (warmup then flat)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (profiled: 4 optimal)")
    parser.add_argument("--no_rot6d", action="store_true",
                        help="Disable rot6d conversion (for RLBench 8D actions)")
    parser.add_argument("--weight_decay_denoiser", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=1e-6)
    parser.add_argument("--p_drop_attn", type=float, default=0.3)
    parser.add_argument("--p_drop_emb", type=float, default=0.0)

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
                        choices=["custom", "robomimic", "rlbench", "joint"],
                        help="'custom'=RobomimicWrapper, 'robomimic'=Chi's pipeline, 'rlbench'=OMPL eval, 'joint'=joint-space eval")
    parser.add_argument("--eval_exec_horizon", type=int, default=8,
                        help="T_a: actions executed before re-planning (robomimic=8, RLBench=1)")
    parser.add_argument("--keyframe_eval", action="store_true",
                        help="Keyframe eval: predict full sequence once, execute all through OMPL")

    # Action space
    parser.add_argument("--action_space", type=str, default=None,
                        choices=["joint", "ee"],
                        help="Shortcut: 'joint' sets ac_dim=8, eval_mode=joint, no_rot6d, norm_mode=minmax; "
                             "'ee' sets ac_dim=7, eval_mode=custom, norm_mode=chi")

    # Normalization mode
    parser.add_argument("--norm_mode", type=str, default="minmax",
                        choices=["minmax", "zscore", "chi", "minmax_margin"],
                        help="'minmax'=all dims [-1,1], 'chi'=pos minmax + rot6d/grip identity, "
                             "'minmax_margin'=all dims with 0.2 margin (robobase-style)")

    # Denoiser backbone
    parser.add_argument("--denoiser_type", type=str, default="transformer",
                        choices=["transformer", "dit"],
                        help="Denoiser: 'transformer' (Chi cross-attn) or 'dit' (adaLN-Zero)")

    # Spatial tokens
    parser.add_argument("--spatial_pool_size", type=int, default=1,
                        choices=[1, 4, 7, 14],
                        help="Spatial pool: 1=avg pool (default), 4/7/14=spatial tokens per camera")
    parser.add_argument("--use_spatial_softmax", action="store_true",
                        help="Use SpatialSoftmax pooling (Chi-style spatial coordinates) instead of avg pool")
    parser.add_argument("--n_cond_layers", type=int, default=0,
                        help="Self-attention encoder layers for conditioning (0=MLP, >0=transformer encoder)")
    parser.add_argument("--use_flow_matching", action="store_true",
                        help="Use L1 Sample Flow (2-step, L1 loss) instead of DDPM/DDIM")

    # Augmentation — periodic token refresh with random crop
    parser.add_argument("--augment_refresh_every", type=int, default=0,
                        help="Re-encode train tokens with random crop every N epochs (0=disabled, 5=recommended)")
    parser.add_argument("--random_crop_size", type=int, default=208,
                        help="Crop to this size, then resize back to 224 (208 = ~93%% of 224)")
    parser.add_argument("--image_hdf5", type=str, default="",
                        help="Raw image HDF5 for token refresh (needed if augment_refresh_every > 0)")

    # Val split override (Chi uses val_ratio=0.02)
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Random val split ratio (0=use HDF5 mask, 0.02=Chi's split)")
    parser.add_argument("--val_seed", type=int, default=42)

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints/v3")
    parser.add_argument("--save_every_epoch", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    # Precision (Chi runs fp32, no compile)
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable BF16 autocast (run in fp32 like Chi)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (Chi doesn't use it)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Global seed for reproducibility (0 = random)")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Apply --action_space defaults (explicit flags still override)
    if args.action_space == "joint":
        if "--ac_dim" not in sys.argv:
            args.ac_dim = 8
        if "--eval_mode" not in sys.argv:
            args.eval_mode = "joint"
        if "--norm_mode" not in sys.argv:
            args.norm_mode = "minmax"
        args.no_rot6d = True
    elif args.action_space == "ee":
        if "--ac_dim" not in sys.argv:
            args.ac_dim = 7
        if "--eval_mode" not in sys.argv:
            args.eval_mode = "custom"
        if "--norm_mode" not in sys.argv:
            args.norm_mode = "chi"

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
        grad_accum_steps=args.grad_accum_steps,
        lr_schedule=args.lr_schedule,
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
        eval_exec_horizon=args.eval_exec_horizon,
        keyframe_eval=args.keyframe_eval,
        val_ratio=args.val_ratio,
        val_seed=args.val_seed,
        no_amp=args.no_amp,
        no_compile=args.no_compile,
        seed=args.seed,
        denoiser_type=args.denoiser_type,
        spatial_pool_size=args.spatial_pool_size,
        use_spatial_softmax=args.use_spatial_softmax,
        n_cond_layers=args.n_cond_layers,
        use_flow_matching=args.use_flow_matching,
        augment_refresh_every=args.augment_refresh_every,
        random_crop_size=args.random_crop_size,
        image_hdf5=args.image_hdf5,
    )

    train_v3(config, device=args.device, resume_from=args.resume)


if __name__ == "__main__":
    main()
