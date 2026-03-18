"""Stage 3 diffusion policy training loop (C.7).

Trains a DiT-Block Policy on adapted visual tokens from Stage 1 using
DDPM noise prediction. Supports optional co-training with reconstruction
loss (lambda_recon > 0).

Uses PolicyDiT (C.10) which composes Stage1Bridge + ViewDropout +
TokenAssembly + _DiTNoiseNet into a single module with:
  policy.compute_loss(batch) -> scalar loss
  policy.predict_action(obs) -> (B, T_p, D_act)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from data_pipeline.datasets.stage3_dataset import Stage3Dataset
from models.ema import EMA

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers (shared with Stage 1)
# ---------------------------------------------------------------------------

def _is_distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _is_distributed() else 0


def _is_main() -> bool:
    return _rank() == 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _is_distributed() else 1


def _unwrap(model: nn.Module) -> nn.Module:
    """Get the underlying module from DDP, DataParallel, or torch.compile wrapper."""
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Stage3Config:
    """Hyperparameters for Stage 3 diffusion policy training."""

    # Stage 1 checkpoint
    stage1_checkpoint: str = ""

    # Data
    hdf5_paths: list = field(default_factory=list)
    batch_size: int = 64
    num_workers: int = 4
    norm_mode: str = "minmax"

    # Architecture
    ac_dim: int = 7             # action dimension (7 for both delta and absolute robomimic)
    abs_action: str = ""        # Path to raw HDF5 with env_args for absolute actions (empty = delta)
    proprio_dim: int = 9        # proprio dimension
    num_views: int = 4          # max camera slots
    T_obs: int = 2
    T_pred: int = 16
    T_act: int = 8              # execution horizon (eval only)
    hidden_dim: int = 256       # Chi transformer: 256
    num_blocks: int = 4         # encoder + decoder layers (matched)
    nhead: int = 8

    # Diffusion
    train_diffusion_steps: int = 100
    eval_diffusion_steps: int = 100  # Chi: 100 (was 10!)

    # Policy type: "ddpm" or "flow_matching"
    policy_type: str = "ddpm"

    # Flow matching hyperparameters (ignored when policy_type="ddpm")
    fm_timestep_dist: str = "beta"       # "uniform" or "beta" (pi0)
    fm_timestep_scale: float = 1000.0    # scale tau before time_net
    fm_beta_a: float = 1.5              # Beta distribution param
    fm_beta_b: float = 1.0
    fm_cutoff: float = 0.999            # max tau (pi0: s=0.999)
    num_flow_steps: int = 10            # Euler integration steps

    # Training
    lr: float = 1e-4
    lr_adapter: float = 1e-5    # 10x lower to prevent drift
    weight_decay: float = 1e-3  # Chi transformer: 1e-3
    num_epochs: int = 3000
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"  # "cosine" or "constant"

    # Co-training (reconstruction alongside policy)
    lambda_recon: float = 0.0   # ablate {0, 0.1, 0.25, 0.5}

    # View dropout
    p: float = 0.0  # 0 for single-task, 0.15 for multi-task

    # EMA
    ema_decay: float = 0.9999

    # Logging & checkpointing
    log_every: int = 100
    save_every_epoch: int = 10
    eval_every_epoch: int = 50
    save_dir: str = "checkpoints/stage3"

    # Inline eval video (set eval_video_task="" to disable)
    eval_video_task: str = ""           # e.g. "lift"
    eval_video_episodes: int = 1
    eval_video_steps: int = 100         # DDIM steps for eval video
    eval_video_dir: str = "eval_videos"
    eval_video_hdf5: str = ""           # unified HDF5 for norm stats


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

def create_noise_scheduler(num_train_steps: int = 100) -> DDIMScheduler:
    """Create DDIM scheduler for DDPM training / DDIM inference."""
    return DDIMScheduler(
        num_train_timesteps=num_train_steps,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False,
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    batch: dict,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Stage3Config,
    global_step: int,
    use_amp: bool = False,
) -> dict:
    """One DDPM training step using PolicyDiT.

    Args:
        batch:       Dict from Stage3Dataset (images_enc, actions, proprio, view_present).
        policy:      PolicyDiT instance (composes bridge + view dropout + token assembly + noise net).
        optimizer:   AdamW optimizer.
        config:      Training config.
        global_step: Current global step (for logging).
        use_amp:     Whether to use BF16 mixed precision.

    Returns:
        Dict of scalar losses for logging.
    """
    device = next(policy.parameters()).device
    device_type = device.type

    # Move batch to device
    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
        loss = policy(batch_dev)

    # Backward + optimizer step
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            config.grad_clip,
        )

    optimizer.step()

    return {"policy": loss.item(), "total": loss.item()}


# ---------------------------------------------------------------------------
# DDIM inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_inference(
    noise_net: nn.Module,
    obs_tokens: torch.Tensor,
    ac_dim: int,
    T_pred: int,
    scheduler: DDIMScheduler,
    num_steps: int = 10,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Run DDIM denoising to produce action predictions.

    Note: For full inference with image encoding, use policy.predict_action(obs).
    This function operates on pre-encoded observation tokens.

    Args:
        noise_net: DiT noise prediction network.
        obs_tokens: [B, S_obs, d] encoded observation tokens.
        ac_dim:     Action dimension.
        T_pred:     Prediction horizon.
        scheduler:  DDIM scheduler.
        num_steps:  Number of DDIM denoising steps.
        device:     Device.

    Returns:
        Predicted actions [B, T_pred, ac_dim].
    """
    B = obs_tokens.shape[0]
    device = obs_tokens.device

    # Cache encoder outputs
    enc_cache = noise_net.forward_enc(obs_tokens)

    # Start from pure noise
    actions = torch.randn(B, T_pred, ac_dim, device=device)

    # Set inference timesteps
    scheduler.set_timesteps(num_steps, device=device)

    for t in scheduler.timesteps:
        t_batch = t.expand(B)
        eps_pred = noise_net.forward_dec(actions, t_batch, enc_cache)
        actions = scheduler.step(eps_pred, t, actions).prev_sample

    return actions


# ---------------------------------------------------------------------------
# LR scheduler with linear warmup
# ---------------------------------------------------------------------------

def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    schedule: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    """LR schedule with linear warmup.

    Args:
        schedule: "cosine" (decay to 0) or "constant" (warmup then flat).
    """
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if schedule == "constant":
            return 1.0
        # cosine decay
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    noise_net: nn.Module,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None,
    val_metrics: dict,
    decoder: nn.Module | None = None,
    obs_proj: nn.Module | None = None,
):
    """Save Stage 3 training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "noise_net": _unwrap(noise_net).state_dict(),
        "adapter": _unwrap(adapter).state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    if decoder is not None:
        ckpt["decoder"] = _unwrap(decoder).state_dict()
    if obs_proj is not None:
        ckpt["obs_proj"] = _unwrap(obs_proj).state_dict()
    torch.save(ckpt, path)
    log.info("Saved checkpoint: %s (epoch %d, step %d)", path, epoch, global_step)


def load_checkpoint(
    path: str,
    noise_net: nn.Module,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: EMA | None = None,
    decoder: nn.Module | None = None,
    obs_proj: nn.Module | None = None,
) -> tuple[int, int]:
    """Load Stage 3 checkpoint. Returns (start_epoch, global_step)."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    def _strip_compile_prefix(state_dict):
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict):
            return {k.removeprefix(prefix): v for k, v in state_dict.items()}
        return state_dict

    noise_net.load_state_dict(_strip_compile_prefix(ckpt["noise_net"]))
    adapter.load_state_dict(_strip_compile_prefix(ckpt["adapter"]))
    optimizer.load_state_dict(ckpt["optimizer"])

    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    if decoder is not None and "decoder" in ckpt:
        decoder.load_state_dict(_strip_compile_prefix(ckpt["decoder"]))
    if obs_proj is not None and "obs_proj" in ckpt:
        obs_proj.load_state_dict(_strip_compile_prefix(ckpt["obs_proj"]))

    log.info("Loaded checkpoint from epoch %d, step %d: %s",
             ckpt["epoch"], ckpt["global_step"], path)
    return ckpt["epoch"] + 1, ckpt["global_step"]


# ---------------------------------------------------------------------------
# Inline eval video recording
# ---------------------------------------------------------------------------

def _run_eval_video(policy, config, epoch, device):
    """Run 1 eval episode with video recording. Called every save_every_epoch."""
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore", module="robosuite")

    from data_pipeline.conversion.compute_norm_stats import load_norm_stats
    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
    from data_pipeline.evaluation.stage3_eval import Stage3PolicyWrapper
    from data_pipeline.evaluation.visualization import (
        plot_action_trajectory, save_rollout_video,
    )
    from training.eval_stage3_video import run_episode_with_recording

    out_dir = os.path.join(config.eval_video_dir, f"epoch_{epoch:04d}")
    os.makedirs(out_dir, exist_ok=True)

    # Override eval diffusion steps for video
    orig_eval_steps = policy._eval_steps
    policy._eval_steps = config.eval_video_steps

    wrapper = Stage3PolicyWrapper(policy, ema=None, device=str(device))

    norm = load_norm_stats(config.eval_video_hdf5)
    action_stats = norm["actions"]
    proprio_stats = norm["proprio"]

    env = RobomimicWrapper(task=config.eval_video_task, seed=42, abs_action=config.abs_action or None)

    for ep in range(config.eval_video_episodes):
        result = run_episode_with_recording(
            wrapper, env, config.norm_mode, action_stats, proprio_stats,
            max_steps=400, exec_horizon=config.T_act,
            rot6d=(config.abs_action and config.ac_dim == 10),
        )
        status = "success" if result["success"] else "fail"
        log.info("Eval video epoch %d ep %d: %s (%d steps)",
                 epoch, ep, status.upper(), result["steps"])

        save_rollout_video(
            result["frames"],
            os.path.join(out_dir, f"ep{ep:02d}_{status}.mp4"),
            fps=20,
        )
        plot_action_trajectory(
            result["actions"],
            action_labels=["dx", "dy", "dz", "drx", "dry", "drz", "grip"],
            title=f"Epoch {epoch} — {status.upper()} ({result['steps']} steps)",
            output_path=os.path.join(out_dir, f"ep{ep:02d}_{status}_actions.png"),
        )

    env.close()
    policy._eval_steps = orig_eval_steps


# ---------------------------------------------------------------------------
# Enhanced diagnostics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_diagnostics(policy, valid_loader, config, epoch, device, use_amp, metrics_path=None):
    """Run detailed diagnostics every eval_every_epoch.

    Reports:
      1. Per-timestep diffusion loss (t=0, 25, 50, 75, 99)
      2. Action prediction quality (DDIM denoise → compare to GT)
      3. Multi-episode rollout success rate (if eval_video_task is set)
    """
    import numpy as np
    pu = _unwrap(policy)
    pu.eval()

    # Grab one val batch for diagnostics
    val_batch = next(iter(valid_loader))
    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in val_batch.items()
    }

    # --- 1. Per-timestep diffusion loss ---
    proprio = batch_dev["proprio"]
    actions = batch_dev["actions"]
    view_present = batch_dev["view_present"]
    obs_tokens, _, _ = pu._encode_and_assemble(batch_dev, proprio, view_present)

    timestep_losses = {}
    for t_val in [0, 25, 50, 75, 99]:
        torch.manual_seed(42)
        noise = torch.randn_like(actions)
        timesteps = torch.full(
            (actions.shape[0],), t_val, device=actions.device, dtype=torch.long
        )
        noisy = pu.scheduler.add_noise(actions, noise, timesteps)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, eps_pred = pu.noise_net(noisy, timesteps, obs_tokens)
        loss_t = nn.functional.mse_loss(eps_pred, noise).item()
        timestep_losses[f"t{t_val}"] = loss_t

    ts_str = " | ".join(f"t{k}={v:.4f}" for k, v in sorted(timestep_losses.items()))
    log.info("  Diagnostics epoch %d — per-timestep: %s", epoch, ts_str)

    # --- 2. Action prediction quality (denoise → compare to GT) ---
    # Run full predict_action on the val batch observations
    obs_dict = {
        "proprio": proprio,
        "view_present": view_present,
    }
    if "cached_tokens" in batch_dev:
        obs_dict["cached_tokens"] = batch_dev["cached_tokens"]
    else:
        obs_dict["images_enc"] = batch_dev["images_enc"]

    with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        pred_actions = pu.predict_action(obs_dict)  # (B, T_pred, ac_dim)

    gt_actions = actions  # (B, T_pred, ac_dim)
    action_mse = nn.functional.mse_loss(pred_actions, gt_actions).item()

    # Per-dimension stats
    pred_np = pred_actions.float().cpu().numpy()
    gt_np = gt_actions.float().cpu().numpy()
    pred_mean = np.mean(pred_np, axis=(0, 1))
    pred_std = np.std(pred_np, axis=(0, 1))
    gt_mean = np.mean(gt_np, axis=(0, 1))
    gt_std = np.std(gt_np, axis=(0, 1))

    log.info("  Diagnostics epoch %d — action MSE: %.4f", epoch, action_mse)
    log.info("  Diagnostics epoch %d — pred mean: %s", epoch,
             np.array2string(pred_mean, precision=3, suppress_small=True))
    log.info("  Diagnostics epoch %d — GT   mean: %s", epoch,
             np.array2string(gt_mean, precision=3, suppress_small=True))
    log.info("  Diagnostics epoch %d — pred std:  %s", epoch,
             np.array2string(pred_std, precision=3, suppress_small=True))
    log.info("  Diagnostics epoch %d — GT   std:  %s", epoch,
             np.array2string(gt_std, precision=3, suppress_small=True))

    # --- 3. Multi-episode rollout (if configured) ---
    n_success = 0
    n_episodes = 0
    if config.eval_video_task:
        import warnings
        warnings.filterwarnings("ignore", module="robosuite")
        from data_pipeline.conversion.compute_norm_stats import load_norm_stats
        from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
        from data_pipeline.evaluation.rollout import evaluate_policy
        from data_pipeline.evaluation.stage3_eval import Stage3PolicyWrapper

        wrapper = Stage3PolicyWrapper(pu, ema=None, device=str(device))
        norm = load_norm_stats(config.eval_video_hdf5)
        action_stats = norm["actions"]
        proprio_stats = norm["proprio"]

        env = RobomimicWrapper(task=config.eval_video_task, seed=42, abs_action=config.abs_action or None)
        n_eval = 10  # 10 episodes per eval
        success_rate, results = evaluate_policy(
            wrapper, env, num_episodes=n_eval, max_steps=400,
            norm_mode=config.norm_mode,
            action_mean=action_stats["mean"], action_std=action_stats["std"],
            action_min=action_stats.get("min"), action_max=action_stats.get("max"),
            proprio_mean=proprio_stats["mean"], proprio_std=proprio_stats["std"],
            proprio_min=proprio_stats.get("min"), proprio_max=proprio_stats.get("max"),
            exec_horizon=config.T_act, obs_horizon=config.T_obs,
            rot6d=(config.abs_action and config.ac_dim == 10),
        )
        n_success = sum(1 for r in results if r["success"])
        n_episodes = n_eval
        env.close()

        log.info("  Diagnostics epoch %d — rollout: %d/%d (%.0f%%)",
                 epoch, n_success, n_episodes, success_rate * 100)

    # Log to metrics JSONL
    if metrics_path:
        import json as _json
        with open(metrics_path, "a") as mf:
            mf.write(_json.dumps({
                "epoch": epoch,
                "diagnostics": {
                    "per_timestep_loss": timestep_losses,
                    "action_mse": action_mse,
                    "pred_action_mean": pred_mean.tolist(),
                    "pred_action_std": pred_std.tolist(),
                    "gt_action_mean": gt_mean.tolist(),
                    "gt_action_std": gt_std.tolist(),
                    "rollout_success": n_success,
                    "rollout_episodes": n_episodes,
                },
            }) + "\n")

    pu.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_stage3(
    config: Stage3Config,
    *,
    device: torch.device | str = "cuda",
    resume_from: str | None = None,
):
    """Main Stage 3 training entry point.

    Creates a PolicyDiT from the config, sets up optimizer with separate
    LR for adapter, and trains with DDPM noise prediction.

    Supports single-GPU and DDP (via torchrun). DDP is auto-detected.

    Args:
        config:      Training configuration.
        device:      Device (ignored under DDP; uses LOCAL_RANK).
        resume_from: Path to checkpoint to resume from.
    """
    from models.policy_dit import PolicyDiT
    from models.stage1_bridge import Stage1Bridge

    distributed = _is_distributed()
    rank = _rank()
    is_main = _is_main()

    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device)

    use_amp = (device.type == "cuda")

    # --- Create PolicyDiT ---
    bridge = Stage1Bridge(
        checkpoint_path=config.stage1_checkpoint,
        pretrained_encoder=True,
        load_decoder=(config.lambda_recon > 0),
    )

    policy = PolicyDiT(
        bridge=bridge,
        ac_dim=config.ac_dim,
        proprio_dim=config.proprio_dim,
        hidden_dim=config.hidden_dim,
        T_obs=config.T_obs,
        T_pred=config.T_pred,
        num_blocks=config.num_blocks,
        nhead=config.nhead,
        num_views=config.num_views,
        train_diffusion_steps=config.train_diffusion_steps,
        eval_diffusion_steps=config.eval_diffusion_steps,
        p_view_drop=config.p,
        lambda_recon=config.lambda_recon,
        policy_type=config.policy_type,
        fm_timestep_dist=config.fm_timestep_dist,
        fm_timestep_scale=config.fm_timestep_scale,
        fm_beta_a=config.fm_beta_a,
        fm_beta_b=config.fm_beta_b,
        fm_cutoff=config.fm_cutoff,
        num_flow_steps=config.num_flow_steps,
    )
    policy = policy.to(device)

    # Performance: TF32 for matmuls (Ampere+), cuDNN autotuner
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # EMA on noise net
    ema = EMA(policy.noise_net, decay=config.ema_decay)

    # Optimizer with separate param groups (must be created before checkpoint load)
    param_groups = [
        {"params": list(policy.noise_net.parameters()), "lr": config.lr},
        {"params": list(policy.bridge.adapter.parameters()), "lr": config.lr_adapter},
        {"params": list(policy.view_dropout.parameters()), "lr": config.lr},
        {"params": list(policy.token_assembly.parameters()), "lr": config.lr},
    ]
    if policy.bridge.decoder is not None and config.lambda_recon > 0:
        param_groups.append(
            {"params": list(policy.bridge.decoder.parameters()), "lr": config.lr_adapter}
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        weight_decay=config.weight_decay,
    )

    # Load checkpoint before DDP wrapping
    start_epoch = 0
    global_step = 0
    if resume_from and os.path.isfile(resume_from):
        start_epoch, global_step = load_checkpoint(
            resume_from, policy.noise_net, policy.bridge.adapter,
            optimizer, ema, policy.bridge.decoder,
            obs_proj=getattr(policy, "obs_proj", None),
        )

    # torch.compile for faster training
    if device.type == "cuda" and not distributed:
        policy = torch.compile(policy)
        log.info("torch.compile enabled")

    # DDP wrapping
    if distributed:
        policy = nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank],
            find_unused_parameters=(config.lambda_recon > 0),
        )

    # Dataset + DataLoader
    train_ds = Stage3Dataset(
        config.hdf5_paths, split="train",
        T_obs=config.T_obs, T_pred=config.T_pred,
        norm_mode=config.norm_mode,
    )
    valid_ds = Stage3Dataset(
        config.hdf5_paths, split="valid",
        T_obs=config.T_obs, T_pred=config.T_pred,
        norm_mode=config.norm_mode,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    persistent = config.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=persistent,
        prefetch_factor=3 if config.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )

    total_steps = config.num_epochs * len(train_loader)
    lr_scheduler = _create_lr_scheduler(
        optimizer, config.warmup_steps, total_steps, schedule=config.lr_schedule,
    )

    # Fast-forward scheduler to match resumed global_step
    if global_step > 0:
        for _ in range(global_step):
            lr_scheduler.step()
        log.info("LR scheduler fast-forwarded to step %d (lr=%.2e)",
                 global_step, optimizer.param_groups[0]["lr"])

    # --- Logging (rank 0 only) ---
    metrics_path = None
    if is_main:
        os.makedirs(config.save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.save_dir, f"train_{ts}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        log.addHandler(fh)

        metrics_path = os.path.join(config.save_dir, f"metrics_{ts}.jsonl")

        run_info = {
            "type": "run_info",
            "timestamp": ts,
            "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
            "config": {k: v for k, v in config.__dict__.items()},
            "resume_from": resume_from,
            "start_epoch": start_epoch,
            "train_samples": len(train_ds),
            "valid_samples": len(valid_ds),
        }
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(run_info, default=str) + "\n")

        log.info("=" * 60)
        log.info("Stage 3 Diffusion Policy Training")
        log.info("=" * 60)
        log.info("Train: %d samples, Valid: %d samples", len(train_ds), len(valid_ds))
        log.info("Config: batch=%d, lr=%.1e, lr_adapter=%.1e, T_pred=%d",
                 config.batch_size, config.lr, config.lr_adapter, config.T_pred)
        log.info("Diffusion: %d train steps, %d eval steps",
                 config.train_diffusion_steps, config.eval_diffusion_steps)
        log.info("Co-training: lambda_recon=%.3f", config.lambda_recon)
        log.info("=" * 60)

    # --- Training loop ---
    best_val_loss = float("inf")

    for epoch in range(start_epoch, config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        policy.train()

        epoch_losses: dict[str, float] = {}
        n_steps = 0

        loader_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
            if is_main else train_loader
        )

        for batch in loader_iter:
            step_losses = train_step(
                batch, policy, optimizer, config, global_step,
                use_amp=use_amp,
            )

            # EMA update
            ema.update()

            # LR schedule
            lr_scheduler.step()

            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_steps += 1
            global_step += 1

        # Average epoch losses
        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

        # --- Validation loop ---
        policy.eval()
        val_losses: dict[str, float] = {}
        val_steps = 0
        with torch.no_grad():
            for val_batch in valid_loader:
                val_batch_dev = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in val_batch.items()
                }
                with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    val_loss = _unwrap(policy)(val_batch_dev)
                val_losses["policy"] = val_losses.get("policy", 0.0) + val_loss.item()
                val_steps += 1
        val_avg = {k: v / max(val_steps, 1) for k, v in val_losses.items()}
        policy.train()

        # Logging (rank 0 only)
        if is_main:
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items()))
            val_str = " | ".join(f"val_{k}={v:.4f}" for k, v in sorted(val_avg.items()))
            log.info("Epoch %d  %s | %s  (lr=%.2e)", epoch, train_str, val_str,
                     optimizer.param_groups[0]["lr"])

            if metrics_path:
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({
                        "epoch": epoch, "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train": avg,
                        "valid": val_avg,
                    }) + "\n")

            # Periodic checkpointing
            if (epoch + 1) % config.save_every_epoch == 0:
                pu = _unwrap(policy)
                save_checkpoint(
                    os.path.join(config.save_dir, f"epoch_{epoch:03d}.pt"),
                    epoch, global_step, pu.noise_net, pu.bridge.adapter,
                    optimizer, ema, avg, decoder=pu.bridge.decoder,
                    obs_proj=getattr(pu, "obs_proj", None),
                )

            # Enhanced diagnostics (separate schedule from checkpointing)
            if (epoch + 1) % config.eval_every_epoch == 0:
                try:
                    _run_diagnostics(
                        policy, valid_loader, config, epoch, device,
                        use_amp, metrics_path,
                    )
                except Exception as e:
                    log.warning("Diagnostics failed at epoch %d: %s", epoch, e)

                # Also save eval video if configured
                if config.eval_video_task:
                    try:
                        pu_eval = _unwrap(policy)
                        pu_eval.eval()
                        _run_eval_video(pu_eval, config, epoch, device)
                        pu_eval.train()
                    except Exception as e:
                        log.warning("Eval video failed at epoch %d: %s", epoch, e)

            # Best checkpoint (based on validation loss)
            if val_avg.get("policy", float("inf")) < best_val_loss:
                best_val_loss = val_avg["policy"]
                pu = _unwrap(policy)
                save_checkpoint(
                    os.path.join(config.save_dir, "best.pt"),
                    epoch, global_step, pu.noise_net, pu.bridge.adapter,
                    optimizer, ema, avg, decoder=pu.bridge.decoder,
                    obs_proj=getattr(pu, "obs_proj", None),
                )

        if distributed:
            torch.distributed.barrier()

    if is_main:
        log.info("Training complete. Best val loss=%.4f", best_val_loss)
        log.removeHandler(fh)
        fh.close()
