"""V3 training loop: PolicyDiTv3 with Chi's cross-attention transformer.

V3-specific features:
  - PolicyDiTv3 instead of PolicyDiT
  - diffusers.EMAModel(power=0.75) adaptive schedule
  - Optimizer betas=(0.9, 0.95), two-tier weight decay
  - V3-specific checkpoint format
  - Stage3Dataset with use_rot6d=True
  - Eval via eval_v3.py (no external ImageNet norm)
"""

import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data_pipeline.datasets.stage3_dataset import Stage3Dataset

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
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
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    batch: dict,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    config,
    global_step: int,
    use_amp: bool = False,
) -> dict:
    """One DDPM training step.

    Args:
        batch:       Dict from Stage3Dataset (images_enc, actions, proprio, view_present).
        policy:      Policy instance (must be callable, returning scalar loss).
        optimizer:   AdamW optimizer.
        config:      Training config (must have .grad_clip attribute).
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
# V3 Config
# ---------------------------------------------------------------------------

@dataclass
class V3Config:
    """Hyperparameters for V3 training (Chi's transformer denoiser)."""

    # Stage 1 checkpoint
    stage1_checkpoint: str = ""

    # Data
    hdf5_paths: list = field(default_factory=list)
    batch_size: int = 64
    num_workers: int = 4
    norm_mode: str = "minmax"
    use_rot6d: bool = True  # convert 7D→10D in __getitem__

    # Architecture
    ac_dim: int = 10            # 10D for robomimic rot6d, 8 for RLBench
    proprio_dim: int = 9
    num_views: int = 4
    n_active_cams: int = 2      # 2 for robomimic (agentview + wrist), 4 for RLBench
    T_obs: int = 2
    T_pred: int = 10            # Chi: horizon=10 (was 16)
    T_act: int = 8              # execution horizon (eval only)
    pad_before: int = 0         # Chi: pad_before=1 (allow windows starting before episode)
    pad_after: int = 7          # Chi: pad_after=7 (repeat last action at demo end)
    d_model: int = 256          # Chi: 256
    n_head: int = 4             # Chi: 4
    n_layers: int = 8           # Chi: 8

    # Diffusion
    train_diffusion_steps: int = 100
    eval_diffusion_steps: int = 100

    # Training — Chi's transformer recipe
    lr: float = 1e-4
    betas: tuple = (0.9, 0.95)          # Chi transformer (NOT 0.9, 0.999)
    weight_decay_denoiser: float = 1e-3  # Chi: transformer_weight_decay
    weight_decay_encoder: float = 1e-6   # Chi: obs_encoder_weight_decay
    num_epochs: int = 100
    grad_clip: float = 0.0              # Chi: no gradient clipping
    warmup_steps: int = 1000            # Chi: 1000
    lr_schedule: str = "cosine"

    # Dropout (Chi: p_drop_attn=0.3, p_drop_emb=0.0 — verified from config)
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3

    # EMA — diffusers adaptive schedule
    ema_power: float = 0.75             # Chi: power=0.75 (adaptive decay)
    ema_max_decay: float = 0.9999

    # Denoiser backbone
    denoiser_type: str = "transformer"  # "transformer" (Chi cross-attn) or "dit" (adaLN-Zero)

    # Spatial tokens
    spatial_pool_size: int = 1          # 1=avg pool (default), 4/7/14=spatial tokens

    # Augmentation — periodic token refresh with random crop
    augment_refresh_every: int = 0      # 0=disabled, 5=refresh train tokens every 5 epochs
    random_crop_size: int = 208         # crop 224→208→resize 224
    image_hdf5: str = ""                # raw image HDF5 for token refresh

    # Logging & checkpointing
    log_every: int = 100
    save_every_epoch: int = 10
    eval_every_epoch: int = 5           # eval every 5 epochs (first 50), then 50
    save_dir: str = "checkpoints/v3"

    # Eval
    eval_task: str = "lift"
    eval_hdf5: str = ""                 # unified HDF5 for norm stats
    eval_episodes: int = 10
    eval_full_episodes: int = 50        # episodes for full eval (with video)
    eval_full_every_epoch: int = 50     # full eval + video every N epochs
    eval_n_envs: int = 10              # max parallel envs for eval
    eval_image_size: int = 84
    eval_mode: str = "custom"           # "custom" | "robomimic" | "rlbench"
    eval_exec_horizon: int = 8          # T_a: robomimic=8, RLBench=1

    # Val split override (0 = use HDF5 mask, >0 = random split like Chi)
    val_ratio: float = 0.0              # Chi uses 0.02 (4 val demos, seed=42)
    val_seed: int = 42                  # seed for random val split

    # Precision — Chi runs fp32, no torch.compile
    no_amp: bool = False                # disable BF16 autocast
    no_compile: bool = False            # disable torch.compile

    # Reproducibility
    seed: int = 0                       # global seed (0 = random)


# ---------------------------------------------------------------------------
# V3 Checkpointing
# ---------------------------------------------------------------------------

def save_v3_checkpoint(
    path: str,
    epoch: int,
    global_step: int,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model,
    val_metrics: dict,
):
    """Save V3 checkpoint — trainable components only (skip frozen encoder)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pu = _unwrap(policy)
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "denoiser": pu.denoiser.state_dict(),
        "obs_encoder": pu.obs_encoder.state_dict(),
        "adapter": pu.bridge.adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_metrics": val_metrics,
    }
    if ema_model is not None:
        ckpt["ema"] = {
            "averaged_model": ema_model.averaged_model.state_dict(),
            "optimization_step": ema_model.optimization_step,
            "decay": ema_model.decay,
        }
    torch.save(ckpt, path)
    log.info("Saved V3 checkpoint: %s (epoch %d, step %d)", path, epoch, global_step)


def load_v3_checkpoint(
    path: str,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema_model=None,
) -> tuple:
    """Load V3 checkpoint. Returns (start_epoch, global_step)."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    def _strip(sd):
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            return {k.removeprefix(prefix): v for k, v in sd.items()}
        return sd

    def _resize_pos_emb(target_module, loaded_sd, key="cond_pos_emb"):
        """Expand positional embeddings if the model's buffer is larger than the checkpoint's."""
        if key in loaded_sd:
            old = loaded_sd[key]
            # Resolve dotted key on the module (e.g. "denoiser.cond_pos_emb")
            obj = target_module
            for attr in key.split("."):
                obj = getattr(obj, attr)
            new_shape = obj.shape
            if old.shape != new_shape:
                new_param = torch.zeros(new_shape, dtype=old.dtype, device=old.device)
                min_len = min(old.shape[1], new_shape[1])
                new_param[:, :min_len, :] = old[:, :min_len, :]
                loaded_sd[key] = new_param
                log.info("Resized %s from %s -> %s", key, list(old.shape), list(new_shape))

    # Load trainable components (backward-compat with old full-policy checkpoints)
    if "denoiser" in ckpt:
        denoiser_sd = _strip(ckpt["denoiser"])
        _resize_pos_emb(policy.denoiser, denoiser_sd)
        policy.denoiser.load_state_dict(denoiser_sd)
        policy.obs_encoder.load_state_dict(_strip(ckpt["obs_encoder"]))
        policy.bridge.adapter.load_state_dict(_strip(ckpt["adapter"]))
    else:
        # Old format: full policy state dict
        policy.load_state_dict(_strip(ckpt["policy"]))
    optimizer.load_state_dict(ckpt["optimizer"])

    # Resize optimizer state buffers for any parameters whose shape changed (e.g. cond_pos_emb)
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None:
                continue
            for buf_key in ("exp_avg", "exp_avg_sq"):
                if buf_key in state and state[buf_key].shape != p.shape:
                    old_buf = state[buf_key]
                    new_buf = torch.zeros_like(p)
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_buf.shape, p.shape))
                    new_buf[slices] = old_buf[slices]
                    state[buf_key] = new_buf
                    log.info("Resized optimizer %s from %s -> %s", buf_key, list(old_buf.shape), list(p.shape))

    if ema_model is not None and "ema" in ckpt:
        ema_state = ckpt["ema"]
        if "averaged_model" in ema_state:
            # Chi's EMA format — resize pos emb if needed
            ema_sd = ema_state["averaged_model"]
            _resize_pos_emb(ema_model.averaged_model, ema_sd,
                            key="denoiser.cond_pos_emb")
            ema_model.averaged_model.load_state_dict(ema_sd)
            ema_model.optimization_step = ema_state.get("optimization_step", 0)
            ema_model.decay = ema_state.get("decay", 0.0)
        else:
            # Old diffusers EMA format — skip (incompatible)
            log.warning("Old diffusers EMA format detected, skipping EMA load (will re-init)")

    log.info("Loaded V3 checkpoint from epoch %d, step %d: %s",
             ckpt["epoch"], ckpt["global_step"], path)
    return ckpt["epoch"] + 1, ckpt["global_step"]


# ---------------------------------------------------------------------------
# Main V3 training loop
# ---------------------------------------------------------------------------

def train_v3(
    config: V3Config,
    *,
    device: str = "cuda",
    resume_from: str | None = None,
):
    """Main V3 training entry point.

    Creates PolicyDiTv3, sets up optimizer with Chi's recipe, and trains.
    """
    import copy
    from models.ema_model import EMAModel
    from models.policy_v3 import PolicyDiTv3
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

    # Seed everything for reproducibility
    if config.seed > 0:
        import random
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log.info("Global seed set to %d (deterministic mode)", config.seed)

    use_amp = (device.type == "cuda") and not config.no_amp

    # --- Create PolicyDiTv3 ---
    bridge = Stage1Bridge(
        checkpoint_path=config.stage1_checkpoint,
        pretrained_encoder=True,
    )

    policy = PolicyDiTv3(
        bridge=bridge,
        ac_dim=config.ac_dim,
        proprio_dim=config.proprio_dim,
        d_model=config.d_model,
        n_head=config.n_head,
        n_layers=config.n_layers,
        T_obs=config.T_obs,
        T_pred=config.T_pred,
        num_views=config.num_views,
        n_active_cams=config.n_active_cams,
        train_diffusion_steps=config.train_diffusion_steps,
        eval_diffusion_steps=config.eval_diffusion_steps,
        p_drop_emb=config.p_drop_emb,
        p_drop_attn=config.p_drop_attn,
        spatial_pool_size=config.spatial_pool_size,
        denoiser_type=config.denoiser_type,
    )
    policy = policy.to(device)

    # Performance (skip benchmark when seeded — it breaks determinism)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        if config.seed <= 0:
            torch.backends.cudnn.benchmark = True

    # --- EMA: Chi's implementation (separate model copy) ---
    ema_policy = copy.deepcopy(policy)
    ema_policy.eval()
    ema_policy.requires_grad_(False)
    ema_model = EMAModel(
        model=ema_policy,
        power=config.ema_power,
        max_value=config.ema_max_decay,
    )

    # --- Optimizer: Chi's transformer recipe ---
    # Denoiser: Chi's get_optim_groups splits into decay/no_decay
    # (biases, LayerNorm, pos_emb get WD=0; Linear/MHA weights get WD=1e-3)
    denoiser_groups = policy.denoiser.get_optim_groups(
        weight_decay=config.weight_decay_denoiser,
    )
    param_groups = denoiser_groups + [
        {
            "params": list(policy.obs_encoder.parameters()),
            "weight_decay": config.weight_decay_encoder,
        },
        {
            "params": list(policy.bridge.adapter.parameters()),
            "weight_decay": config.weight_decay_encoder,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
    )

    # Load checkpoint before DDP/compile
    start_epoch = 0
    global_step = 0
    if resume_from and os.path.isfile(resume_from):
        start_epoch, global_step = load_v3_checkpoint(
            resume_from, policy, optimizer, ema_model,
        )

    # Ensure EMA model is on the right device
    ema_model.averaged_model.to(device)

    # torch.compile (Chi doesn't use it)
    if device.type == "cuda" and not distributed and not config.no_compile:
        policy = torch.compile(policy)
        log.info("torch.compile enabled")

    # DDP
    if distributed:
        policy = nn.parallel.DistributedDataParallel(
            policy, device_ids=[int(os.environ.get("LOCAL_RANK", 0))],
        )

    # --- Dataset ---
    # If val_ratio > 0, create a random split (matching Chi's approach)
    train_keys_override = None
    valid_keys_override = None
    if config.val_ratio > 0:
        import h5py
        # Get all demo keys from the first HDF5 file
        with h5py.File(config.hdf5_paths[0], "r") as f:
            all_keys = sorted(f["data"].keys())
        n_demos = len(all_keys)
        n_val = min(max(1, round(n_demos * config.val_ratio)), n_demos - 1)
        rng = np.random.default_rng(seed=config.val_seed)
        val_idxs = rng.choice(n_demos, size=n_val, replace=False)
        val_mask = np.zeros(n_demos, dtype=bool)
        val_mask[val_idxs] = True
        valid_keys_override = [all_keys[i] for i in range(n_demos) if val_mask[i]]
        train_keys_override = [all_keys[i] for i in range(n_demos) if not val_mask[i]]
        if _is_main():
            log.info("Custom val split: %d train, %d valid (val_ratio=%.2f, seed=%d)",
                     len(train_keys_override), len(valid_keys_override),
                     config.val_ratio, config.val_seed)

    train_ds = Stage3Dataset(
        config.hdf5_paths, split="train",
        T_obs=config.T_obs, T_pred=config.T_pred,
        norm_mode=config.norm_mode, use_rot6d=config.use_rot6d,
        pad_before=config.pad_before, pad_after=config.pad_after,
        demo_keys_override=train_keys_override,
        image_hdf5_path=config.image_hdf5,
    )
    valid_ds = Stage3Dataset(
        config.hdf5_paths, split="valid",
        T_obs=config.T_obs, T_pred=config.T_pred,
        norm_mode=config.norm_mode, use_rot6d=config.use_rot6d,
        pad_before=config.pad_before, pad_after=config.pad_after,
        demo_keys_override=valid_keys_override,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    persistent = config.num_workers > 0

    from data_pipeline.datasets.stage3_dataset import worker_init_open_handles

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=persistent,
        prefetch_factor=3 if config.num_workers > 0 else None,
        worker_init_fn=worker_init_open_handles if config.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
        worker_init_fn=worker_init_open_handles if config.num_workers > 0 else None,
    )

    total_steps = config.num_epochs * len(train_loader)
    lr_scheduler = _create_lr_scheduler(
        optimizer, config.warmup_steps, total_steps, schedule=config.lr_schedule,
    )

    if global_step > 0:
        for _ in range(global_step):
            lr_scheduler.step()
        log.info("LR scheduler fast-forwarded to step %d", global_step)

    # --- Logging ---
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
            "type": "run_info", "version": "v3", "timestamp": ts,
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
        denoiser_label = {
            "transformer": "Chi's Cross-Attention Transformer",
            "dit": "DiT with adaLN-Zero Blocks (Peebles & Xie)",
        }.get(config.denoiser_type, config.denoiser_type)
        log.info("V3 Training: %s", denoiser_label)
        log.info("=" * 60)
        log.info("Train: %d, Valid: %d, Batch: %d", len(train_ds), len(valid_ds), config.batch_size)
        log.info("Arch: d=%d, heads=%d, layers=%d, ac_dim=%d",
                 config.d_model, config.n_head, config.n_layers, config.ac_dim)
        log.info("Optim: lr=%.1e, betas=%s, WD_den=%.1e, WD_enc=%.1e",
                 config.lr, config.betas, config.weight_decay_denoiser, config.weight_decay_encoder)
        log.info("EMA: power=%.2f, Diffusion: %d train / %d eval steps",
                 config.ema_power, config.train_diffusion_steps, config.eval_diffusion_steps)
        log.info("Precision: AMP=%s, compile=%s", not config.no_amp, not config.no_compile)
        log.info("=" * 60)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_success_rate = -1.0
    last_best_save_epoch = -999  # cooldown for best.pt saves

    # Save first training batch for diagnostics (Chi measures t0 on this)
    train_sampling_batch = None

    for epoch in range(start_epoch, config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Periodic token refresh with random crop augmentation
        if (config.augment_refresh_every > 0
                and epoch > 0
                and epoch % config.augment_refresh_every == 0):
            import time as _time
            _t0 = _time.time()
            log.info("Refreshing train tokens with random crop (epoch %d)...", epoch)
            bridge = _unwrap(policy).bridge
            train_ds.refresh_cached_tokens(
                encoder=bridge.encoder,
                device=device,
                crop_size=config.random_crop_size,
            )
            log.info("Token refresh done in %.1fs", _time.time() - _t0)

        policy.train()
        epoch_losses = {}
        n_steps = 0

        loader_iter = (
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
            if is_main else train_loader
        )

        for batch in loader_iter:
            if train_sampling_batch is None:
                train_sampling_batch = {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

            step_losses = train_step(
                batch, policy, optimizer, config, global_step, use_amp=use_amp,
            )

            ema_model.step(_unwrap(policy))
            lr_scheduler.step()

            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_steps += 1
            global_step += 1

            # Update tqdm with per-step loss
            if is_main and hasattr(loader_iter, 'set_postfix'):
                loader_iter.set_postfix(
                    loss=f"{step_losses['total']:.4f}",
                    avg=f"{epoch_losses['total'] / n_steps:.4f}",
                )

        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

        # --- Validation ---
        policy.eval()
        val_losses = {}
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

        # --- Logging ---
        if is_main:
            # EMA decay value
            ema_decay = ema_model.decay if hasattr(ema_model, 'decay') else -1
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items()))
            val_str = " | ".join(f"val_{k}={v:.4f}" for k, v in sorted(val_avg.items()))
            log.info("Epoch %d  %s | %s  (lr=%.2e, ema=%.4f)", epoch, train_str, val_str,
                     optimizer.param_groups[0]["lr"], ema_decay)

            if metrics_path:
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({
                        "epoch": epoch, "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "ema_decay": ema_decay,
                        "train": avg, "valid": val_avg,
                    }) + "\n")

            # Periodic checkpoint
            if (epoch + 1) % config.save_every_epoch == 0:
                save_v3_checkpoint(
                    os.path.join(config.save_dir, f"epoch_{epoch:03d}.pt"),
                    epoch, global_step, policy, optimizer, ema_model, avg,
                )

            # Per-timestep diagnostics (t0, denoised-MSE) — every eval_every_epoch + full eval epochs
            is_last_epoch = (epoch == config.num_epochs - 1)
            is_full_eval_epoch = (epoch + 1) % config.eval_full_every_epoch == 0
            run_diag = (epoch + 1) % config.eval_every_epoch == 0 or is_full_eval_epoch or is_last_epoch
            if run_diag:
                try:
                    _run_per_timestep_diagnostic(
                        policy, train_loader, valid_loader, ema_model,
                        epoch, device, use_amp, metrics_path,
                        train_sampling_batch=train_sampling_batch,
                    )
                except Exception as e:
                    log.warning("Per-timestep diagnostic failed at epoch %d: %s", epoch, e)

            # Quick eval (no video) — runs on eval_every_epoch but NOT on full eval epochs
            if (epoch + 1) % config.eval_every_epoch == 0 and not is_full_eval_epoch:
                if config.eval_task and config.eval_hdf5:
                    try:
                        sr = _run_v3_eval(policy, ema_model, config, epoch, device,
                                          num_episodes=config.eval_episodes, save_video=False)
                        if metrics_path:
                            with open(metrics_path, "a") as mf:
                                mf.write(json.dumps({
                                    "epoch": epoch,
                                    "eval_success_rate": sr,
                                    "eval_episodes": config.eval_episodes,
                                    "eval_full": False,
                                }) + "\n")
                        if sr > best_success_rate:
                            best_success_rate = sr
                            save_v3_checkpoint(
                                os.path.join(config.save_dir, "best_success.pt"),
                                epoch, global_step, policy, optimizer, ema_model,
                                {**avg, "success_rate": sr},
                            )
                            log.info("New best success rate: %.1f%% (epoch %d)", sr * 100, epoch)
                    except Exception as e:
                        log.warning("Eval failed at epoch %d: %s", epoch, e)

            # Full eval (with video) every eval_full_every_epoch
            if is_full_eval_epoch:
                if config.eval_task and config.eval_hdf5:
                    try:
                        n_eps = config.eval_full_episodes
                        sr = _run_v3_eval(policy, ema_model, config, epoch, device,
                                          num_episodes=n_eps, save_video=True)

                        # Log eval success rate to metrics jsonl
                        if metrics_path:
                            with open(metrics_path, "a") as mf:
                                mf.write(json.dumps({
                                    "epoch": epoch,
                                    "eval_success_rate": sr,
                                    "eval_episodes": n_eps,
                                    "eval_full": True,
                                }) + "\n")

                        if sr > best_success_rate:
                            best_success_rate = sr
                            save_v3_checkpoint(
                                os.path.join(config.save_dir, "best_success.pt"),
                                epoch, global_step, policy, optimizer, ema_model,
                                {**avg, "success_rate": sr},
                            )
                            log.info("New best success rate: %.1f%% (epoch %d)", sr * 100, epoch)
                    except Exception as e:
                        log.warning("Eval failed at epoch %d: %s", epoch, e)

            # Best checkpoint (by val loss) with 5-epoch cooldown
            if val_avg.get("policy", float("inf")) < best_val_loss:
                best_val_loss = val_avg["policy"]
                if epoch - last_best_save_epoch >= 5:
                    save_v3_checkpoint(
                        os.path.join(config.save_dir, "best.pt"),
                        epoch, global_step, policy, optimizer, ema_model, avg,
                    )
                    last_best_save_epoch = epoch
                else:
                    log.info("New best val loss=%.4f but skipping save (cooldown, last saved epoch %d)", best_val_loss, last_best_save_epoch)

        if distributed:
            torch.distributed.barrier()

    if is_main:
        log.info("V3 training complete. Best val loss=%.4f", best_val_loss)
        log.removeHandler(fh)
        fh.close()


@torch.no_grad()
def _run_per_timestep_diagnostic(
    policy, train_loader, valid_loader, ema_model,
    epoch, device, use_amp, metrics_path,
    train_sampling_batch=None,
):
    """Per-timestep diffusion loss + denoised-action-MSE diagnostic.

    Measures t0 on BOTH train batch (Chi's method) and val batch.
    Chi saves the first training batch and reuses it every diagnostic.
    """
    import torch.nn.functional as F
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    pu = _unwrap(policy)
    pu.eval()

    # EMA model for diagnostics (Chi's approach)
    diag_model = ema_model.averaged_model if ema_model is not None else pu
    diag_model.eval()

    def _t0_loss(batch_d, model):
        """Compute t0 noise prediction loss on a batch."""
        obs_c = model._encode_obs(batch_d)
        actions = batch_d["actions"]
        torch.manual_seed(42)
        noise = torch.randn_like(actions)
        timesteps = torch.zeros(actions.shape[0], device=device, dtype=torch.long)
        noisy = model.noise_scheduler.add_noise(actions, noise, timesteps)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = model.denoiser(noisy, timesteps, obs_c)
        return F.mse_loss(pred, noise).item()

    # --- Per-timestep loss on val batch (EMA weights) ---
    val_batch = next(iter(valid_loader))
    val_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in val_batch.items()
    }

    timestep_losses = {}
    obs_cond_val = diag_model._encode_obs(val_dev)
    n_train_steps = diag_model.train_diffusion_steps
    # Scale diagnostic timesteps to actual schedule (handles 50 or 100 steps)
    for t_val in sorted(set([0, n_train_steps // 4, n_train_steps // 2,
                             3 * n_train_steps // 4, n_train_steps - 1])):
        torch.manual_seed(42)
        noise = torch.randn_like(val_dev["actions"])
        timesteps = torch.full((val_dev["actions"].shape[0],), t_val, device=device, dtype=torch.long)
        noisy = diag_model.noise_scheduler.add_noise(val_dev["actions"], noise, timesteps)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred = diag_model.denoiser(noisy, timesteps, obs_cond_val)
        loss_t = F.mse_loss(pred, noise).item()
        timestep_losses[f"t{t_val}"] = loss_t

    ts_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(timestep_losses.items()))
    log.info("  Per-timestep epoch %d: %s", epoch, ts_str)

    # --- t0 on train batch (Chi's method) and val batch ---
    train_t0 = None
    if train_sampling_batch is not None:
        train_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in train_sampling_batch.items()
        }
        train_t0 = _t0_loss(train_dev, diag_model)

    val_t0 = _t0_loss(val_dev, diag_model)

    log.info("  t0 epoch %d: train=%.4f | val=%.4f",
             epoch, train_t0 if train_t0 is not None else -1, val_t0)

    # --- Denoised-action-MSE (Chi's train_action_mse_error equivalent) ---
    def _denoised_mse(batch_d, obs_c, model=None):
        """Run DDIM sampling and compute MSE vs GT actions."""
        m = model or pu
        scheduler = DDIMScheduler(
            num_train_timesteps=m.train_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )
        scheduler.set_timesteps(m.eval_diffusion_steps, device=device)
        B = batch_d["actions"].shape[0]
        noisy_actions = torch.randn_like(batch_d["actions"])
        for t in scheduler.timesteps:
            ts = t.expand(B)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                noise_pred = m.denoiser(noisy_actions, ts, obs_c)
            noisy_actions = scheduler.step(noise_pred, t, noisy_actions).prev_sample
        return F.mse_loss(noisy_actions, batch_d["actions"]).item()

    # EMA denoised-MSE on val
    ema_val_action_mse = None
    if ema_model is not None:
        torch.manual_seed(42)
        ema_val_action_mse = _denoised_mse(val_dev, obs_cond_val, model=diag_model)

    log.info("  Denoised-MSE epoch %d: ema_val=%s",
             epoch,
             f"{ema_val_action_mse:.4f}" if ema_val_action_mse is not None else "N/A")

    if metrics_path:
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps({
                "epoch": epoch,
                "per_timestep_loss": timestep_losses,
                "train_t0": train_t0,
                "val_t0": val_t0,
                "ema_val_action_mse": ema_val_action_mse,
            }) + "\n")

    pu.train()


def _run_v3_eval(policy, ema_model, config, epoch, device,
                 num_episodes=None, save_video=False) -> float:
    """Run V3 rollout evaluation during training. Returns success rate."""
    from data_pipeline.conversion.compute_norm_stats import load_norm_stats

    if num_episodes is None:
        num_episodes = config.eval_episodes

    # Use EMA model for eval (Chi's approach: separate model, no store/restore)
    if ema_model is not None:
        eval_policy = ema_model.averaged_model
    else:
        eval_policy = _unwrap(policy)
    eval_policy.eval()
    norm_stats = load_norm_stats(config.eval_hdf5)

    if config.eval_mode == "robomimic":
        from training.eval_v3_robomimic import evaluate_v3_robomimic_parallel
        video_dir = os.path.join(config.save_dir, "media", f"epoch_{epoch:04d}")
        success_rate, results = evaluate_v3_robomimic_parallel(
            policy=eval_policy, ema_model=None,
            hdf5_path=config.eval_hdf5,
            norm_stats=norm_stats,
            num_episodes=num_episodes,
            n_envs=min(config.eval_n_envs, num_episodes),
            use_rot6d=config.use_rot6d,
            device=str(device),
            norm_mode=config.norm_mode,
            save_video=save_video,
            video_dir=video_dir,
        )
        n_success = sum(1 for r in results.values() if r["success"])
    elif config.eval_mode == "rlbench":
        from training.eval_v3 import V3PolicyWrapper
        from training.eval_v3_rlbench import evaluate_v3_rlbench
        video_dir = os.path.join(config.save_dir, "media", f"epoch_{epoch:04d}")
        wrapper = V3PolicyWrapper(eval_policy, device=str(device))
        success_rate, results, _run_v3_eval._rlbench_env = evaluate_v3_rlbench(
            wrapper, norm_stats,
            task=config.eval_task,
            num_episodes=num_episodes,
            obs_horizon=config.T_obs,
            save_video=save_video,
            video_dir=video_dir,
            _cached_env=getattr(_run_v3_eval, '_rlbench_env', None),
        )
        n_success = sum(1 for r in results if r["success"])
    elif config.eval_mode == "joint":
        from training.eval_v3 import V3PolicyWrapper
        from training.eval_v3_joint import evaluate_v3_joint
        video_dir = os.path.join(config.save_dir, "media", f"epoch_{epoch:04d}")
        wrapper = V3PolicyWrapper(eval_policy, device=str(device))
        success_rate, results = evaluate_v3_joint(
            wrapper, norm_stats,
            task=config.eval_task,
            num_episodes=num_episodes,
            obs_horizon=config.T_obs,
            exec_horizon=config.eval_exec_horizon,
            save_video=save_video,
            video_dir=video_dir,
        )
        n_success = sum(1 for r in results if r["success"])
    else:
        from training.eval_v3 import V3PolicyWrapper, evaluate_v3
        wrapper = V3PolicyWrapper(_unwrap(policy), ema_model=ema_model, device=str(device))
        success_rate, results = evaluate_v3(
            wrapper, norm_stats,
            num_episodes=num_episodes,
            task=config.eval_task,
            image_size=config.eval_image_size,
            use_rot6d=config.use_rot6d,
        )
        n_success = sum(1 for r in results if r["success"])

    log.info("Eval epoch %d: %d/%d (%.1f%%)",
             epoch, n_success, num_episodes, success_rate * 100)
    return success_rate
