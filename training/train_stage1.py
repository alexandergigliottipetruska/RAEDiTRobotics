"""Phased RAE training loop (Phase A.6).

Implements the three-phase training schedule from Zheng et al. 2025:
  Phase 1 (epochs 0-E_d): L1 + LPIPS only
  Phase 2 (epochs E_d-E_g): + discriminator training
  Phase 3 (epochs E_g-end): + GAN loss for decoder

Separate optimizers for decoder/adapter vs discriminator head.
Hyperparameters from Zheng et al. Table 12.

Supports single-GPU, multi-GPU (DataParallel), and multi-node
(DistributedDataParallel via torchrun). Single-GPU is the default;
DDP activates automatically when launched with torchrun.

Expected component interfaces (filled in by A.1-A.3):
  encoder(x)  ->  tokens       x: (B, 3, 224, 224) -> (B, N, d)
  adapter(z)  ->  adapted      z: (B, N, d) -> (B, N, d')
  adapter.noise_augment(z)     z: (B, N, d') -> (B, N, d')  [training only]
  decoder(z)  ->  images       z: (B, N, d') -> (B, 3, 224, 224) in [0, 1]
  decoder.last_layer_weight    nn.Parameter for adaptive lambda
"""

import contextlib
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

from data_pipeline.datasets.stage1_dataset import Stage1Dataset
from models.losses import (
    l1_loss,
    lpips_loss_fn,
    gan_generator_loss,
    gan_discriminator_loss,
    compute_adaptive_lambda,
    create_lpips_net,
)
from models.discriminator import PatchDiscriminator

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
    # torch.compile wraps the original module as _orig_mod
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Stage1Config:
    """Hyperparameters for Stage 1 RAE training.

    Defaults from Zheng et al. 2025 Table 12, scaled for robotics.
    """
    # Data — accepts single path (str) or list of paths for multi-task
    hdf5_path: str = ""         # single file (backward compat)
    hdf5_paths: list = field(default_factory=list)  # multi-file
    batch_size: int = 32
    num_workers: int = 4

    # Training schedule (epoch-based phases)
    num_epochs: int = 50
    epoch_start_disc: int = 6   # E_d: start discriminator training
    epoch_start_gan: int = 8    # E_g: add GAN loss to generator

    # Loss weights (Zheng et al. Table 12)
    omega_L: float = 1.0       # LPIPS weight
    omega_G: float = 0.75      # GAN weight (before adaptive lambda)

    # Optimizer (AdamW, betas from Zheng et al.)
    lr_gen: float = 1e-4       # LR for adapter + decoder
    lr_disc: float = 1e-4      # LR for discriminator head
    betas: tuple = (0.5, 0.9)
    weight_decay: float = 0.01
    grad_accum_steps: int = 1  # gradient accumulation steps

    # Checkpointing
    save_every: int = 5
    save_dir: str = "checkpoints/stage1"

    # Model
    disc_pretrained: bool = True


# ---------------------------------------------------------------------------
# Discriminator helper (gen step needs gradient flow)
# ---------------------------------------------------------------------------

def disc_forward_with_grad(
    disc: PatchDiscriminator, x: torch.Tensor
) -> torch.Tensor:
    """Discriminator forward allowing gradient flow for adaptive lambda.

    disc.forward() wraps backbone in torch.no_grad(), severing the
    computation graph from logits_fake back to decoder weights. This
    blocks torch.autograd.grad(L_gan, last_layer_weight) needed by
    compute_adaptive_lambda.

    Fix: call backbone + head separately. backbone params have
    requires_grad=False, preventing weight updates while allowing
    gradient flow through inputs.

    Use for the GENERATOR step only. The disc step uses disc() normally.
    """
    raw = _unwrap(disc)
    feat = raw.backbone(x)
    return raw.head(feat)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(
    batch: dict,
    encoder: nn.Module,
    adapter: nn.Module,
    decoder: nn.Module,
    disc: PatchDiscriminator,
    lpips_net,
    opt_gen: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
    epoch: int,
    config: Stage1Config,
    use_amp: bool = False,
    loss_scale: float = 1.0,
    step_optimizers: bool = True,
) -> dict:
    """One training micro-step: forward + backward, optionally step optimizers.

    For gradient accumulation, the caller sets loss_scale=1/accum_steps and
    step_optimizers=False for intermediate micro-batches, then calls with
    step_optimizers=True on the final micro-batch.

    Args:
        loss_scale:       Multiply losses by this before backward (1/accum_steps).
        step_optimizers:  If True, call opt.step() + opt.zero_grad() after backward.

    Returns dict of scalar loss values for logging (unscaled).
    """
    images_enc = batch["images_enc"]        # (B, K, 3, H, W)
    images_target = batch["images_target"]  # (B, K, 3, H, W)
    view_present = batch["view_present"]    # (B, K)

    B, K = view_present.shape

    # Flatten views and select real ones
    mask = view_present.reshape(-1)                       # (B*K,)
    real_enc = images_enc.reshape(B * K, 3, 224, 224)[mask]    # (N, 3, H, W)
    real_tgt = images_target.reshape(B * K, 3, 224, 224)[mask] # (N, 3, H, W)

    device_type = real_enc.device.type

    # ---- Forward: encoder (frozen) -> adapter -> decoder ----
    with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
        with torch.no_grad():
            tokens = encoder(real_enc)          # (N, num_patches, d)

        adapted = adapter(tokens)               # (N, num_patches, d')

        raw_adapter = _unwrap(adapter)
        if hasattr(raw_adapter, "noise_augment"):
            adapted = raw_adapter.noise_augment(adapted)

        pred = decoder(adapted)                 # (N, 3, 224, 224) in [0, 1]

        # ---- Generator loss ----
        L_l1 = l1_loss(pred, real_tgt)
        L_lpips = lpips_loss_fn(pred, real_tgt, lpips_net)
        L_rec = L_l1 + config.omega_L * L_lpips

        use_gan = epoch >= config.epoch_start_gan
        if use_gan:
            logits_fake = disc_forward_with_grad(disc, pred)
            L_gan = gan_generator_loss(logits_fake)
            lam = compute_adaptive_lambda(L_rec, L_gan, _unwrap(decoder).last_layer_weight)
            L_total = L_rec + config.omega_G * lam * L_gan
        else:
            L_total = L_rec

    losses = {
        "l1": L_l1.item(),
        "lpips": L_lpips.item(),
        "rec": L_rec.item(),
    }
    if use_gan:
        losses["gan_gen"] = L_gan.item()
        losses["lambda"] = lam.item()

    losses["total_gen"] = L_total.item()

    # Scale loss for gradient accumulation (backward accumulates into .grad)
    (L_total * loss_scale).backward()

    if step_optimizers:
        opt_gen.step()
        opt_gen.zero_grad()

    # ---- Discriminator step (Phase 2+) ----
    if epoch >= config.epoch_start_disc:
        with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
            logits_real = disc(real_tgt.detach())
            logits_fake_d = disc(pred.detach())
            L_disc = gan_discriminator_loss(logits_real, logits_fake_d)

        (L_disc * loss_scale).backward()

        if step_optimizers:
            opt_disc.step()
            opt_disc.zero_grad()

        losses["disc"] = L_disc.item()

    return losses


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    loader: DataLoader,
    encoder: nn.Module,
    adapter: nn.Module,
    decoder: nn.Module,
    lpips_net,
    use_amp: bool = False,
) -> dict:
    """Compute validation L1 + LPIPS (no GAN, no noise augment)."""
    device = next(decoder.parameters()).device
    device_type = device.type
    adapter.eval()
    decoder.eval()

    total_l1 = 0.0
    total_lpips = 0.0
    n_batches = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        images_enc = batch["images_enc"]
        images_target = batch["images_target"]
        view_present = batch["view_present"]

        B, K = view_present.shape
        mask = view_present.reshape(-1)
        real_enc = images_enc.reshape(B * K, 3, 224, 224)[mask]
        real_tgt = images_target.reshape(B * K, 3, 224, 224)[mask]

        with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
            tokens = encoder(real_enc)
            adapted = adapter(tokens)
            pred = decoder(adapted)

            total_l1 += l1_loss(pred, real_tgt).item()
            total_lpips += lpips_loss_fn(pred, real_tgt, lpips_net).item()
        n_batches += 1

    adapter.train()
    decoder.train()

    return {
        "val_l1": total_l1 / max(n_batches, 1),
        "val_lpips": total_lpips / max(n_batches, 1),
        "val_rec": (total_l1 + total_lpips) / max(n_batches, 1),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    epoch: int,
    adapter: nn.Module,
    decoder: nn.Module,
    disc: PatchDiscriminator,
    opt_gen: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
    val_metrics: dict,
):
    """Save training checkpoint (adapter + decoder + disc head + optimizers).

    Automatically unwraps DDP/DataParallel wrappers so checkpoints
    are portable across single-GPU and distributed setups.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "adapter": _unwrap(adapter).state_dict(),
            "decoder": _unwrap(decoder).state_dict(),
            "disc_head": _unwrap(disc).head.state_dict(),
            "opt_gen": opt_gen.state_dict(),
            "opt_disc": opt_disc.state_dict(),
            "val_metrics": val_metrics,
        },
        path,
    )
    log.info("Saved checkpoint: %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str,
    adapter: nn.Module,
    decoder: nn.Module,
    disc: PatchDiscriminator,
    opt_gen: torch.optim.Optimizer,
    opt_disc: torch.optim.Optimizer,
) -> int:
    """Load checkpoint. Returns the epoch to resume from (saved_epoch + 1)."""
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    def _strip_compile_prefix(state_dict):
        """Remove '_orig_mod.' prefix left by torch.compile."""
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in state_dict):
            return {k.removeprefix(prefix): v for k, v in state_dict.items()}
        return state_dict

    adapter.load_state_dict(_strip_compile_prefix(ckpt["adapter"]))
    decoder.load_state_dict(_strip_compile_prefix(ckpt["decoder"]))
    disc.head.load_state_dict(_strip_compile_prefix(ckpt["disc_head"]))
    opt_gen.load_state_dict(ckpt["opt_gen"])
    opt_disc.load_state_dict(ckpt["opt_disc"])
    log.info("Loaded checkpoint from epoch %d: %s", ckpt["epoch"], path)
    return ckpt["epoch"] + 1


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_stage1(
    config: Stage1Config,
    *,
    encoder: nn.Module,
    adapter: nn.Module,
    decoder: nn.Module,
    device: torch.device | str = "cuda",
    resume_from: str | None = None,
):
    """Main Stage 1 training entry point.

    Supports single-GPU (default), multi-GPU DataParallel, and multi-node
    DistributedDataParallel (via torchrun). DDP is auto-detected — no flag
    needed. The notebook and script both call this function identically.

    Args:
        config:      Training configuration.
        encoder:     Frozen encoder (A.1).
        adapter:     Trainable adapter (A.2).
        decoder:     Trainable decoder (A.3). Must expose `last_layer_weight`.
        device:      Device to train on (ignored under DDP; uses LOCAL_RANK).
        resume_from: Path to checkpoint to resume from.
    """
    distributed = _is_distributed()
    rank = _rank()
    is_main = _is_main()

    # Under DDP, device is set from LOCAL_RANK
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(device)

    use_amp = (device.type == "cuda")

    # Move models to device
    encoder = encoder.to(device).eval()
    adapter = adapter.to(device).train()
    decoder = decoder.to(device).train()

    disc = PatchDiscriminator(pretrained=config.disc_pretrained).to(device)
    disc.train()  # backbone stays eval via overridden .train()

    lpips_net = create_lpips_net().to(device)

    # Load checkpoint BEFORE wrapping in DDP (state_dicts are unwrapped)
    start_epoch = 0
    if resume_from and os.path.isfile(resume_from):
        # Create temporary optimizers for checkpoint loading
        gen_params = list(adapter.parameters()) + list(decoder.parameters())
        _opt_gen = torch.optim.AdamW(gen_params, lr=config.lr_gen,
                                     betas=config.betas, weight_decay=config.weight_decay)
        _opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=config.lr_disc,
                                      betas=config.betas, weight_decay=config.weight_decay)
        start_epoch = load_checkpoint(
            resume_from, adapter, decoder, disc, _opt_gen, _opt_disc
        )

    # Wrap trainable models in DDP (after checkpoint load, before optimizer creation)
    if distributed:
        adapter = nn.parallel.DistributedDataParallel(adapter, device_ids=[local_rank])
        decoder = nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank])
        disc = nn.parallel.DistributedDataParallel(disc, device_ids=[local_rank])
        if is_main:
            log.info("DDP enabled: %d GPUs across %d nodes", _world_size(), _world_size())

    # Dataloaders — resolve hdf5_paths (multi-file) vs hdf5_path (single)
    paths = config.hdf5_paths if config.hdf5_paths else [config.hdf5_path]
    train_ds = Stage1Dataset(paths, split="train")
    valid_ds = Stage1Dataset(paths, split="valid")

    # DistributedSampler splits data across ranks; shuffle is handled by sampler
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),  # only shuffle if no sampler
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Optimizers: adapter+decoder share one, disc head separate
    gen_params = list(adapter.parameters()) + list(decoder.parameters())
    opt_gen = torch.optim.AdamW(
        gen_params,
        lr=config.lr_gen,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )
    opt_disc = torch.optim.AdamW(
        _unwrap(disc).head.parameters(),
        lr=config.lr_disc,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    # If we loaded a checkpoint, reload optimizer states into the new optimizers
    if resume_from and os.path.isfile(resume_from) and start_epoch > 0:
        opt_gen.load_state_dict(_opt_gen.state_dict())
        opt_disc.load_state_dict(_opt_disc.state_dict())

    # --- Logging & metrics (rank 0 only) ---
    metrics_path = None
    if is_main:
        os.makedirs(config.save_dir, exist_ok=True)

        # Timestamped log file
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(config.save_dir, f"train_{ts}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        log.addHandler(fh)

        # Metrics JSONL (append-friendly, one line per epoch)
        metrics_path = os.path.join(config.save_dir, f"metrics_{ts}.jsonl")

        # Write run header as first line
        run_info = {
            "type": "run_info",
            "timestamp": ts,
            "gpu": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
            "vram_gb": round(torch.cuda.get_device_properties(device).total_memory / 1e9, 1) if torch.cuda.is_available() else 0,
            "use_amp": use_amp,
            "distributed": distributed,
            "world_size": _world_size(),
            "resume_from": resume_from,
            "start_epoch": start_epoch,
            "config": {
                "hdf5_paths": paths,
                "batch_size": config.batch_size,
                "num_workers": config.num_workers,
                "num_epochs": config.num_epochs,
                "epoch_start_disc": config.epoch_start_disc,
                "epoch_start_gan": config.epoch_start_gan,
                "omega_L": config.omega_L,
                "omega_G": config.omega_G,
                "lr_gen": config.lr_gen,
                "lr_disc": config.lr_disc,
                "betas": list(config.betas),
                "weight_decay": config.weight_decay,
                "disc_pretrained": config.disc_pretrained,
                "grad_accum_steps": config.grad_accum_steps,
            },
        }
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(run_info) + "\n")

        # Log run info
        log.info("=" * 60)
        log.info("Stage 1 RAE Training Run")
        log.info("=" * 60)

        # Hardware
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device)
            vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            log.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
        else:
            log.info("Device: CPU")
        if use_amp:
            log.info("BF16 mixed precision enabled")
        if distributed:
            log.info("DDP: %d GPUs across %d nodes", _world_size(), _world_size())

        # Config
        eff_batch = config.batch_size * config.grad_accum_steps * _world_size()
        log.info("Config: batch_size=%d, grad_accum=%d, effective_batch=%d, num_workers=%d, lr_gen=%.1e, lr_disc=%.1e",
                 config.batch_size, config.grad_accum_steps, eff_batch, config.num_workers, config.lr_gen, config.lr_disc)
        log.info("Config: omega_L=%.2f, omega_G=%.2f, weight_decay=%.4f, betas=%s",
                 config.omega_L, config.omega_G, config.weight_decay, config.betas)
        log.info("Config: disc_pretrained=%s", config.disc_pretrained)

        # Data & schedule
        log.info(
            "Data: epochs %d-%d, %d train / %d valid samples",
            start_epoch, config.num_epochs - 1, len(train_ds), len(valid_ds),
        )
        log.info(
            "Phase schedule: disc @ epoch %d, GAN @ epoch %d",
            config.epoch_start_disc, config.epoch_start_gan,
        )
        if resume_from:
            log.info("Resumed from: %s (epoch %d)", resume_from, start_epoch)

        log.info("Log file: %s", log_file)
        log.info("Metrics file: %s", metrics_path)
        log.info("=" * 60)

    best_val_rec = float("inf")

    for epoch in range(start_epoch, config.num_epochs):
        # DistributedSampler must know the epoch for proper shuffling
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Phase label for logging
        if epoch < config.epoch_start_disc:
            phase = "phase1_rec"
        elif epoch < config.epoch_start_gan:
            phase = "phase2_disc"
        else:
            phase = "phase3_gan"

        adapter.train()
        decoder.train()

        epoch_losses: dict[str, float] = {}
        n_steps = 0
        accum = config.grad_accum_steps

        # Zero grads at start of epoch
        opt_gen.zero_grad()
        opt_disc.zero_grad()

        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch} [{phase}]", leave=True) if is_main else train_loader
        for step_idx, batch in enumerate(loader_iter):
            batch = {k: v.to(device) for k, v in batch.items()}

            is_accum_step = (step_idx + 1) % accum == 0

            # In DDP, skip AllReduce on intermediate accumulation steps
            if distributed and not is_accum_step:
                ctx_a = adapter.no_sync()
                ctx_d = decoder.no_sync()
                ctx_disc = disc.no_sync()
            else:
                ctx_a = contextlib.nullcontext()
                ctx_d = contextlib.nullcontext()
                ctx_disc = contextlib.nullcontext()

            with ctx_a, ctx_d, ctx_disc:
                step_losses = train_step(
                    batch, encoder, adapter, decoder, disc, lpips_net,
                    opt_gen, opt_disc, epoch, config, use_amp=use_amp,
                    loss_scale=1.0 / accum,
                    step_optimizers=is_accum_step,
                )

            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_steps += 1

        # Handle leftover micro-batches that didn't complete an accumulation window
        if n_steps % accum != 0:
            opt_gen.step()
            opt_gen.zero_grad()
            opt_disc.step()
            opt_disc.zero_grad()

        # Average epoch losses
        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

        # Validation + logging + checkpointing only on rank 0
        if is_main:
            val = validate(valid_loader, encoder, adapter, decoder, lpips_net, use_amp=use_amp)

            train_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items()))
            val_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(val.items()))
            log.info("Epoch %d [%s]  %s  ||  %s", epoch, phase, train_str, val_str)

            # Append metrics for plotting
            if metrics_path:
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({
                        "epoch": epoch, "phase": phase,
                        "train": avg, "val": val,
                    }) + "\n")

            if val["val_rec"] < best_val_rec:
                best_val_rec = val["val_rec"]
                save_checkpoint(
                    os.path.join(config.save_dir, "best.pt"),
                    epoch, adapter, decoder, disc, opt_gen, opt_disc, val,
                )

            if (epoch + 1) % config.save_every == 0:
                save_checkpoint(
                    os.path.join(config.save_dir, f"epoch_{epoch:03d}.pt"),
                    epoch, adapter, decoder, disc, opt_gen, opt_disc, val,
                )

        # Sync all ranks before next epoch
        if distributed:
            torch.distributed.barrier()

    if is_main:
        log.info("Training complete. Best val_rec=%.4f", best_val_rec)
        log.removeHandler(fh)
        fh.close()
