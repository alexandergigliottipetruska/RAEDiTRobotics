"""Stage 3 diffusion policy training loop (C.7) with Optuna HP Tuning.

Trains a DiT-Block Policy on adapted visual tokens from Stage 1 using
DDPM noise prediction. Supports optional co-training with reconstruction
loss (lambda_recon > 0).

Includes a validation loop and DDP-safe Optuna pruning logic.
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
import optuna

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
    ac_dim: int = 7             
    proprio_dim: int = 9        
    num_views: int = 4          
    T_obs: int = 2
    T_pred: int = 16
    T_act: int = 8              
    hidden_dim: int = 512
    num_blocks: int = 6
    nhead: int = 8
    use_lightning: bool = True

    # Diffusion
    train_diffusion_steps: int = 100
    eval_diffusion_steps: int = 10
    policy_type: str = "flow_matching"
    fm_timestep_dist: str = "beta"       
    fm_timestep_scale: float = 1000.0    
    fm_beta_a: float = 1.5              
    fm_beta_b: float = 1.0
    fm_cutoff: float = 0.999            
    num_flow_steps: int = 10            

    # Training
    lr: float = 1e-4
    lr_adapter: float = 1e-5    
    weight_decay: float = 1e-4
    num_epochs: int = 300
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"  

    # Co-training (reconstruction alongside policy)
    lambda_recon: float = 0.0   

    # View dropout & EMA
    p: float = 0.15
    ema_decay: float = 0.9999

    # Logging & checkpointing
    log_every: int = 100
    save_every_epoch: int = 10
    eval_every_epoch: int = 50
    save_dir: str = "checkpoints/stage3"

    # Inline eval video 
    eval_video_task: str = ""           
    eval_video_episodes: int = 1
    eval_video_steps: int = 100         
    eval_video_dir: str = "eval_videos"
    eval_video_hdf5: str = ""           


# ---------------------------------------------------------------------------
# DDPM noise schedule
# ---------------------------------------------------------------------------

def create_noise_scheduler(num_train_steps: int = 100) -> DDIMScheduler:
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
    batch: dict, policy: nn.Module, optimizer: torch.optim.Optimizer,
    config: Stage3Config, global_step: int, use_amp: bool = False,
) -> dict:
    device = next(policy.parameters()).device
    device_type = device.type

    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
        loss = policy(batch_dev)

    optimizer.zero_grad()
    loss.backward()

    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            config.grad_clip,
        )

    optimizer.step()
    return {"policy": loss.item(), "total": loss.item()}


# ---------------------------------------------------------------------------
# Validation step (NEW FOR HP TUNING)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    loader: DataLoader, policy: nn.Module, use_amp: bool = False
) -> dict:
    """Compute validation loss on unseen data without updating weights."""
    device = next(policy.parameters()).device
    device_type = device.type
    policy.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        with torch.amp.autocast(device_type, dtype=torch.bfloat16, enabled=use_amp):
            loss = policy(batch_dev)
            total_loss += loss.item()
        n_batches += 1

    policy.train()
    
    return {
        "val_policy": total_loss / max(n_batches, 1)
    }


import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase

def run_mini_eval(model, config, num_episodes=10):
    """
    Runs a small number of simulator rollouts to calculate Success Rate.
    """
    # 1. Create the environment from the HDF5 metadata
    # We use the first path in your config's hdf5_paths
    original_hdf5 = config.eval_video_hdf5 
    
    # Safety check: if for some reason it's empty, fall back or error out
    if not original_hdf5 or not os.path.exists(original_hdf5):
        log.error(f"Missing eval_video_hdf5: {original_hdf5}")
        return 0.0

    env_meta = FileUtils.get_env_metadata_from_dataset(original_hdf5)

    # Force headless rendering for the lab PCs
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=True, 
        use_image_obs=True
    )

    model.eval()
    success_count = 0
    
    for _ in range(num_episodes):
        obs = env.reset()
        goal = None # Lift is typically not goal-conditioned
        done = False
        
        # 'horizon' is usually 400 for Lift
        for _ in range(400): 
            with torch.no_grad():
                # Ensure observations are on the right device and have batch dim
                action = _unwrap(model).predict_action(obs)
                
            obs, reward, done, info = env.step(action)
            
            # robomimic environments have an is_success() check
            if env.is_success()["task"]:
                success_count += 1
                break
            if done:
                break
                
    env.close()
    return success_count / num_episodes

# ---------------------------------------------------------------------------
# LR scheduler with linear warmup
# ---------------------------------------------------------------------------

def _create_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, schedule: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        if schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str, epoch: int, global_step: int, noise_net: nn.Module, adapter: nn.Module,
    optimizer: torch.optim.Optimizer, ema: EMA | None, val_metrics: dict, decoder: nn.Module | None = None,
):
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
    torch.save(ckpt, path)
    log.info("Saved checkpoint: %s (epoch %d, step %d)", path, epoch, global_step)


def load_checkpoint(
    path: str, noise_net: nn.Module, adapter: nn.Module, optimizer: torch.optim.Optimizer,
    ema: EMA | None = None, decoder: nn.Module | None = None,
) -> tuple[int, int]:
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

    log.info("Loaded checkpoint from epoch %d, step %d: %s", ckpt["epoch"], ckpt["global_step"], path)
    return ckpt["epoch"] + 1, ckpt["global_step"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_stage3(
    config: Stage3Config,
    *,
    device: torch.device | str = "cuda",
    resume_from: str | None = None,
    trial=None # ADDED FOR OPTUNA
):
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
        use_lightning=config.use_lightning,
        policy_type=config.policy_type,
        fm_timestep_dist=config.fm_timestep_dist,
        fm_timestep_scale=config.fm_timestep_scale,
        fm_beta_a=config.fm_beta_a,
        fm_beta_b=config.fm_beta_b,
        fm_cutoff=config.fm_cutoff,
        num_flow_steps=config.num_flow_steps,
    )
    policy = policy.to(device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    ema = EMA(policy.noise_net, decay=config.ema_decay)

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
        param_groups, betas=(0.9, 0.999), weight_decay=config.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if resume_from and os.path.isfile(resume_from):
        start_epoch, global_step = load_checkpoint(
            resume_from, policy.noise_net, policy.bridge.adapter,
            optimizer, ema, policy.bridge.decoder,
        )

    if device.type == "cuda" and not distributed:
        policy = torch.compile(policy)
        log.info("torch.compile enabled")

    if distributed:
        policy = nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank],
            find_unused_parameters=(config.lambda_recon > 0),
        )

    train_ds = Stage3Dataset(
        config.hdf5_paths, split="train",
        T_obs=config.T_obs, T_pred=config.T_pred, norm_mode=config.norm_mode,
    )
    valid_ds = Stage3Dataset(
        config.hdf5_paths, split="valid",
        T_obs=config.T_obs, T_pred=config.T_pred, norm_mode=config.norm_mode,
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    persistent = config.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=persistent, prefetch_factor=3 if config.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=persistent,
    )

    total_steps = config.num_epochs * len(train_loader)
    lr_scheduler = _create_lr_scheduler(
        optimizer, config.warmup_steps, total_steps, schedule=config.lr_schedule,
    )

    if global_step > 0:
        for _ in range(global_step):
            lr_scheduler.step()

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
        
        # Omitted run_info write block for brevity here, but keep your existing one if you wish.
        log.info("=" * 60)
        log.info("Stage 3 DiT Policy Training (Optuna Ready)")
        log.info("=" * 60)

    # --- Training loop ---
    best_val_loss = float("inf")
    best_epoch_loss = -1

    max_sr_found = -1.0
    best_epoch_sr = -1

    last_sim_epoch = -1
    for epoch in range(start_epoch, config.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        policy.train()
        epoch_losses: dict[str, float] = {}
        n_steps = 0

        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True) if is_main else train_loader

        for batch in loader_iter:
            step_losses = train_step(
                batch, policy, optimizer, config, global_step, use_amp=use_amp,
            )
            ema.update()
            lr_scheduler.step()

            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            n_steps += 1
            global_step += 1

        avg = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}

        # --- 1. SETUP SHARED VARIABLES ---
        val = None
        should_stop = torch.tensor(0).to(device)

        # --- 2. RANK 0 LOGIC (Validate, Log, Save, Report) ---
        if is_main:
            # ACTUALLY RUN VALIDATION
            val = validate(valid_loader, policy, use_amp=use_amp)

            pu = _unwrap(policy)
            
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(avg.items()))
            val_str = " | ".join(f"{k}={v:.4f}" for k, v in sorted(val.items()))
            
            log.info("Epoch %d  Train: %s  ||  Val: %s  (lr=%.2e)", 
                     epoch, train_str, val_str, optimizer.param_groups[0]["lr"])

            if metrics_path:
                with open(metrics_path, "a") as mf:
                    mf.write(json.dumps({
                        "epoch": epoch, "global_step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        "train": avg, "val": val,
                    }) + "\n")

            is_best = val["val_policy"] < best_val_loss

            # Checkpointing
            if is_best:
                best_val_loss = val["val_policy"]
                best_epoch_loss = epoch

                save_checkpoint(
                    os.path.join(config.save_dir, "best_loss.pt"),
                    epoch, global_step, pu.noise_net, pu.bridge.adapter,
                    optimizer, ema, val, decoder=pu.bridge.decoder,
                )

                # Sync to Optuna Table
                trial.set_user_attr("best_loss", float(best_val_loss))
                trial.set_user_attr("best_epoch_loss", int(best_epoch_loss))

            # OPTUNA PRUNING & EXTRA METRICS CHECK 
            if trial is not None:
                # 1. Report the main objective (for the learning curve and pruning)
                trial.report(val["val_policy"], epoch)
                
                # 2. Record additional metrics to the Optuna Dashboard ONLY when we hit a new best
                #    This creates new columns in the dashboard UI!
                if is_best or (epoch > 0 and epoch % config.save_every_epoch == 0):
                    if epoch - last_sim_epoch >= 10:
                        last_sim_epoch = epoch
                        success_rate = run_mini_eval(policy, config, num_episodes=10)

                        trial.set_user_attr("current_success_rate", success_rate)
                        trial.set_user_attr("policy_type", config.policy_type) # Helpful for sorting FM vs DDPM

                        if success_rate > max_sr_found:
                            max_sr_found = success_rate
                            best_epoch_sr = epoch

                            # These are the columns that stay pinned to the peak performance
                            trial.set_user_attr("best_success_rate", float(max_sr_found))
                            trial.set_user_attr("best_epoch_success_rate", epoch)

                            save_checkpoint(os.path.join(config.save_dir, "best_success_rate.pt"), 
                                            epoch, global_step, pu.noise_net, pu.bridge.adapter,
                                            optimizer, ema, val, decoder=pu.bridge.decoder,)
                        
                        # Dynamically unpack all train/val dictionary items as UI columns
                        for k, v in avg.items():
                            trial.set_user_attr(f"train_{k}", float(v))
                        for k, v in val.items():
                            if k != "val_policy": # Skip the main objective since it's already tracked
                                trial.set_user_attr(f"{k}", float(v))

                # 3. Check if the trial is a dud
                if trial.should_prune():
                    log.info(f"🚩 Trial pruned by Optuna at epoch {epoch}")
                    should_stop += 1 
                    log.removeHandler(fh)
                    fh.close()

        # --- 3. DDP BROADCAST (Sync workers) ---
        if distributed:
            torch.distributed.broadcast(should_stop, src=0)

        # --- 4. SAFE EXIT FOR ALL RANKS ---
        if should_stop > 0:
            if distributed:
                torch.distributed.destroy_process_group() 
            if is_main and trial is not None:
                raise optuna.exceptions.TrialPruned() 
            else:
                return best_val_loss 

        # Sync all ranks before next epoch
        if distributed:
            torch.distributed.barrier()

    if is_main:
        log.info("Training complete. Best val_policy=%.4f", best_val_loss)
        log.removeHandler(fh)
        fh.close()

    return best_val_loss