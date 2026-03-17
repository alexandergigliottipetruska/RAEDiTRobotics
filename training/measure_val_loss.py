"""Measure validation loss for a Stage 3 checkpoint.

Diagnoses training loss discrepancy by:
1. Seeded loss (reproducible across runs)
2. Per-timestep loss breakdown (which timesteps are easy/hard?)
3. Comparison with a random (untrained) model

Usage:
  python training/measure_val_loss.py \
    --checkpoint checkpoints/stage3_lift_2k/epoch_1799.pt \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --hdf5 data/robomimic/lift/ph_tokens_fast.hdf5
"""

import argparse
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.datasets.stage3_dataset import Stage3Dataset
from torch.utils.data import DataLoader
from models.policy_dit import PolicyDiT
from models.stage1_bridge import Stage1Bridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def compute_loss_at_timestep(policy, batch_dev, timestep_val):
    """Compute DDPM loss at a specific fixed timestep."""
    proprio = batch_dev["proprio"]
    actions = batch_dev["actions"]
    view_present = batch_dev["view_present"]

    obs_tokens, _, _ = policy._encode_and_assemble(
        batch_dev, proprio, view_present
    )

    noise = torch.randn_like(actions)
    timesteps = torch.full(
        (actions.shape[0],), timestep_val, device=actions.device, dtype=torch.long
    )
    noisy_actions = policy.scheduler.add_noise(actions, noise, timesteps)
    _, eps_pred = policy.noise_net(noisy_actions, timesteps, obs_tokens)
    return F.mse_loss(eps_pred, noise)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--hdf5", required=True, nargs="+")
    parser.add_argument("--policy_type", default="ddpm", choices=["ddpm", "flow_matching"])
    parser.add_argument("--batch_size", type=int, default=44)
    parser.add_argument("--norm_mode", default="minmax")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_seeded", action="store_true", help="Skip Test 1 (slow seeded loss)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load Stage 1 bridge
    bridge = Stage1Bridge(
        checkpoint_path=args.stage1_checkpoint,
        pretrained_encoder=True,
        load_decoder=False,
    )

    # Create policy (p_view_drop=0 to match training and eliminate train/eval difference)
    policy = PolicyDiT(
        bridge=bridge,
        ac_dim=7,
        proprio_dim=9,
        policy_type=args.policy_type,
        p_view_drop=0.0,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    log.info("Loading checkpoint from epoch %d, step %d", ckpt["epoch"], ckpt["global_step"])

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

    # Create datasets
    train_ds = Stage3Dataset(args.hdf5, split="train", T_obs=2, T_pred=16, norm_mode=args.norm_mode)
    valid_ds = Stage3Dataset(args.hdf5, split="valid", T_obs=2, T_pred=16, norm_mode=args.norm_mode)
    log.info("Train: %d samples, Valid: %d samples", len(train_ds), len(valid_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Test 1: Seeded loss (reproducible) ----
    if args.skip_seeded:
        log.info("Skipping Test 1 (--skip_seeded)")
    else:
        log.info("=" * 60)
        log.info("TEST 1: Seeded random loss (should be reproducible)")
        log.info("=" * 60)
        for run in range(2):
            for name, loader in [("train", train_loader), ("valid", valid_loader)]:
                torch.manual_seed(42)
                total_loss = 0.0
                n = 0
                with torch.no_grad():
                    for batch in loader:
                        batch_dev = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                            loss = policy(batch_dev)
                        total_loss += loss.item()
                        n += 1
                avg = total_loss / max(n, 1)
                log.info("  Run %d | %s loss: %.6f", run, name, avg)

    # ---- Test 2: Per-timestep loss (first 5 train batches) ----
    log.info("=" * 60)
    log.info("TEST 2: Per-timestep loss breakdown (trained model)")
    log.info("=" * 60)
    # Grab a batch
    test_batch = next(iter(train_loader))
    test_batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in test_batch.items()
    }
    timesteps_to_test = [0, 10, 25, 50, 75, 99]
    with torch.no_grad():
        for t_val in timesteps_to_test:
            torch.manual_seed(42)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                loss = compute_loss_at_timestep(policy, test_batch_dev, t_val)
            log.info("  timestep=%2d | loss=%.6f", t_val, loss.item())

    # ---- Test 3: Compare with random (untrained) model ----
    log.info("=" * 60)
    log.info("TEST 3: Random (untrained) model for comparison")
    log.info("=" * 60)
    bridge_rand = Stage1Bridge(
        checkpoint_path=args.stage1_checkpoint,
        pretrained_encoder=True,
        load_decoder=False,
    )
    policy_rand = PolicyDiT(
        bridge=bridge_rand, ac_dim=7, proprio_dim=9,
        policy_type=args.policy_type, p_view_drop=0.0,
    )
    policy_rand.to(device)
    policy_rand.eval()

    with torch.no_grad():
        for t_val in timesteps_to_test:
            torch.manual_seed(42)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                loss = compute_loss_at_timestep(policy_rand, test_batch_dev, t_val)
            log.info("  timestep=%2d | loss=%.6f (random)", t_val, loss.item())

    # ---- Test 4: Train vs Eval mode should match with dropout=0 ----
    log.info("=" * 60)
    log.info("TEST 4: Train vs Eval mode (should match if dropout=0)")
    log.info("=" * 60)
    for mode in ["eval", "train"]:
        if mode == "eval":
            policy.eval()
        else:
            policy.train()
        torch.manual_seed(42)
        test_batch2 = next(iter(train_loader))
        test_batch2_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in test_batch2.items()
        }
        with torch.no_grad():
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                loss = policy(test_batch2_dev)
        log.info("  [%s mode] loss=%.6f", mode, loss.item())

    log.info("Done.")


if __name__ == "__main__":
    main()
