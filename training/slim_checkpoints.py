"""Re-save V3 checkpoints with component-wise EMA (strips frozen encoder).

The old EMA format stores the full model state_dict including the frozen
DINOv3-L encoder (303M params, ~1.2 GB). The new format stores only the
trainable components (denoiser, obs_encoder, adapter, decoder).

Usage:
    # Dry run (report sizes, don't modify):
    python training/slim_checkpoints.py checkpoints/v3_can_d512_*/best_success.pt --dry-run

    # Slim a single checkpoint:
    python training/slim_checkpoints.py checkpoints/v3_can_d512_seed42/best_success.pt

    # Slim all checkpoints:
    python training/slim_checkpoints.py checkpoints/v3_*/best_success.pt checkpoints/v3_*/rolling_*.pt
"""

import argparse
import os
import sys
import torch


def strip_prefix(sd):
    prefix = "_orig_mod."
    if any(k.startswith(prefix) for k in sd):
        return {k.removeprefix(prefix): v for k, v in sd.items()}
    return sd


def extract_component(full_sd, component_prefix):
    """Extract keys matching a component prefix and strip the prefix."""
    prefix = component_prefix + "."
    return {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}


def slim_checkpoint(path, dry_run=False):
    """Convert old-format EMA (full model) to new format (components only)."""
    orig_size = os.path.getsize(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "ema" not in ckpt:
        print(f"  SKIP (no EMA): {path}")
        return 0

    ema = ckpt["ema"]
    if "averaged_model" not in ema:
        print(f"  SKIP (already new format): {path}")
        return 0

    full_sd = ema["averaged_model"]
    n_keys = len(full_sd)

    # Extract trainable components from the full state dict
    denoiser_sd = strip_prefix(extract_component(full_sd, "denoiser"))
    obs_encoder_sd = strip_prefix(extract_component(full_sd, "obs_encoder"))
    adapter_sd = strip_prefix(extract_component(full_sd, "bridge.adapter"))
    decoder_sd = strip_prefix(extract_component(full_sd, "bridge.decoder"))

    n_kept = len(denoiser_sd) + len(obs_encoder_sd) + len(adapter_sd) + len(decoder_sd)
    n_dropped = n_keys - n_kept

    # Build new EMA dict
    new_ema = {
        "denoiser": denoiser_sd,
        "obs_encoder": obs_encoder_sd,
        "adapter": adapter_sd,
        "optimization_step": ema.get("optimization_step", 0),
        "decay": ema.get("decay", 0.0),
    }
    if decoder_sd:
        new_ema["decoder"] = decoder_sd

    if dry_run:
        # Estimate savings
        dropped_bytes = sum(v.numel() * v.element_size() for k, v in full_sd.items()
                           if not any(k.startswith(p) for p in
                                      ["denoiser.", "obs_encoder.", "bridge.adapter.", "bridge.decoder."]))
        print(f"  DRY RUN: {path}")
        print(f"    Current size: {orig_size / 1e9:.2f} GB")
        print(f"    EMA keys: {n_keys} total, {n_kept} kept, {n_dropped} dropped (frozen encoder)")
        print(f"    Estimated savings: {dropped_bytes / 1e9:.2f} GB")
        return dropped_bytes

    # Replace and save
    ckpt["ema"] = new_ema
    torch.save(ckpt, path)
    new_size = os.path.getsize(path)
    saved = orig_size - new_size

    print(f"  SLIMMED: {path}")
    print(f"    {orig_size / 1e9:.2f} GB -> {new_size / 1e9:.2f} GB (saved {saved / 1e9:.2f} GB)")
    print(f"    EMA keys: {n_keys} -> {n_kept} (dropped {n_dropped} frozen encoder keys)")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Slim V3 checkpoints by removing frozen encoder from EMA")
    parser.add_argument("files", nargs="+", help="Checkpoint .pt files to process")
    parser.add_argument("--dry-run", action="store_true", help="Report sizes without modifying files")
    args = parser.parse_args()

    total_saved = 0
    n_processed = 0

    for path in args.files:
        if not os.path.exists(path):
            print(f"  NOT FOUND: {path}")
            continue
        if not path.endswith(".pt"):
            continue
        saved = slim_checkpoint(path, dry_run=args.dry_run)
        total_saved += saved
        n_processed += 1

    print(f"\n{'DRY RUN ' if args.dry_run else ''}Summary: {n_processed} files, {total_saved / 1e9:.2f} GB {'would be ' if args.dry_run else ''}saved")


if __name__ == "__main__":
    main()
