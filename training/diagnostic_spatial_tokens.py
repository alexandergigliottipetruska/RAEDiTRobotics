"""Diagnostic tests: does the Stage 1 adapter preserve DINOv3 spatial info?

Three tests from Plan B (Spatial Tokens) document:
  1. RAE Decoder Reconstruction Ablation — full vs pooled vs 4×4 tokens
  2. Linear Depth Probe — predict per-patch depth from adapter tokens
  3. Cosine Similarity + PCA — visualize spatial coherence

All tests use only Stage 1 artifacts (frozen encoder, trained adapter + decoder).
No Stage 3 policy or diffusion training required.

Usage:
    PYTHONPATH=. python training/diagnostic_spatial_tokens.py --pretrained

Owner: Swagman
"""

import os, sys, argparse, logging
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.encoder import FrozenMultiViewEncoder
from models.adapter import TrainableAdapter
from models.decoder import ViTDecoder
from models.losses import create_lpips_net, l1_loss, lpips_loss_fn
from data_pipeline.datasets.stage1_dataset import Stage1Dataset
from data_pipeline.conversion.unified_schema import read_mask

log = logging.getLogger(__name__)

# ImageNet normalization (must match Stage1Dataset exactly)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Spatial token diagnostic tests")
    p.add_argument("--checkpoint", default="checkpoints/stage1_full_rtx5090/epoch_024.pt")
    p.add_argument("--open-drawer", default="data/rlbench/open_drawer.hdf5",
                    dest="open_drawer")
    p.add_argument("--close-jar", default="data/rlbench/close_jar.hdf5",
                    dest="close_jar")
    p.add_argument("--depth-raw", default="data/raw/close_jar",
                    dest="depth_raw")
    p.add_argument("--output-dir", default="outputs/spatial_diagnostic",
                    dest="output_dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=8, dest="num_samples")
    p.add_argument("--pretrained", action="store_true",
                    help="Use real DINOv3-L (requires HF token)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    device = torch.device(args.device)

    log.info("Loading encoder (pretrained=%s)...", args.pretrained)
    encoder = FrozenMultiViewEncoder(pretrained=args.pretrained).to(device).eval()

    adapter = TrainableAdapter().to(device)
    decoder = ViTDecoder().to(device)

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    def strip(sd):
        prefix = "_orig_mod."
        if any(k.startswith(prefix) for k in sd):
            return {k.removeprefix(prefix): v for k, v in sd.items()}
        return sd

    adapter.load_state_dict(strip(ckpt["adapter"]))
    decoder.load_state_dict(strip(ckpt["decoder"]))
    adapter.eval()
    decoder.eval()
    log.info("Loaded checkpoint epoch %d from %s", ckpt["epoch"], args.checkpoint)

    lpips_net = create_lpips_net().to(device)
    return encoder, adapter, decoder, lpips_net


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def imagenet_normalize(imgs_01_hwc: np.ndarray) -> np.ndarray:
    """(K, H, W, 3) float32 [0,1] → (K, 3, H, W) ImageNet-normalized.

    Matches Stage1Dataset exactly: (x - mean) / std, NOT Chi's (x*2-1 - mean)/std.
    """
    normed = (imgs_01_hwc - _MEAN) / _STD
    return np.moveaxis(normed, -1, -3)  # (K, 3, H, W)


def denormalize_imagenet(img_chw: np.ndarray) -> np.ndarray:
    """(3, H, W) ImageNet-normalized → (H, W, 3) [0,1] for display."""
    img_hwc = np.moveaxis(img_chw, -3, -1)
    return np.clip(img_hwc * _STD + _MEAN, 0, 1)


@torch.no_grad()
def encode_single(encoder, adapter, img_enc, device):
    """(3, H, W) ImageNet-normed → (196, 1024) raw tokens, (196, 512) adapted."""
    x = torch.from_numpy(img_enc).unsqueeze(0).to(device)
    raw = encoder(x)                  # (1, 196, 1024) — cancel_affine_ln applied internally
    adapted = adapter(raw)            # (1, 196, 512)
    return raw[0].cpu(), adapted[0].cpu()


def load_depth_png(path: str) -> np.ndarray:
    """Load RLBench depth PNG → float32 (128, 128).

    RLBench stores depth as 24-bit across RGB: depth = R + G/256 + B/65536
    (near-depth format, NOT R*65536+G*256+B).
    Actually, CoppeliaSim stores depth as a single float packed into RGB.
    Let's just use the red channel as a proxy — it captures the coarse depth.
    For the probe we just need relative depth ordering, not absolute values.
    """
    img = np.array(Image.open(path))  # (128, 128, 3) uint8
    # Use all 3 channels for better precision
    depth = img[:, :, 0].astype(np.float32) + \
            img[:, :, 1].astype(np.float32) / 256.0 + \
            img[:, :, 2].astype(np.float32) / 65536.0
    return depth


# ---------------------------------------------------------------------------
# Test 1: RAE Decoder Reconstruction Ablation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test1(encoder, adapter, decoder, lpips_net, hdf5_path, device,
              output_dir, num_samples):
    """Compare reconstruction across all spatial pool sizes 1-14 plus full."""
    log.info("=== Test 1: RAE Decoder Reconstruction Ablation (sweep 1-14 + full) ===")

    ds = Stage1Dataset(hdf5_path, split="valid")
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)

    # All conditions: "full" (no pooling), "pooled" (mean→tile), and spatial S for S=2..14
    # S=1 is the same as "pooled", S=14 should match "full" (sanity check)
    pool_sizes = list(range(2, 15))  # 2,3,...,14
    all_names = ["full", "pooled"] + [f"spatial{s}" for s in pool_sizes]

    metrics = {name: {"l1": [], "lpips": []} for name in all_names}

    # For the image grid, pick a readable subset
    fig_keys = ["original", "full", "pooled", "spatial4", "spatial7",
                "spatial8", "spatial10", "spatial14"]
    fig_data = []

    for idx in indices:
        sample = ds[int(idx)]
        imgs_enc = sample["images_enc"]      # (K, 3, 224, 224)
        imgs_tgt = sample["images_target"]   # (K, 3, 224, 224)
        vp = sample["view_present"]          # (K,)

        for k in range(vp.shape[0]):
            if not vp[k]:
                continue

            img_enc = imgs_enc[k].unsqueeze(0).to(device)   # (1, 3, 224, 224)
            img_tgt = imgs_tgt[k].unsqueeze(0).to(device)   # (1, 3, 224, 224)

            # Encode
            raw_tokens = encoder(img_enc)       # (1, 196, 1024)
            z_full = adapter(raw_tokens)        # (1, 196, 512)
            z_2d = z_full.reshape(1, 14, 14, 512).permute(0, 3, 1, 2)  # (B, 512, 14, 14)

            # (a) Full tokens — no pooling
            recon_full = decoder(z_full)

            # (b) Mean-pooled → expand (equivalent to spatial_pool_size=1)
            z_pooled = z_full.mean(dim=1, keepdim=True).expand(-1, 196, -1)
            recon_pooled = decoder(z_pooled)

            # Metrics for full and pooled
            metrics["full"]["l1"].append(l1_loss(recon_full, img_tgt).item())
            metrics["full"]["lpips"].append(lpips_loss_fn(recon_full, img_tgt, lpips_net).item())
            metrics["pooled"]["l1"].append(l1_loss(recon_pooled, img_tgt).item())
            metrics["pooled"]["lpips"].append(lpips_loss_fn(recon_pooled, img_tgt, lpips_net).item())

            # (c) Sweep spatial pool sizes 2..14
            recons = {"full": recon_full, "pooled": recon_pooled}
            for s in pool_sizes:
                z_s = F.adaptive_avg_pool2d(z_2d, s)             # (1, 512, s, s)
                z_up = F.interpolate(z_s, size=14, mode="nearest")
                z_tiled = z_up.permute(0, 2, 3, 1).reshape(1, 196, 512)
                recon_s = decoder(z_tiled)
                name = f"spatial{s}"
                metrics[name]["l1"].append(l1_loss(recon_s, img_tgt).item())
                metrics[name]["lpips"].append(lpips_loss_fn(recon_s, img_tgt, lpips_net).item())
                recons[name] = recon_s

            # Save first few for figure (subset of columns)
            if len(fig_data) < 4:
                row_data = {"original": imgs_tgt[k].permute(1, 2, 0).numpy()}
                for key in fig_keys:
                    if key == "original":
                        continue
                    row_data[key] = recons[key][0].cpu().permute(1, 2, 0).numpy()
                fig_data.append(row_data)

    # Average metrics
    results = {}
    for name in all_names:
        results[name] = {
            "l1": np.mean(metrics[name]["l1"]),
            "lpips": np.mean(metrics[name]["lpips"]),
        }

    # Print full table
    print("\n  Test 1: Reconstruction Ablation (full sweep)")
    print(f"  {'Condition':15s} {'Tokens':>8s} {'L1':>8s} {'LPIPS':>8s} {'Gap recovered':>14s}")
    print("  " + "-" * 50)

    lpips_full = results["full"]["lpips"]
    lpips_pooled = results["pooled"]["lpips"]
    gap_total = lpips_pooled - lpips_full

    for name in all_names:
        if name == "full":
            n_tokens = 196
        elif name == "pooled":
            n_tokens = 1
        else:
            s = int(name.replace("spatial", ""))
            n_tokens = s * s
        gap_recovered = (lpips_pooled - results[name]["lpips"]) / gap_total * 100 if gap_total > 1e-8 else 0
        print(f"  {name:15s} {n_tokens:>8d} {results[name]['l1']:8.4f} {results[name]['lpips']:8.4f} {gap_recovered:13.1f}%")

    ratio = lpips_full / max(lpips_pooled, 1e-8)
    go = ratio < 0.5
    print(f"\n  LPIPS ratio (full/pooled): {ratio:.3f}")
    print(f"  Go/No-Go (< 0.5): {'GO ✓' if go else 'NO-GO ✗'}")

    # Save LPIPS vs pool size chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sizes = [1] + pool_sizes
    labels = ["pooled"] + [f"{s}×{s}" for s in pool_sizes]
    lpips_vals = [results["pooled"]["lpips"]] + [results[f"spatial{s}"]["lpips"] for s in pool_sizes]
    l1_vals = [results["pooled"]["l1"]] + [results[f"spatial{s}"]["l1"] for s in pool_sizes]
    n_tokens_list = [1] + [s*s for s in pool_sizes]

    # LPIPS vs pool size
    ax1.plot(sizes, lpips_vals, "o-", color="#0F6E56", linewidth=2, markersize=6, label="Spatial pool")
    ax1.axhline(y=lpips_full, color="#378ADD", linestyle="--", linewidth=1.5, label=f"Full 196 tokens ({lpips_full:.4f})")
    ax1.axhline(y=lpips_pooled, color="#E24B4A", linestyle="--", linewidth=1.5, label=f"Avg pooled ({lpips_pooled:.4f})")
    # Mark key sizes
    for s_mark, color, lbl in [(4, "#BA7517", "4×4"), (7, "#534AB7", "7×7"),
                                (8, "#993C1D", "8×8 (native)")]:
        idx = sizes.index(s_mark)
        ax1.plot(s_mark, lpips_vals[idx], "s", color=color, markersize=10, zorder=5)
        ax1.annotate(f"{lbl}\n{lpips_vals[idx]:.4f}", (s_mark, lpips_vals[idx]),
                     textcoords="offset points", xytext=(10, 10), fontsize=9, color=color)
    ax1.set_xlabel("Spatial pool size (S)", fontsize=12)
    ax1.set_ylabel("LPIPS (lower = better)", fontsize=12)
    ax1.set_title("Reconstruction quality vs spatial resolution", fontsize=13)
    ax1.set_xticks(sizes)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Gap recovered vs token count
    gap_pct = [(lpips_pooled - lp) / gap_total * 100 if gap_total > 1e-8 else 0 for lp in lpips_vals]
    ax2.plot(n_tokens_list, gap_pct, "o-", color="#0F6E56", linewidth=2, markersize=6)
    ax2.axhline(y=100, color="#378ADD", linestyle="--", linewidth=1.5, label="Full (100%)")
    for s_mark, color, lbl in [(4, "#BA7517", "4×4 (16)"), (7, "#534AB7", "7×7 (49)"),
                                (8, "#993C1D", "8×8 (64)")]:
        idx = sizes.index(s_mark)
        ax2.plot(n_tokens_list[idx], gap_pct[idx], "s", color=color, markersize=10, zorder=5)
        ax2.annotate(f"{lbl}\n{gap_pct[idx]:.1f}%", (n_tokens_list[idx], gap_pct[idx]),
                     textcoords="offset points", xytext=(10, -15), fontsize=9, color=color)
    ax2.set_xlabel("Token count per camera", fontsize=12)
    ax2.set_ylabel("Gap recovered vs avg pooling (%)", fontsize=12)
    ax2.set_title("Information recovery vs token budget", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "test1_lpips_vs_poolsize.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", path)

    # Save image grid figure (subset of conditions)
    if fig_data:
        n = len(fig_data)
        n_cols = len(fig_keys)
        fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        fig_titles = ["Original", "Full (196)", "Pooled (1→196)", "4×4 (16→196)",
                      "7×7 (49→196)", "8×8 (64→196)", "10×10 (100→196)", "14×14 (196→196)"]
        for row, d in enumerate(fig_data):
            for col, (key, title) in enumerate(zip(fig_keys, fig_titles)):
                axes[row, col].imshow(np.clip(d[key], 0, 1))
                if row == 0:
                    axes[row, col].set_title(title, fontsize=10)
                axes[row, col].axis("off")
        plt.suptitle("Test 1: Reconstruction Ablation", fontsize=14, y=1.01)
        plt.tight_layout()
        path = os.path.join(output_dir, "test1_reconstruction_grid.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved %s", path)

    return results, go


# ---------------------------------------------------------------------------
# Test 2: Linear Depth Probe
# ---------------------------------------------------------------------------

def run_test2(encoder, adapter, hdf5_path, raw_dir, device,
              output_dir, max_samples=2000):
    """Train a linear probe to predict per-patch depth from adapter tokens."""
    log.info("=== Test 2: Linear Depth Probe ===")

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        log.warning("Raw depth dir not found: %s — skipping Test 2", raw_dir)
        return None, False

    # Collect (adapter_tokens, depth_target) pairs from close_jar train split
    all_tokens = []
    all_depths = []

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = read_mask(f, "train")
        log.info("Collecting depth probe data from %d train demos...", len(demo_keys))

        for demo_key in demo_keys:
            ep_idx = int(demo_key.split("_")[1])
            grp = f[f"data/{demo_key}"]
            T = grp["images"].shape[0]

            for t in range(0, T, 4):  # subsample every 4th frame
                # Depth PNG
                depth_path = raw_dir / f"all_variations/episodes/episode{ep_idx}/front_depth/{t}.png"
                if not depth_path.exists():
                    continue

                depth = load_depth_png(str(depth_path))  # (128, 128)

                # Pool depth to 14×14 patches
                depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # (1,1,128,128)
                depth_patches = F.adaptive_avg_pool2d(depth_t, 14)  # (1,1,14,14)
                depth_flat = depth_patches.reshape(196)
                # Normalize per-image to [0,1]
                d_min, d_max = depth_flat.min(), depth_flat.max()
                if d_max - d_min > 1e-6:
                    depth_flat = (depth_flat - d_min) / (d_max - d_min)
                else:
                    depth_flat = torch.zeros(196)

                # RGB → adapter tokens (front camera = slot 0)
                img_hwc = grp["images"][t, 0]  # (224, 224, 3) uint8
                img_01 = img_hwc.astype(np.float32) / 255.0
                img_enc = imagenet_normalize(img_01[np.newaxis])[0]  # (3, 224, 224)

                _, adapted = encode_single(encoder, adapter, img_enc, device)
                # adapted: (196, 512)

                all_tokens.append(adapted)
                all_depths.append(depth_flat)

                if len(all_tokens) >= max_samples:
                    break
            if len(all_tokens) >= max_samples:
                break

    if len(all_tokens) < 50:
        log.warning("Only %d samples collected — too few for depth probe", len(all_tokens))
        return None, False

    log.info("Collected %d samples for depth probe", len(all_tokens))

    # Stack into tensors
    X = torch.stack(all_tokens)   # (N, 196, 512)
    Y = torch.stack(all_depths)   # (N, 196)

    # Flatten: treat each patch independently
    X_flat = X.reshape(-1, 512)   # (N*196, 512)
    Y_flat = Y.reshape(-1, 1)    # (N*196, 1)

    # 80/20 split
    n = X_flat.shape[0]
    perm = torch.randperm(n)
    n_train = int(0.8 * n)
    X_train, X_val = X_flat[perm[:n_train]], X_flat[perm[n_train:]]
    Y_train, Y_val = Y_flat[perm[:n_train]], Y_flat[perm[n_train:]]

    # Train linear probe
    probe = nn.Linear(512, 1).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    train_losses, val_losses = [], []

    for epoch in range(100):
        # Train (mini-batch)
        probe.train()
        epoch_loss = 0.0
        bs = 4096
        for i in range(0, X_train.shape[0], bs):
            xb = X_train[i:i+bs].to(device)
            yb = Y_train[i:i+bs].to(device)
            pred = probe(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        train_losses.append(epoch_loss / X_train.shape[0])

        # Val
        probe.eval()
        with torch.no_grad():
            val_pred = []
            for i in range(0, X_val.shape[0], bs):
                val_pred.append(probe(X_val[i:i+bs].to(device)).cpu())
            val_pred = torch.cat(val_pred)
            val_loss = F.mse_loss(val_pred, Y_val).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0:
            log.info("  Depth probe epoch %3d: train=%.5f val=%.5f", epoch+1,
                     train_losses[-1], val_losses[-1])

    # R-squared on validation set
    probe.eval()
    with torch.no_grad():
        val_pred_all = []
        for i in range(0, X_val.shape[0], 4096):
            val_pred_all.append(probe(X_val[i:i+4096].to(device)).cpu())
        val_pred_all = torch.cat(val_pred_all)
    ss_res = ((val_pred_all - Y_val) ** 2).sum().item()
    ss_tot = ((Y_val - Y_val.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    go = r2 > 0.5
    print(f"\n  Test 2: Linear Depth Probe")
    print(f"  R² = {r2:.4f}")
    print(f"  Go/No-Go (R² > 0.5): {'GO ✓' if go else 'NO-GO ✗'}")

    # Save loss plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="Train MSE")
    ax.plot(val_losses, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Test 2: Depth Probe Training (R² = {r2:.4f})")
    ax.legend()
    path = os.path.join(output_dir, "test2_depth_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", path)

    # Save prediction visualization (4 samples from validation)
    probe.eval()
    n_viz = min(4, len(all_tokens) - int(0.8 * len(all_tokens)))
    if n_viz > 0:
        fig, axes = plt.subplots(n_viz, 4, figsize=(16, 4 * n_viz))
        if n_viz == 1:
            axes = axes[np.newaxis, :]

        # Use last few samples as viz (from validation portion)
        viz_start = int(0.8 * len(all_tokens))
        for row in range(n_viz):
            si = viz_start + row
            if si >= len(all_tokens):
                break
            tok = all_tokens[si].unsqueeze(0).to(device)  # (1, 196, 512)
            with torch.no_grad():
                pred = probe(tok.reshape(-1, 512)).cpu().reshape(14, 14)
            gt = all_depths[si].reshape(14, 14)

            # Load original RGB for context
            # (approximate — just show the depth maps)
            axes[row, 0].set_title("GT Depth (14×14)" if row == 0 else "")
            axes[row, 0].imshow(gt.numpy(), cmap="viridis")
            axes[row, 0].axis("off")

            axes[row, 1].set_title("Predicted Depth" if row == 0 else "")
            axes[row, 1].imshow(pred.numpy(), cmap="viridis")
            axes[row, 1].axis("off")

            axes[row, 2].set_title("Error |GT - Pred|" if row == 0 else "")
            axes[row, 2].imshow(np.abs(gt.numpy() - pred.numpy()), cmap="hot")
            axes[row, 2].axis("off")

            axes[row, 3].set_title("GT Depth (full res)" if row == 0 else "")
            # Show the original 128x128 depth
            ep_idx = int(demo_keys[si // (all_tokens[0].shape[0])].split("_")[1]) if si < len(demo_keys) else 0
            axes[row, 3].imshow(gt.numpy(), cmap="viridis")
            axes[row, 3].axis("off")

        plt.suptitle(f"Test 2: Depth Probe Predictions (R² = {r2:.4f})", fontsize=14)
        plt.tight_layout()
        path = os.path.join(output_dir, "test2_depth_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved %s", path)

    return {"r2": r2, "final_train_mse": train_losses[-1],
            "final_val_mse": val_losses[-1]}, go


# ---------------------------------------------------------------------------
# Test 3: Cosine Similarity + PCA Visualization
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_test3(encoder, adapter, hdf5_path, device, output_dir, num_samples):
    """Cosine similarity matrices and PCA maps for adapter vs raw tokens."""
    log.info("=== Test 3: Cosine Similarity + PCA ===")

    ds = Stage1Dataset(hdf5_path, split="valid")
    indices = np.linspace(0, len(ds) - 1, min(num_samples, len(ds)), dtype=int)

    locality_raw_list = []
    locality_adapted_list = []
    fig_data = []

    # Build neighbor mask for locality score
    # 14×14 grid: for each patch, find patches within distance 2 vs distance > 5
    coords = np.array([(i, j) for i in range(14) for j in range(14)])  # (196, 2)
    dists = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))  # (196, 196)
    near_mask = (dists <= 2.0) & (dists > 0)    # ~3×3 neighborhood excluding self
    far_mask = dists > 5.0

    for idx in indices:
        sample = ds[int(idx)]
        imgs_enc = sample["images_enc"]
        imgs_tgt = sample["images_target"]
        vp = sample["view_present"]

        # Use first active camera
        cam = 0
        for k in range(vp.shape[0]):
            if vp[k]:
                cam = k
                break

        raw, adapted = encode_single(encoder, adapter, imgs_enc[cam].numpy(), device)
        # raw: (196, 1024), adapted: (196, 512)

        # Cosine similarity matrices
        raw_norm = F.normalize(raw.float(), dim=-1)
        adapted_norm = F.normalize(adapted.float(), dim=-1)
        sim_raw = (raw_norm @ raw_norm.T).numpy()
        sim_adapted = (adapted_norm @ adapted_norm.T).numpy()

        # Locality scores
        loc_raw = sim_raw[near_mask].mean() / max(sim_raw[far_mask].mean(), 1e-8)
        loc_adapted = sim_adapted[near_mask].mean() / max(sim_adapted[far_mask].mean(), 1e-8)
        locality_raw_list.append(loc_raw)
        locality_adapted_list.append(loc_adapted)

        # PCA
        pca_raw = PCA(n_components=3).fit_transform(raw.numpy())
        pca_raw = pca_raw.reshape(14, 14, 3)
        pca_raw = (pca_raw - pca_raw.min()) / (pca_raw.max() - pca_raw.min() + 1e-8)

        pca_adapted = PCA(n_components=3).fit_transform(adapted.numpy())
        pca_adapted = pca_adapted.reshape(14, 14, 3)
        pca_adapted = (pca_adapted - pca_adapted.min()) / (pca_adapted.max() - pca_adapted.min() + 1e-8)

        fig_data.append({
            "original": imgs_tgt[cam].permute(1, 2, 0).numpy(),
            "sim_raw": sim_raw,
            "sim_adapted": sim_adapted,
            "pca_raw": pca_raw,
            "pca_adapted": pca_adapted,
            "loc_raw": loc_raw,
            "loc_adapted": loc_adapted,
        })

    avg_loc_raw = np.mean(locality_raw_list)
    avg_loc_adapted = np.mean(locality_adapted_list)

    print(f"\n  Test 3: Cosine Similarity + PCA")
    print(f"  Locality ratio (near/far cosine sim):")
    print(f"    Raw DINOv3:    {avg_loc_raw:.3f}")
    print(f"    Adapted (512): {avg_loc_adapted:.3f}")
    preserved = avg_loc_adapted > 1.2  # near patches should be noticeably more similar
    print(f"  Spatial structure: {'PRESERVED ✓' if preserved else 'COLLAPSED ✗'}")

    # Save cosine similarity figure
    n_show = min(4, len(fig_data))
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    for row in range(n_show):
        d = fig_data[row]
        axes[row, 0].imshow(np.clip(d["original"], 0, 1))
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(d["sim_raw"], cmap="viridis", vmin=-0.2, vmax=1.0)
        axes[row, 1].set_title(f"Raw DINOv3 Cosine Sim (loc={d['loc_raw']:.2f})" if row == 0 else f"loc={d['loc_raw']:.2f}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(d["sim_adapted"], cmap="viridis", vmin=-0.2, vmax=1.0)
        axes[row, 2].set_title(f"Adapted Cosine Sim (loc={d['loc_adapted']:.2f})" if row == 0 else f"loc={d['loc_adapted']:.2f}")
        axes[row, 2].axis("off")
    plt.suptitle("Test 3: Cosine Similarity Matrices", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "test3_cosine_sim.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", path)

    # Save PCA figure
    fig, axes = plt.subplots(n_show, 3, figsize=(15, 5 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]
    for row in range(n_show):
        d = fig_data[row]
        axes[row, 0].imshow(np.clip(d["original"], 0, 1))
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(d["pca_raw"])
        axes[row, 1].set_title("Raw DINOv3 PCA (3 comp → RGB)" if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(d["pca_adapted"])
        axes[row, 2].set_title("Adapted PCA (3 comp → RGB)" if row == 0 else "")
        axes[row, 2].axis("off")
    plt.suptitle("Test 3: PCA Visualization (14×14 spatial maps)", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "test3_pca_maps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", path)

    return {"locality_raw": avg_loc_raw, "locality_adapted": avg_loc_adapted}, preserved


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(r1, go1, r2, go2, r3, go3, output_dir):
    lines = []
    lines.append("=" * 50)
    lines.append("  SPATIAL DIAGNOSTIC SUMMARY")
    lines.append("=" * 50)

    lines.append("\n  Test 1: Reconstruction Ablation")
    if r1:
        pool_sizes = list(range(2, 15))
        all_names = ["full", "pooled"] + [f"spatial{s}" for s in pool_sizes]
        lpips_full = r1["full"]["lpips"]
        lpips_pooled = r1["pooled"]["lpips"]
        gap_total = lpips_pooled - lpips_full
        for name in all_names:
            if name in r1:
                gap_pct = (lpips_pooled - r1[name]["lpips"]) / gap_total * 100 if gap_total > 1e-8 else 0
                lines.append(f"    {name:12s}  L1={r1[name]['l1']:.4f}  LPIPS={r1[name]['lpips']:.4f}  gap_recovered={gap_pct:.1f}%")
        ratio = lpips_full / max(lpips_pooled, 1e-8)
        lines.append(f"    LPIPS ratio (full/pooled): {ratio:.3f}")
        lines.append(f"    VERDICT: {'GO' if go1 else 'NO-GO'}")

    lines.append("\n  Test 2: Linear Depth Probe")
    if r2:
        lines.append(f"    R² = {r2['r2']:.4f}")
        lines.append(f"    VERDICT: {'GO' if go2 else 'NO-GO'}")
    else:
        lines.append("    SKIPPED (no depth data)")

    lines.append("\n  Test 3: Cosine Similarity + PCA")
    if r3:
        lines.append(f"    Raw locality:     {r3['locality_raw']:.3f}")
        lines.append(f"    Adapted locality: {r3['locality_adapted']:.3f}")
        lines.append(f"    VERDICT: {'PRESERVED' if go3 else 'COLLAPSED'}")

    overall = go1 and go3 and (go2 if r2 else True)
    lines.append(f"\n  OVERALL: {'GO for Plan B ✓' if overall else 'NO-GO for Plan B ✗'}")
    lines.append("=" * 50)

    text = "\n".join(lines)
    print(text)

    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(text + "\n")
    log.info("Saved summary to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "diagnostic.log")),
        ],
    )

    encoder, adapter, decoder, lpips_net = load_models(args)
    device = torch.device(args.device)

    # Test 1: open_drawer
    r1, go1 = run_test1(encoder, adapter, decoder, lpips_net,
                        args.open_drawer, device, args.output_dir, args.num_samples)

    # Test 2: close_jar + raw depth
    r2, go2 = run_test2(encoder, adapter, args.close_jar, args.depth_raw,
                        device, args.output_dir)

    # Test 3: open_drawer
    r3, go3 = run_test3(encoder, adapter, args.open_drawer, device,
                        args.output_dir, args.num_samples)

    # Summary
    print_summary(r1, go1, r2, go2, r3, go3, args.output_dir)


if __name__ == "__main__":
    main()
