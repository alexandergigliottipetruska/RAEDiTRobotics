"""Precompute frozen encoder tokens for fast Stage 3 training.

Runs frozen DINOv3-L + Cancel-Affine LN on all images once and saves
the (196, 1024) tokens to a cache HDF5. Training then skips the encoder
entirely and just runs the adapter on cached tokens.

Usage:
  python training/precompute_tokens.py \
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \
    --output data/unified/robomimic/lift/ph_tokens.hdf5

  # Specify batch size and device
  python training/precompute_tokens.py \
    --hdf5 data/unified/robomimic/lift/ph.hdf5 \
    --batch_size 64 --device cuda
"""

import argparse
import logging
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.encoder import FrozenMultiViewEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def precompute(
    hdf5_path: str,
    output_path: str,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Precompute encoder tokens and save to cache HDF5.

    The output HDF5 has the same structure as the input but with
    `tokens` (T, K, 196, 1024) float16 instead of `images`.
    Actions, proprio, view_present, masks, and norm_stats are copied over.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # Load encoder
    encoder = FrozenMultiViewEncoder(pretrained=True)
    encoder = encoder.to(device)
    encoder.eval()

    cancel_affine_ln = nn.LayerNorm(1024, elementwise_affine=False).to(device)

    with h5py.File(hdf5_path, "r") as src, h5py.File(output_path, "w") as dst:
        # Copy file-level attributes
        for attr in src.attrs:
            dst.attrs[attr] = src.attrs[attr]
        dst.attrs["has_cached_tokens"] = True

        # Copy masks
        if "mask" in src:
            src.copy("mask", dst)

        # Copy norm_stats
        if "norm_stats" in src:
            src.copy("norm_stats", dst)

        # Get all demo keys
        all_keys = sorted(src["data"].keys())
        log.info("Processing %d demos from %s", len(all_keys), hdf5_path)

        for key in tqdm(all_keys, desc="Precomputing tokens"):
            grp = src[f"data/{key}"]
            T = grp["images"].shape[0]
            K = grp["images"].shape[1]
            view_present = grp["view_present"][:]

            # Output tokens: (T, K, 196, 1024) in float16 to save space
            tokens_all = np.zeros((T, K, 196, 1024), dtype=np.float16)

            for k in range(K):
                if not view_present[k]:
                    continue

                # Load all T images for this view
                imgs_raw = grp["images"][:, k]  # (T, H, W, 3) uint8

                # Convert to float32 [0,1] and ImageNet normalize
                imgs_float = imgs_raw.astype(np.float32) / 255.0
                imgs_norm = (imgs_float - _IMAGENET_MEAN) / _IMAGENET_STD
                imgs_chw = np.ascontiguousarray(
                    np.moveaxis(imgs_norm, -1, -3)
                )  # (T, 3, H, W)
                imgs_tensor = torch.from_numpy(imgs_chw)

                # Process in batches
                view_tokens = []
                for i in range(0, T, batch_size):
                    batch = imgs_tensor[i : i + batch_size].to(device)
                    with torch.no_grad():
                        raw = encoder(batch)  # (B, 196, 1024)
                        normed = cancel_affine_ln(raw)
                    view_tokens.append(normed.cpu().to(torch.float16).numpy())

                tokens_all[:, k] = np.concatenate(view_tokens, axis=0)

            # Save to output (gzip compresses zero-padded absent views well)
            dst_grp = dst.create_group(f"data/{key}")
            dst_grp.create_dataset(
                "tokens", data=tokens_all,
                chunks=(min(T, 16), K, 196, 1024),
                compression="gzip", compression_opts=1,
            )
            # Copy actions, proprio, view_present
            for ds_name in ["actions", "proprio", "view_present"]:
                dst_grp.create_dataset(ds_name, data=grp[ds_name][:])

    # Report size
    src_size = os.path.getsize(hdf5_path) / (1024**3)
    dst_size = os.path.getsize(output_path) / (1024**3)
    log.info("Done! Original: %.2f GB, Cache: %.2f GB", src_size, dst_size)


def main():
    parser = argparse.ArgumentParser(description="Precompute encoder tokens")
    parser.add_argument("--hdf5", required=True, help="Input unified HDF5")
    parser.add_argument("--output", default=None,
                        help="Output cache HDF5 (default: <input>_tokens.hdf5)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.hdf5)
        args.output = f"{base}_tokens{ext}"

    precompute(args.hdf5, args.output, args.batch_size, args.device)


if __name__ == "__main__":
    main()
