"""Stage 1 RAE dataset: image reconstruction from unified HDF5 (Phase A.7).

Loads single-timestep multi-view images for encoder-decoder training.
Returns both ImageNet-normalized images (encoder input) and raw [0,1]
images (reconstruction target for L1 + LPIPS losses).

No actions or proprio — Stage 1 is pure image reconstruction.

Supports single or multiple HDF5 files for combined multi-task training.
When multiple files are provided, samples from all files are merged into
one flat index and shuffled by the DataLoader.

Output per sample (dict):
  images_enc:    (K, 3, H, W)  float32, ImageNet-normalized
  images_target: (K, 3, H, W)  float32, [0, 1] range
  view_present:  (K,)          bool

Owner: Swagman
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.conversion.unified_schema import NUM_CAMERA_SLOTS, IMAGE_SIZE
from data_pipeline.conversion.unified_schema import read_mask


# ImageNet normalization constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Stage1Dataset(Dataset):
    """Dataset for Stage 1 RAE training (image reconstruction).

    Each sample is one timestep with K camera views. Returns both
    ImageNet-normalized images (for the frozen encoder) and raw [0,1]
    images (for reconstruction loss computation).

    Accepts a single HDF5 path (backward-compatible) or a list of paths
    for combined multi-task training.

    Args:
        hdf5_paths: Path or list of paths to unified HDF5 files.
        split:      "train" or "valid".
    """

    def __init__(self, hdf5_paths: "str | list[str]", split: str = "train"):
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self._hdf5_paths = list(hdf5_paths)

        # Build flat index: one entry per (file_idx, demo_key, timestep)
        self._index = []
        self._view_present_per_file = []

        for file_idx, path in enumerate(self._hdf5_paths):
            with h5py.File(path, "r") as f:
                demo_keys = read_mask(f, split)

                for key in demo_keys:
                    T = f[f"data/{key}/images"].shape[0]
                    for t in range(T):
                        self._index.append((file_idx, key, t))

                # Cache view_present per file (differs by benchmark)
                self._view_present_per_file.append(
                    f[f"data/{demo_keys[0]}/view_present"][:]
                )

        # Backward compat: expose single path for existing code
        self.hdf5_path = self._hdf5_paths[0] if len(self._hdf5_paths) == 1 else None

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        file_idx, demo_key, t = self._index[idx]

        with h5py.File(self._hdf5_paths[file_idx], "r") as f:
            imgs_hwc = f[f"data/{demo_key}/images"][t]  # (K, H, W, 3)

        # Convert to float32 [0, 1]
        if imgs_hwc.dtype == np.uint8:
            imgs_01 = imgs_hwc.astype(np.float32) / 255.0
        else:
            imgs_01 = imgs_hwc.astype(np.float32)

        # Raw target: HWC -> CHW
        images_target = np.moveaxis(imgs_01, -1, -3)  # (K, 3, H, W)

        # ImageNet-normalized: for encoder input
        images_enc = (imgs_01 - _IMAGENET_MEAN) / _IMAGENET_STD
        images_enc = np.moveaxis(images_enc, -1, -3)   # (K, 3, H, W)

        return {
            "images_enc":    torch.from_numpy(images_enc),
            "images_target": torch.from_numpy(images_target),
            "view_present":  torch.from_numpy(
                self._view_present_per_file[file_idx]
            ),
        }
