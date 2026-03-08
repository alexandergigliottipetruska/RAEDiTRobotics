"""PyTorch dataset class for unified multi-view manipulation HDF5 files.

Benchmark-agnostic. Reads unified HDF5, builds flat index of
(demo_key, timestep) pairs, opens HDF5 per-call for multi-worker safety.

Output per sample (dict):
  images:       (T_obs, K, 3, H, W)  float32, ImageNet-normalized, CHW
  actions:      (T_pred, action_dim) float32, normalized (zscore or minmax)
  proprio:      (T_obs, D_prop)      float32, raw (not normalized)
  view_present: (K,)                 bool
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.conversion.unified_schema import NUM_CAMERA_SLOTS, IMAGE_SIZE
from data_pipeline.conversion.compute_norm_stats import load_norm_stats
from data_pipeline.conversion.unified_schema import read_mask

# ImageNet normalization constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _imagenet_normalize(imgs_hwc: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization and convert HWC -> CHW.

    Handles both float32 [0,1] (robomimic) and uint8 [0,255] (RLBench)
    storage formats transparently.

    Args:
        imgs_hwc: array [..., H, W, 3], dtype float32 in [0,1] or uint8 [0,255]

    Returns:
        float32 array [..., 3, H, W] normalized to ~[-2.1, 2.6]
    """
    if imgs_hwc.dtype == np.uint8:
        imgs_hwc = imgs_hwc.astype(np.float32) / 255.0
    normalized = (imgs_hwc - _IMAGENET_MEAN) / _IMAGENET_STD
    # Move channel axis: (..., H, W, 3) -> (..., 3, H, W)
    return np.moveaxis(normalized, -1, -3)


class MultiViewManipulationDataset(Dataset):
    """Benchmark-agnostic dataset for unified HDF5 files.

    Builds a flat list of (demo_key, center_timestep) pairs covering the
    full split. At each center timestep t the sample contains:
      - observations from frames [t - T_obs + 1, ..., t]  (padded at start)
      - actions from frames      [t, ..., t + T_pred - 1] (padded at end)

    HDF5 is opened fresh per __getitem__ call so torch DataLoader workers
    with num_workers > 0 work correctly.

    Args:
        hdf5_path: Path to unified HDF5 file.
        split:     "train" or "valid".
        T_obs:     Observation horizon (number of past frames). Default 2.
        T_pred:    Prediction horizon (number of future actions). Default 50.
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = "train",
        T_obs: int = 2,
        T_pred: int = 50,
        norm_mode: str = "zscore",
    ):
        self.hdf5_path = hdf5_path
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.norm_mode = norm_mode

        if norm_mode not in ("zscore", "minmax"):
            raise ValueError(f"norm_mode must be 'zscore' or 'minmax', got '{norm_mode}'")

        # Load norm stats once (small, safe to keep in memory)
        norm = load_norm_stats(hdf5_path)
        self._a_mean = norm["actions"]["mean"]   # [action_dim]
        self._a_std  = norm["actions"]["std"]    # [action_dim]
        self._a_min  = norm["actions"]["min"]    # [action_dim] or None
        self._a_max  = norm["actions"]["max"]    # [action_dim] or None

        if norm_mode == "minmax" and self._a_min is None:
            raise ValueError(
                "norm_mode='minmax' requires min/max in HDF5. "
                "Re-run conversion to generate them."
            )

        # Read metadata and build flat index
        with h5py.File(hdf5_path, "r") as f:
            self.action_dim  = int(f.attrs["action_dim"])
            self.proprio_dim = int(f.attrs["proprio_dim"])

            demo_keys = read_mask(f, split)

            # Build flat index: one entry per (demo, timestep)
            self._index = []  # list of (demo_key, t, demo_length)
            for key in demo_keys:
                T = f[f"data/{key}/actions"].shape[0]
                for t in range(T):
                    self._index.append((key, t, T))

            # Cache view_present from first demo (constant across all demos)
            self._view_present = f[f"data/{demo_keys[0]}/view_present"][:]  # [K]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        demo_key, t, T = self._index[idx]
        T_obs, T_pred = self.T_obs, self.T_pred

        with h5py.File(self.hdf5_path, "r") as f:
            grp = f[f"data/{demo_key}"]

            # --- Observations: frames [t - T_obs + 1, ..., t] ---
            # Use contiguous slice then pad start by repeating frame 0.
            # (h5py fancy indexing requires strictly increasing indices,
            #  so we cannot pass a list with duplicates like [0, 0].)
            obs_start = max(0, t - T_obs + 1)
            imgs_slice    = grp["images"][obs_start : t + 1]    # (<=T_obs, K, H, W, 3)
            proprio_slice = grp["proprio"][obs_start : t + 1]   # (<=T_obs, D_prop)

            pad_before = T_obs - imgs_slice.shape[0]
            if pad_before > 0:
                imgs_raw    = np.concatenate(
                    [np.repeat(imgs_slice[:1], pad_before, axis=0), imgs_slice], axis=0)
                proprio_raw = np.concatenate(
                    [np.repeat(proprio_slice[:1], pad_before, axis=0), proprio_slice], axis=0)
            else:
                imgs_raw    = imgs_slice
                proprio_raw = proprio_slice

            # --- Actions: frames [t, ..., t + T_pred - 1] ---
            # Use contiguous slice then pad end by repeating last action.
            act_end = min(T, t + T_pred)
            actions_slice = grp["actions"][t : act_end]          # (<=T_pred, action_dim)

            pad_after = T_pred - actions_slice.shape[0]
            if pad_after > 0:
                actions_raw = np.concatenate(
                    [actions_slice, np.repeat(actions_slice[-1:], pad_after, axis=0)], axis=0)
            else:
                actions_raw = actions_slice

        # ImageNet normalize + HWC -> CHW: (T_obs, K, H, W, 3) -> (T_obs, K, 3, H, W)
        images = _imagenet_normalize(imgs_raw)

        # Normalize actions
        if self.norm_mode == "minmax":
            a_range = np.clip(self._a_max - self._a_min, 1e-6, None)
            actions = 2.0 * (actions_raw - self._a_min) / a_range - 1.0
        else:
            actions = (actions_raw - self._a_mean) / self._a_std

        return {
            "images":       torch.from_numpy(images),               # (T_obs, K, 3, H, W)
            "actions":      torch.from_numpy(actions.astype(np.float32)),
            "proprio":      torch.from_numpy(proprio_raw),          # (T_obs, D_prop)
            "view_present": torch.from_numpy(self._view_present),   # (K,)
        }
