"""Stage 3 policy dataset: temporal windows for diffusion policy training.

Samples (T_obs observation frames + T_pred action targets) from unified HDF5.
Unlike Stage 1 (single timestep), Stage 3 needs temporal context for the
observation encoder and action chunks for DDPM noise prediction.

Supports two modes:
  1. Standard: loads images, returns images_enc for encoder at train time.
  2. Cached: loads precomputed encoder tokens (from precompute_tokens.py),
     skips the expensive encoder forward pass entirely.

Output per sample (dict):
  images_enc:      (T_obs, K, 3, H, W)     float32, ImageNet-normalized  [standard mode]
  cached_tokens:   (T_obs, K, 196, 1024)   float32, post-encoder/LN      [cached mode]
  images_target:   (T_obs, K, 3, H, W)     float32, [0, 1] range         [standard mode only]
  actions:         (T_pred, D_act)          float32, normalized
  proprio:         (T_obs, D_prop)          float32, normalized
  view_present:    (K,)                     bool

Supports single or multiple HDF5 files for per-task or multi-task training.

Owner: Swagman
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.conversion.unified_schema import read_mask
from data_pipeline.conversion.compute_norm_stats import load_norm_stats

# ImageNet normalization constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Stage3Dataset(Dataset):
    """Dataset for Stage 3 diffusion policy training.

    Each sample is a temporal window: T_obs observation frames and T_pred
    future actions. Start padding repeats frame 0 when t < T_obs - 1.
    End trimming excludes timesteps where T_pred actions aren't available.

    Args:
        hdf5_paths: Path or list of paths to unified HDF5 files.
            Can be original (with images) or cached (with tokens).
            Cached files are auto-detected via 'has_cached_tokens' attr.
        split:      "train" or "valid".
        T_obs:      Observation horizon (number of past frames).
        T_pred:     Prediction horizon (number of future actions).
        norm_mode:  "zscore" or "minmax" for action/proprio normalization.
    """

    def __init__(
        self,
        hdf5_paths: "str | list[str]",
        split: str = "train",
        T_obs: int = 2,
        T_pred: int = 16,
        norm_mode: str = "minmax",
    ):
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self._hdf5_paths = list(hdf5_paths)
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.norm_mode = norm_mode

        if norm_mode not in ("zscore", "minmax"):
            raise ValueError(f"norm_mode must be 'zscore' or 'minmax', got '{norm_mode}'")

        # Build flat index and load norm stats per file
        self._index = []  # list of (file_idx, demo_key, t)
        self._view_present_per_file = []
        self._norm_per_file = []  # per-file norm stats (action/proprio dims may differ)
        self._cached_per_file = []  # bool per file: True if tokens are precomputed

        for file_idx, path in enumerate(self._hdf5_paths):
            # Load norm stats
            norm = load_norm_stats(path)
            file_norm = {}
            for field in ("actions", "proprio"):
                file_norm[field] = {
                    "mean": norm[field]["mean"],
                    "std":  norm[field]["std"],
                    "min":  norm[field].get("min"),
                    "max":  norm[field].get("max"),
                }
            if norm_mode == "minmax" and file_norm["actions"]["min"] is None:
                raise ValueError(
                    f"norm_mode='minmax' requires min/max in {path}. "
                    "Re-run conversion to generate them."
                )
            self._norm_per_file.append(file_norm)

            with h5py.File(path, "r") as f:
                # Auto-detect cached tokens
                is_cached = bool(f.attrs.get("has_cached_tokens", False))
                self._cached_per_file.append(is_cached)

                demo_keys = read_mask(f, split)
                if len(demo_keys) == 0:
                    self._view_present_per_file.append(np.zeros(4, dtype=bool))
                    continue

                for key in demo_keys:
                    T = f[f"data/{key}/actions"].shape[0]
                    # Valid range: t in [0, T - T_pred)
                    # At timestep t, actions are [t, ..., t + T_pred - 1]
                    # Need t + T_pred <= T, so t <= T - T_pred - 1
                    n_valid = max(0, T - T_pred)
                    for t in range(n_valid):
                        self._index.append((file_idx, key, t))

                # Cache view_present per file
                self._view_present_per_file.append(
                    f[f"data/{demo_keys[0]}/view_present"][:]
                )

        # Log mode
        n_cached = sum(self._cached_per_file)
        if n_cached > 0:
            import logging
            logging.getLogger(__name__).info(
                "Using precomputed tokens for %d/%d files", n_cached, len(self._hdf5_paths)
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        file_idx, demo_key, t = self._index[idx]
        T_obs, T_pred = self.T_obs, self.T_pred
        is_cached = self._cached_per_file[file_idx]

        with h5py.File(self._hdf5_paths[file_idx], "r") as f:
            grp = f[f"data/{demo_key}"]

            # --- Observations: frames [t - T_obs + 1, ..., t] ---
            obs_start = max(0, t - T_obs + 1)
            proprio_slice = grp["proprio"][obs_start : t + 1]   # (<=T_obs, D_prop)

            if is_cached:
                tokens_slice = grp["tokens"][obs_start : t + 1]  # (<=T_obs, K, 196, 1024) float16
            else:
                imgs_slice = grp["images"][obs_start : t + 1]    # (<=T_obs, K, H, W, 3)

            # Start padding: repeat first available frame
            actual_len = proprio_slice.shape[0]
            pad_before = T_obs - actual_len
            if pad_before > 0:
                proprio_raw = np.concatenate(
                    [np.repeat(proprio_slice[:1], pad_before, axis=0), proprio_slice],
                    axis=0,
                )
                if is_cached:
                    tokens_raw = np.concatenate(
                        [np.repeat(tokens_slice[:1], pad_before, axis=0), tokens_slice],
                        axis=0,
                    )
                else:
                    imgs_raw = np.concatenate(
                        [np.repeat(imgs_slice[:1], pad_before, axis=0), imgs_slice],
                        axis=0,
                    )
            else:
                proprio_raw = proprio_slice
                if is_cached:
                    tokens_raw = tokens_slice
                else:
                    imgs_raw = imgs_slice

            # --- Actions: frames [t, ..., t + T_pred - 1] ---
            actions_raw = grp["actions"][t : t + T_pred]  # (T_pred, D_act)

        # --- Normalize actions and proprio ---
        norm = self._norm_per_file[file_idx]
        actions = self._normalize(actions_raw, norm["actions"])
        proprio = self._normalize(proprio_raw, norm["proprio"])

        result = {
            "actions":      torch.from_numpy(actions.astype(np.float32)),
            "proprio":      torch.from_numpy(proprio.astype(np.float32)),
            "view_present": torch.from_numpy(self._view_present_per_file[file_idx]),
        }

        if is_cached:
            # Return precomputed tokens (cast float16 -> float32)
            result["cached_tokens"] = torch.from_numpy(tokens_raw.astype(np.float32))
        else:
            # Process images
            if imgs_raw.dtype == np.uint8:
                imgs_01 = imgs_raw.astype(np.float32) / 255.0
            else:
                imgs_01 = imgs_raw.astype(np.float32)

            # Raw target: HWC -> CHW for co-training L_recon
            images_target = np.moveaxis(imgs_01, -1, -3)  # (T_obs, K, 3, H, W)

            # ImageNet-normalized: for frozen encoder input
            images_enc = (imgs_01 - _IMAGENET_MEAN) / _IMAGENET_STD
            images_enc = np.moveaxis(images_enc, -1, -3)  # (T_obs, K, 3, H, W)

            result["images_enc"] = torch.from_numpy(images_enc)
            result["images_target"] = torch.from_numpy(images_target)

        return result

    def _normalize(self, x: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize using zscore or minmax."""
        if self.norm_mode == "minmax":
            a_range = np.clip(stats["max"] - stats["min"], 1e-6, None)
            return 2.0 * (x - stats["min"]) / a_range - 1.0
        else:
            return (x - stats["mean"]) / np.clip(stats["std"], 1e-6, None)
