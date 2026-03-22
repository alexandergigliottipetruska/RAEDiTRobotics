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
from data_pipeline.utils.rotation import convert_actions_to_rot6d

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
        use_rot6d:  If True, convert 7D actions (pos3+aa3+grip1) to 10D
                    (pos3+rot6d6+grip1) in __getitem__. The norm_stats in
                    the HDF5 must already be 10D (computed with rot6d=True).
        pad_after:  Number of timesteps to pad at the end of each demo by
                    repeating the last action. Chi uses pad_after=7 with
                    horizon=10, allowing almost every timestep to be a
                    valid sample start. Default 0 = no padding (trim).
    """

    def __init__(
        self,
        hdf5_paths: "str | list[str]",
        split: str = "train",
        T_obs: int = 2,
        T_pred: int = 16,
        norm_mode: str = "minmax",
        use_rot6d: bool = False,
        pad_before: int = 0,
        pad_after: int = 0,
        demo_keys_override: "list[str] | None" = None,
        preload_to_ram: bool = False,
    ):
        if isinstance(hdf5_paths, str):
            hdf5_paths = [hdf5_paths]
        self._hdf5_paths = list(hdf5_paths)
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.norm_mode = norm_mode
        self.use_rot6d = use_rot6d
        self.pad_before = pad_before
        self.pad_after = pad_after

        if norm_mode not in ("zscore", "minmax", "chi"):
            raise ValueError(f"norm_mode must be 'zscore', 'minmax', or 'chi', got '{norm_mode}'")

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

                demo_keys = demo_keys_override if demo_keys_override is not None else read_mask(f, split)
                if len(demo_keys) == 0:
                    self._view_present_per_file.append(np.zeros(4, dtype=bool))
                    continue

                for key in demo_keys:
                    T = f[f"data/{key}/actions"].shape[0]
                    # Valid range: t in [-pad_before, T - T_pred + pad_after)
                    # Chi: pad_before=1 allows starting one step before episode
                    # pad_after allows sampling near end by repeating last action
                    min_start = -self.pad_before
                    max_start = T - T_pred + self.pad_after
                    for t in range(min_start, max_start + 1):
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

        # Pre-load all data into RAM for faster training
        self._ram_cache = None
        if preload_to_ram:
            import logging
            log = logging.getLogger(__name__)
            log.info("Pre-loading data into RAM...")
            self._ram_cache = {}
            total_bytes = 0
            for file_idx, path in enumerate(self._hdf5_paths):
                is_cached = self._cached_per_file[file_idx]
                file_cache = {}
                with h5py.File(path, "r") as f:
                    keys_in_index = set(k for fi, k, _ in self._index if fi == file_idx)
                    for key in keys_in_index:
                        grp = f[f"data/{key}"]
                        demo = {
                            "actions": grp["actions"][:],
                            "proprio": grp["proprio"][:],
                        }
                        if is_cached:
                            demo["tokens"] = grp["tokens"][:]
                        else:
                            demo["images"] = grp["images"][:]
                        for v in demo.values():
                            total_bytes += v.nbytes
                        file_cache[key] = demo
                self._ram_cache[file_idx] = file_cache
            log.info("Pre-loaded %.1f GB into RAM (%d demos)",
                     total_bytes / 1e9, sum(len(fc) for fc in self._ram_cache.values()))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        file_idx, demo_key, t = self._index[idx]
        T_obs, T_pred = self.T_obs, self.T_pred
        is_cached = self._cached_per_file[file_idx]

        # Load data from RAM cache or HDF5
        if self._ram_cache is not None:
            demo = self._ram_cache[file_idx][demo_key]
            obs_start = max(0, t - T_obs + 1)
            obs_end = max(0, t) + 1
            proprio_slice = demo["proprio"][obs_start:obs_end].copy()
            if is_cached:
                tokens_slice = demo["tokens"][obs_start:obs_end].copy()
            else:
                imgs_slice = demo["images"][obs_start:obs_end].copy()
            T_demo = demo["actions"].shape[0]
            act_start = max(0, t)
            end = min(t + T_pred, T_demo)
            actions_raw = demo["actions"][act_start:end].copy()
        else:
            f = h5py.File(self._hdf5_paths[file_idx], "r")
            grp = f[f"data/{demo_key}"]
            obs_start = max(0, t - T_obs + 1)
            obs_end = max(0, t) + 1
            proprio_slice = grp["proprio"][obs_start:obs_end]
            if is_cached:
                tokens_slice = grp["tokens"][obs_start:obs_end]
            else:
                imgs_slice = grp["images"][obs_start:obs_end]
            T_demo = grp["actions"].shape[0]
            act_start = max(0, t)
            end = min(t + T_pred, T_demo)
            actions_raw = grp["actions"][act_start:end]
            f.close()

        # --- Obs padding: repeat first available frame ---
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

        # Pad front if t < 0 (repeat first action)
        if t < 0:
            front_pad = min(-t, T_pred)
            actions_raw = np.concatenate(
                [np.repeat(actions_raw[:1], front_pad, axis=0), actions_raw],
                axis=0,
            )[:T_pred]

        # Pad with last action if beyond demo end (pad_after)
            if actions_raw.shape[0] < T_pred:
                pad_len = T_pred - actions_raw.shape[0]
                actions_raw = np.concatenate(
                    [actions_raw, np.repeat(actions_raw[-1:], pad_len, axis=0)],
                    axis=0,
                )

        # --- Convert 7D → 10D if using rot6d representation ---
        if self.use_rot6d:
            actions_raw = convert_actions_to_rot6d(actions_raw)

        # --- Normalize actions and proprio ---
        norm = self._norm_per_file[file_idx]
        actions = self._normalize_actions(actions_raw, norm["actions"])
        proprio = self._normalize(proprio_raw, norm["proprio"])

        # Use torch.tensor (not torch.from_numpy) to ensure resizable storage
        # for DataLoader collation with multiprocessing workers
        result = {
            "actions":      torch.tensor(actions.astype(np.float32)),
            "proprio":      torch.tensor(proprio.astype(np.float32)),
            "view_present": torch.tensor(self._view_present_per_file[file_idx]),
        }

        if is_cached:
            # Return precomputed tokens (cast float16 -> float32)
            result["cached_tokens"] = torch.tensor(tokens_raw.astype(np.float32))
        else:
            # Process images
            if imgs_raw.dtype == np.uint8:
                imgs_01 = imgs_raw.astype(np.float32) / 255.0
            else:
                imgs_01 = imgs_raw.astype(np.float32)

            # Raw target: HWC -> CHW for co-training L_recon
            images_target = np.moveaxis(imgs_01, -1, -3)  # (T_obs, K, 3, H, W)

            # Chi's normalization: [0,1] -> [-1,1] -> ImageNet norm
            imgs_neg11 = imgs_01 * 2.0 - 1.0
            images_enc = (imgs_neg11 - _IMAGENET_MEAN) / _IMAGENET_STD
            images_enc = np.moveaxis(images_enc, -1, -3)  # (T_obs, K, 3, H, W)

            result["images_enc"] = torch.tensor(np.ascontiguousarray(images_enc))
            result["images_target"] = torch.tensor(np.ascontiguousarray(images_target))

        return result

    def _normalize_actions(self, x: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize actions. In 'chi' mode, only position dims get minmax."""
        if self.norm_mode == "chi":
            # Chi's approach: position [0:3] = minmax [-1,1], rest = identity
            result = x.copy()
            pos_min = stats["min"][:3]
            pos_max = stats["max"][:3]
            pos_range = np.clip(pos_max - pos_min, 1e-6, None)
            result[..., :3] = 2.0 * (x[..., :3] - pos_min) / pos_range - 1.0
            return result
        return self._normalize(x, stats)

    def _normalize(self, x: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize using zscore or minmax. Chi mode uses minmax for non-action fields."""
        if self.norm_mode in ("minmax", "chi"):
            a_range = np.clip(stats["max"] - stats["min"], 1e-6, None)
            return 2.0 * (x - stats["min"]) / a_range - 1.0
        else:
            return (x - stats["mean"]) / np.clip(stats["std"], 1e-6, None)
