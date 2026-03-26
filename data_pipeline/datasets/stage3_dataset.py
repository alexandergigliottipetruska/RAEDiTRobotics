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
from data_pipeline.utils.rotation import convert_actions_to_rot6d, convert_actions_quat_to_rot6d

# ImageNet normalization constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def worker_init_open_handles(worker_id):
    """DataLoader worker_init_fn: open persistent HDF5 handles per worker."""
    import torch.utils.data
    ds = torch.utils.data.get_worker_info().dataset
    ds._open_handles()


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
        norm_mode:  "zscore", "minmax", or "chi" for action/proprio normalization.
                    use_rot6d=True requires norm_mode="chi" because the HDF5
                    stores norm stats in the original format (7D/8D), not 10D.
        use_rot6d:  If True, convert actions to 10D rot6d in __getitem__:
                    robomimic 7D (aa) → 10D, RLBench 8D (quat) → 10D.
                    Norm stats stay in original dims — chi mode only uses
                    position min/max [0:3] which are the same in all formats.
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
        if use_rot6d and norm_mode != "chi":
            raise ValueError(
                "use_rot6d=True requires norm_mode='chi' — stored norm stats are "
                "in original action format (7D/8D), not 10D rot6d"
            )

        # Build flat index and load norm stats per file
        self._index = []  # list of (file_idx, demo_key, t)
        self._view_present_per_file = []
        self._norm_per_file = []  # per-file norm stats (action/proprio dims may differ)
        self._cached_per_file = []  # bool per file: True if tokens are precomputed
        self._benchmark_per_file = []  # "robomimic" or "rlbench" per file

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

                # Read benchmark for rot6d conversion branching
                benchmark = f.attrs.get("benchmark", "robomimic")
                if isinstance(benchmark, bytes):
                    benchmark = benchmark.decode()
                self._benchmark_per_file.append(benchmark)

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

    def __len__(self) -> int:
        return len(self._index)

    def _open_handles(self):
        """Open persistent HDF5 file handles (called from worker_init_fn)."""
        self._handles = [h5py.File(p, "r") for p in self._hdf5_paths]

    def _close_handles(self):
        """Close persistent HDF5 file handles."""
        if hasattr(self, '_handles') and self._handles:
            for h in self._handles:
                h.close()
            self._handles = None

    def _get_grp(self, file_idx, demo_key):
        """Get HDF5 group, using persistent handle if available."""
        if hasattr(self, '_handles') and self._handles:
            return self._handles[file_idx][f"data/{demo_key}"], None
        f = h5py.File(self._hdf5_paths[file_idx], "r")
        return f[f"data/{demo_key}"], f

    def __getitem__(self, idx: int) -> dict:
        file_idx, demo_key, t = self._index[idx]
        T_obs, T_pred = self.T_obs, self.T_pred
        is_cached = self._cached_per_file[file_idx]

        grp, f_handle = self._get_grp(file_idx, demo_key)

        # --- Observations: frames [t - T_obs + 1, ..., t] ---
        obs_start = max(0, t - T_obs + 1)
        obs_end = max(0, t) + 1  # handle t < 0: clamp to frame 0
        proprio_slice = grp["proprio"][obs_start : obs_end]   # (<=T_obs, D_prop)

        if is_cached:
            tokens_slice = grp["tokens"][obs_start : obs_end]  # (<=T_obs, K_active, 196, 1024)
            # Compact tokens: only active views stored, pad back to full K
            if "active_cam_indices" in grp:
                K_full = int(grp.file.attrs.get("num_cam_slots", 4))
                if tokens_slice.shape[1] < K_full:
                    active_idx = grp["active_cam_indices"][:]
                    padded = np.zeros(
                        (*tokens_slice.shape[:1], K_full, *tokens_slice.shape[2:]),
                        dtype=tokens_slice.dtype,
                    )
                    padded[:, active_idx] = tokens_slice
                    tokens_slice = padded
        else:
            imgs_slice = grp["images"][obs_start : obs_end]    # (<=T_obs, K, H, W, 3)

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
        T_demo = grp["actions"].shape[0]
        act_start = max(0, t)
        end = min(t + T_pred, T_demo)
        actions_raw = grp["actions"][act_start : end]  # (<=T_pred, D_act)

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

        # Close file handle if not persistent
        if f_handle is not None:
            f_handle.close()

        # --- Convert to 10D rot6d if using rot6d representation ---
        if self.use_rot6d:
            benchmark = self._benchmark_per_file[file_idx]
            if benchmark == "rlbench":
                actions_raw = convert_actions_quat_to_rot6d(actions_raw)   # 8D quat → 10D
            else:
                actions_raw = convert_actions_to_rot6d(actions_raw)        # 7D aa → 10D

        # --- Normalize actions and proprio ---
        norm = self._norm_per_file[file_idx]
        actions = self._normalize_actions(actions_raw, norm["actions"])
        proprio = self._normalize_proprio(proprio_raw, norm["proprio"])

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

            # Chi's normalization: [0,1] → [-1,1] → ImageNet norm
            # LinearNormalizer maps to [-1,1] before ImageNet norm is applied
            imgs_neg11 = imgs_01 * 2.0 - 1.0
            images_enc = (imgs_neg11 - _IMAGENET_MEAN) / _IMAGENET_STD
            images_enc = np.moveaxis(images_enc, -1, -3)  # (T_obs, K, 3, H, W)

            result["images_enc"] = torch.from_numpy(images_enc)
            result["images_target"] = torch.from_numpy(images_target)

        return result

    def _normalize_actions(self, x: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize actions. In 'chi' mode, only position dims get minmax.

        After rot6d conversion, both robomimic and RLBench have 10D actions:
        [pos(3), rot6d(6), grip(1)]. Chi normalizes pos[0:3] only.
        Without rot6d, robomimic=7D and RLBench=8D — chi still normalizes [0:3].
        """
        if self.norm_mode == "chi":
            # Chi's approach: position [0:3] = minmax [-1,1], rest = identity
            assert x.shape[-1] in (7, 8, 10), (
                f"chi action norm: unexpected dim {x.shape[-1]}, "
                f"expected 7 (robomimic raw), 8 (rlbench raw), or 10 (rot6d)"
            )
            result = x.copy()
            pos_min = stats["min"][:3]
            pos_max = stats["max"][:3]
            pos_range = np.clip(pos_max - pos_min, 1e-6, None)
            result[..., :3] = 2.0 * (x[..., :3] - pos_min) / pos_range - 1.0
            return result
        return self._normalize(x, stats)

    def _normalize_proprio(self, x: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize proprio. In 'chi' mode: pos+grip minmax, quat identity.

        Layout for both robomimic (9D) and RLBench (8D):
          [0:3] = eef_pos → minmax
          [3:7] = eef_quat → identity
          [7:]  = gripper → minmax (robomimic 2D, RLBench 1D)
        """
        if self.norm_mode == "chi":
            assert x.shape[-1] in (8, 9), (
                f"chi proprio norm: unexpected dim {x.shape[-1]}, "
                f"expected 8 (rlbench) or 9 (robomimic)"
            )
            result = x.copy()
            # pos [0:3] — minmax [-1, 1]
            pos_min, pos_max = stats["min"][:3], stats["max"][:3]
            pos_range = np.clip(pos_max - pos_min, 1e-6, None)
            result[..., :3] = 2.0 * (x[..., :3] - pos_min) / pos_range - 1.0
            # quat [3:7] — identity (no normalization)
            # grip [7:9] — minmax [-1, 1]
            grip_min, grip_max = stats["min"][7:9], stats["max"][7:9]
            grip_range = np.clip(grip_max - grip_min, 1e-6, None)
            result[..., 7:9] = 2.0 * (x[..., 7:9] - grip_min) / grip_range - 1.0
            return result
        return self._normalize(x, stats)

    def _normalize(self, x: np.ndarray, stats: dict) -> np.ndarray:
        """Normalize using zscore or minmax. Chi mode uses minmax for non-action fields."""
        if self.norm_mode in ("minmax", "chi"):
            a_range = np.clip(stats["max"] - stats["min"], 1e-6, None)
            return 2.0 * (x - stats["min"]) / a_range - 1.0
        else:
            return (x - stats["mean"]) / np.clip(stats["std"], 1e-6, None)
