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

from PIL import Image

from data_pipeline.conversion.unified_schema import read_mask
from data_pipeline.conversion.compute_norm_stats import load_norm_stats
from data_pipeline.utils.rotation import convert_actions_to_rot6d, convert_actions_quat_to_rot6d

# ImageNet normalization constants (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _random_crop_resize(img_hwc: np.ndarray, crop_size: int = 208,
                        target_size: int = 224) -> np.ndarray:
    """Random crop then resize back to target size. Applied per camera view.

    Args:
        img_hwc: uint8 (H, W, 3) image, H=W=224
        crop_size: crop to this size (208 ≈ 93% of 224, similar to Chi's 76/84 ≈ 90%)
        target_size: resize back to this (224 for ViT patch grid)

    Returns:
        uint8 (target_size, target_size, 3) image
    """
    H, W = img_hwc.shape[:2]
    top = np.random.randint(0, H - crop_size + 1)
    left = np.random.randint(0, W - crop_size + 1)
    cropped = img_hwc[top:top + crop_size, left:left + crop_size]
    resized = np.array(
        Image.fromarray(cropped).resize((target_size, target_size), Image.LANCZOS)
    )
    return resized


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
        image_hdf5_path: str = "",
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
        self._image_hdf5_path = image_hdf5_path
        self._refreshed_tokens = {}  # (file_idx, demo_key) → np.ndarray fp16

        if norm_mode not in ("zscore", "minmax", "chi", "minmax_margin"):
            raise ValueError(f"norm_mode must be 'zscore', 'minmax', 'chi', or 'minmax_margin', got '{norm_mode}'")
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

        # RAM cache: disabled by default, enabled via cache_in_ram() or cache_as_torch_tensors()
        self._ram_cache = None  # dict of numpy arrays (cache_in_ram)
        self._torch_cache = None  # dict of torch tensors (cache_as_torch_tensors)

    def cache_in_ram(self):
        """Pre-load all demo data into RAM to eliminate HDF5 I/O during training.

        Call after construction: `ds.cache_in_ram()`. Loads demos one-by-one
        to avoid peak memory spikes. Tokens stay in compact form (K_active only).
        """
        import logging, gc
        log = logging.getLogger(__name__)
        log.info("Caching dataset in RAM...")
        self._ram_cache = {}
        total_bytes = 0
        n_demos = 0

        for file_idx, path in enumerate(self._hdf5_paths):
            demo_keys = sorted(set(k for fi, k, t in self._index if fi == file_idx))
            f = h5py.File(path, "r")
            for demo_key in demo_keys:
                grp = f[f"data/{demo_key}"]
                entry = {
                    "actions": np.array(grp["actions"]),
                    "proprio": np.array(grp["proprio"]),
                }
                if self._cached_per_file[file_idx]:
                    entry["tokens"] = np.array(grp["tokens"])
                    if "active_cam_indices" in grp:
                        entry["active_cam_indices"] = np.array(grp["active_cam_indices"])
                        entry["K_full"] = int(f.attrs.get("num_cam_slots", 4))
                else:
                    entry["images"] = np.array(grp["images"])
                total_bytes += sum(v.nbytes for v in entry.values() if isinstance(v, np.ndarray))
                self._ram_cache[(file_idx, demo_key)] = entry
                n_demos += 1

                if n_demos % 20 == 0:
                    log.info("  Cached %d demos (%.1f GB)...", n_demos, total_bytes / 1e9)
            f.close()

        gc.collect()
        log.info("Cached %d demos (%.1f GB) in RAM", n_demos, total_bytes / 1e9)

    def cache_as_torch_tensors(self):
        """Pre-load and pre-process all samples as torch tensors.

        This is the fastest cache mode: rot6d conversion, normalization, and
        temporal padding are done once at init. __getitem__ returns tensor views
        (~5 us per sample). Tokens stay compact (K_active cameras only).

        Use with num_workers=0 to avoid IPC overhead.
        """
        import logging, gc
        log = logging.getLogger(__name__)
        log.info("Building torch tensor cache (compact, pre-normalized)...")

        # First pass: load all demo data and pre-process
        demo_cache = {}  # (file_idx, demo_key) -> {tokens, actions, proprio}
        total_bytes = 0

        for file_idx, path in enumerate(self._hdf5_paths):
            demo_keys = sorted(set(k for fi, k, t in self._index if fi == file_idx))
            is_cached = self._cached_per_file[file_idx]
            benchmark = self._benchmark_per_file[file_idx]
            norm = self._norm_per_file[file_idx]

            f = h5py.File(path, "r")
            for demo_key in demo_keys:
                grp = f[f"data/{demo_key}"]

                # Actions: load, convert rot6d, normalize — all at once for the full demo
                actions_np = np.array(grp["actions"], dtype=np.float32)
                if self.use_rot6d:
                    if benchmark == "rlbench":
                        actions_np = convert_actions_quat_to_rot6d(actions_np)
                    else:
                        actions_np = convert_actions_to_rot6d(actions_np)
                actions_np = self._normalize_actions(actions_np, norm["actions"]).astype(np.float32)

                # Proprio: normalize full demo
                proprio_np = self._normalize_proprio(
                    np.array(grp["proprio"], dtype=np.float32), norm["proprio"]
                ).astype(np.float32)

                entry = {
                    "actions": torch.from_numpy(actions_np),
                    "proprio": torch.from_numpy(proprio_np),
                }

                if is_cached:
                    entry["tokens"] = torch.from_numpy(np.array(grp["tokens"]))  # preserve dtype (fp32 or bf16/fp16)
                    if "active_cam_indices" in grp:
                        entry["active_cam_indices"] = torch.from_numpy(np.array(grp["active_cam_indices"]))
                        entry["K_full"] = int(f.attrs.get("num_cam_slots", 4))

                total_bytes += sum(t.nelement() * t.element_size() for t in entry.values() if isinstance(t, torch.Tensor))
                demo_cache[(file_idx, demo_key)] = entry
            f.close()

        self._torch_cache = demo_cache
        gc.collect()
        log.info("Torch tensor cache ready: %d demos, %.1f GB", len(demo_cache), total_bytes / 1e9)

    def __len__(self) -> int:
        return len(self._index)

    def _open_handles(self):
        """Open persistent HDF5 file handles (called from worker_init_fn)."""
        self._handles = [h5py.File(p, "r") for p in self._hdf5_paths]
        if self._image_hdf5_path:
            self._image_handle = h5py.File(self._image_hdf5_path, "r")

    def _close_handles(self):
        """Close persistent HDF5 file handles."""
        if hasattr(self, '_handles') and self._handles:
            for h in self._handles:
                h.close()
            self._handles = None
        if hasattr(self, '_image_handle') and self._image_handle:
            self._image_handle.close()
            self._image_handle = None

    @torch.no_grad()
    def refresh_cached_tokens(self, encoder, device, crop_size=208):
        """Re-encode TRAINING images with random crop augmentation.

        Called every N epochs from the training loop. Only refreshes demos
        that belong to this dataset instance (training split). Validation
        datasets should never call this.

        Args:
            encoder: FrozenMultiViewEncoder (frozen DINOv3-L). Its forward()
                     already applies cancel_affine_ln internally.
            device: torch.device for encoder forward pass.
            crop_size: crop 224→this size, then resize back to 224.
        """
        import logging
        log = logging.getLogger(__name__)

        if not self._image_hdf5_path:
            raise ValueError("image_hdf5_path required for token refresh")

        self._refreshed_tokens.clear()  # free old tokens

        from tqdm import tqdm

        # Collect all demo keys that this dataset instance uses
        demo_keys_by_file = {}
        seen = set()
        for file_idx, demo_key, _t in self._index:
            k = (file_idx, demo_key)
            if k not in seen:
                seen.add(k)
                demo_keys_by_file.setdefault(file_idx, []).append(demo_key)

        # Count total demos for progress bar
        all_demos = [(fi, dk) for fi, dks in demo_keys_by_file.items() for dk in dks]

        total_images = 0
        with h5py.File(self._image_hdf5_path, 'r') as img_f:
            for file_idx, demo_key in tqdm(all_demos, desc="Refreshing tokens", unit="demo"):
                grp_path = f'data/{demo_key}/images'
                if grp_path not in img_f:
                    continue

                imgs_raw = img_f[grp_path][:]  # (T, K, H, W, 3) uint8
                T, K = imgs_raw.shape[:2]

                # Random crop each camera view independently
                imgs_cropped = imgs_raw.copy()
                for t_idx in range(T):
                    for k_idx in range(K):
                        imgs_cropped[t_idx, k_idx] = _random_crop_resize(
                            imgs_cropped[t_idx, k_idx], crop_size, target_size=224)

                # ImageNet normalize: uint8 → [0,1] → [-1,1] → ImageNet norm → CHW
                imgs_01 = imgs_cropped.astype(np.float32) / 255.0
                imgs_neg11 = imgs_01 * 2.0 - 1.0
                imgs_enc = (imgs_neg11 - _IMAGENET_MEAN) / _IMAGENET_STD
                imgs_enc = np.moveaxis(imgs_enc, -1, -3)  # (T, K, 3, H, W)

                # Flatten to (T*K, 3, 224, 224) for batched encoder forward
                flat = imgs_enc.reshape(-1, 3, 224, 224)

                # Process in chunks to avoid OOM
                tokens_list = []
                for i in range(0, len(flat), 32):
                    batch_t = torch.from_numpy(flat[i:i + 32]).to(device)
                    raw = encoder(batch_t)  # (B, 196, 1024) — includes internal LN
                    tokens_list.append(raw.cpu().numpy())

                tokens = np.concatenate(tokens_list).reshape(T, K, 196, -1)
                self._refreshed_tokens[(file_idx, demo_key)] = tokens.astype(np.float16)
                total_images += T * K

        log.info("Refreshed %d demos (%d images) with crop_size=%d",
                 len(self._refreshed_tokens), total_images, crop_size)

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

        # --- Fast path: torch tensor cache (pre-normalized, compact tokens) ---
        if self._torch_cache is not None:
            entry = self._torch_cache[(file_idx, demo_key)]
            T_demo = entry["actions"].shape[0]

            # Obs window with start-padding (repeat first frame)
            obs_start = max(0, t - T_obs + 1)
            obs_end = max(0, t) + 1
            proprio = entry["proprio"][obs_start:obs_end]
            pad_before = T_obs - proprio.shape[0]
            if pad_before > 0:
                proprio = torch.cat([proprio[:1].expand(pad_before, -1), proprio], dim=0)

            # Actions window with front/end padding
            act_start = max(0, t)
            end = min(t + T_pred, T_demo)
            actions = entry["actions"][act_start:end]
            if t < 0:
                front_pad = min(-t, T_pred)
                actions = torch.cat([actions[:1].expand(front_pad, -1), actions], dim=0)[:T_pred]
            if actions.shape[0] < T_pred:
                pad_len = T_pred - actions.shape[0]
                actions = torch.cat([actions, actions[-1:].expand(pad_len, -1)], dim=0)

            result = {
                "actions": actions,
                "proprio": proprio,
                "view_present": torch.from_numpy(self._view_present_per_file[file_idx]),
            }

            if "tokens" in entry:
                tokens = entry["tokens"][obs_start:obs_end]
                if pad_before > 0:
                    tokens = torch.cat([tokens[:1].expand(pad_before, *tokens.shape[1:]), tokens], dim=0)
                result["cached_tokens"] = tokens
                # Pass compact token metadata for GPU zero-pad
                if "active_cam_indices" in entry:
                    result["active_cam_indices"] = entry["active_cam_indices"]
                    result["K_full"] = torch.tensor(entry["K_full"], dtype=torch.long)

            return result

        # --- Standard path: RAM cache or HDF5 ---
        cache_entry = self._ram_cache.get((file_idx, demo_key)) if self._ram_cache else None

        if cache_entry is None:
            grp, f_handle = self._get_grp(file_idx, demo_key)
        else:
            grp = cache_entry  # dict with numpy arrays, same key names
            f_handle = None

        # --- Observations: frames [t - T_obs + 1, ..., t] ---
        obs_start = max(0, t - T_obs + 1)
        obs_end = max(0, t) + 1  # handle t < 0: clamp to frame 0
        proprio_slice = grp["proprio"][obs_start : obs_end]   # (<=T_obs, D_prop)

        refresh_key = (file_idx, demo_key)
        use_refreshed = refresh_key in self._refreshed_tokens

        if is_cached and use_refreshed:
            # Use augmented tokens from periodic refresh (training only)
            all_tokens = self._refreshed_tokens[refresh_key]
            tokens_slice = all_tokens[obs_start:obs_end].astype(np.float32)
        elif is_cached:
            tokens_slice = grp["tokens"][obs_start : obs_end]  # (<=T_obs, K_active, 196, 1024)
            # Compact tokens: only active views stored, pad back to full K
            if cache_entry is not None:
                # RAM cache: active_cam_indices stored in entry
                if "active_cam_indices" in cache_entry:
                    K_full = cache_entry["K_full"]
                    if tokens_slice.shape[1] < K_full:
                        active_idx = cache_entry["active_cam_indices"]
                        padded = np.zeros(
                            (*tokens_slice.shape[:1], K_full, *tokens_slice.shape[2:]),
                            dtype=tokens_slice.dtype,
                        )
                        padded[:, active_idx] = tokens_slice
                        tokens_slice = padded
            elif "active_cam_indices" in grp:
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

            # Co-training: load raw images from original HDF5 for reconstruction target
            if self._image_hdf5_path:
                if hasattr(self, '_image_handle') and self._image_handle:
                    img_f = self._image_handle
                    _close_img = False
                else:
                    img_f = h5py.File(self._image_hdf5_path, 'r')
                    _close_img = True
                try:
                    img_grp = img_f[f'data/{demo_key}']
                    imgs_co = img_grp["images"][obs_start:obs_end]  # (<=T_obs, K, H, W, 3)
                    if pad_before > 0:
                        imgs_co = np.concatenate(
                            [np.repeat(imgs_co[:1], pad_before, axis=0), imgs_co],
                            axis=0,
                        )
                    if imgs_co.dtype == np.uint8:
                        imgs_01 = imgs_co.astype(np.float32) / 255.0
                    else:
                        imgs_01 = imgs_co.astype(np.float32)
                    images_target = np.moveaxis(imgs_01, -1, -3)  # (T_obs, K, 3, H, W)
                    result["images_target"] = torch.from_numpy(images_target)
                finally:
                    if _close_img:
                        img_f.close()
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
        """Normalize using zscore, minmax, or minmax_margin."""
        if self.norm_mode == "minmax_margin":
            # Robobase-style: expand bounds by 20% margin, normalize ALL dims to [-1,1]
            margin = 0.2
            a_min = stats["min"] - np.abs(stats["min"]) * margin
            a_max = stats["max"] + np.abs(stats["max"]) * margin
            a_range = np.clip(a_max - a_min, 1e-6, None)
            return 2.0 * (x - a_min) / a_range - 1.0
        elif self.norm_mode in ("minmax", "chi"):
            a_range = np.clip(stats["max"] - stats["min"], 1e-6, None)
            return 2.0 * (x - stats["min"]) / a_range - 1.0
        else:
            return (x - stats["mean"]) / np.clip(stats["std"], 1e-6, None)
