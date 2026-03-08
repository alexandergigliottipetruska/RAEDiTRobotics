"""Compute per-dimension normalization statistics.

Rules:
  - Computed on TRAINING split only. Never val/test.
  - Saved into the unified HDF5 file under /norm_stats/ group.
  - Used by the dataset class for normalization at load time.

Saved layout inside unified HDF5:
  /norm_stats/actions/mean   float32  [ACTION_DIM]
  /norm_stats/actions/std    float32  [ACTION_DIM]
  /norm_stats/actions/min    float32  [ACTION_DIM]
  /norm_stats/actions/max    float32  [ACTION_DIM]
  /norm_stats/proprio/mean   float32  [D_prop]
  /norm_stats/proprio/std    float32  [D_prop]
  /norm_stats/proprio/min    float32  [D_prop]
  /norm_stats/proprio/max    float32  [D_prop]

Two normalization modes are supported:
  - "zscore": (x - mean) / std          -> unbounded
  - "minmax": 2*(x - min)/(max - min) - 1  -> [-1, 1]

std and range values are clipped to a minimum of 1e-6 to avoid division
by zero for dimensions that are constant.
"""

import h5py
import numpy as np

STD_MIN = 1e-6
RANGE_MIN = 1e-6


def compute_and_save_norm_stats(unified_hdf5: h5py.File, train_demo_keys: list) -> dict:
    """Compute per-dim mean/std/min/max from training demos and save to HDF5.

    Args:
        unified_hdf5:    Open h5py.File (already contains converted demo data).
        train_demo_keys: List of demo key strings that belong to the train split.

    Returns:
        dict with structure:
            {
              "actions": {"mean": ..., "std": ..., "min": ..., "max": ...},
              "proprio": {"mean": ..., "std": ..., "min": ..., "max": ...},
            }
    """
    all_actions = []
    all_proprio = []

    for key in train_demo_keys:
        grp = unified_hdf5[f"data/{key}"]
        all_actions.append(grp["actions"][:])
        all_proprio.append(grp["proprio"][:])

    all_actions = np.concatenate(all_actions, axis=0)  # [N_total, action_dim]
    all_proprio = np.concatenate(all_proprio, axis=0)  # [N_total, D_prop]

    stats = {}
    for name, data in [("actions", all_actions), ("proprio", all_proprio)]:
        stats[name] = {
            "mean": data.mean(axis=0).astype(np.float32),
            "std": np.clip(data.std(axis=0), STD_MIN, None).astype(np.float32),
            "min": data.min(axis=0).astype(np.float32),
            "max": data.max(axis=0).astype(np.float32),
        }

    # Write to /norm_stats/ group (overwrite if exists)
    if "norm_stats" in unified_hdf5:
        del unified_hdf5["norm_stats"]
    ns = unified_hdf5.create_group("norm_stats")

    for field, field_stats in stats.items():
        grp = ns.create_group(field)
        for key, val in field_stats.items():
            grp.create_dataset(key, data=val)

    return stats


def load_norm_stats(unified_hdf5_path: str) -> dict:
    """Load norm stats from a unified HDF5 file.

    Returns all available stats (mean, std, min, max).
    Backward-compatible: min/max will be None if HDF5 predates this change.
    """
    with h5py.File(unified_hdf5_path, "r") as f:
        ns = f["norm_stats"]
        result = {}
        for field in ("actions", "proprio"):
            grp = ns[field]
            result[field] = {
                "mean": grp["mean"][:],
                "std": grp["std"][:],
                "min": grp["min"][:] if "min" in grp else None,
                "max": grp["max"][:] if "max" in grp else None,
            }
        return result
