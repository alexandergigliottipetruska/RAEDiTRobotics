"""Compute per-dimension normalization statistics (mean, std).

Rules:
  - Computed on TRAINING split only. Never val/test.
  - Saved into the unified HDF5 file under /norm_stats/ group.
  - Used by the dataset class for z-score normalization at load time.

Saved layout inside unified HDF5:
  /norm_stats/actions/mean   float32  [ACTION_DIM]
  /norm_stats/actions/std    float32  [ACTION_DIM]
  /norm_stats/proprio/mean   float32  [D_prop]
  /norm_stats/proprio/std    float32  [D_prop]

std values are clipped to a minimum of 1e-6 to avoid division by zero
for dimensions that are constant (e.g. an unused action dim).
"""

import h5py
import numpy as np
from pathlib import Path

STD_MIN = 1e-6


def compute_and_save_norm_stats(unified_hdf5: h5py.File, train_demo_keys: list) -> dict:
    """Compute per-dim mean/std from training demos and save to the open HDF5 file.

    Args:
        unified_hdf5:    Open h5py.File (already contains converted demo data).
        train_demo_keys: List of demo key strings that belong to the train split.

    Returns:
        dict with structure:
            {
              "actions": {"mean": np.ndarray[7], "std": np.ndarray[7]},
              "proprio": {"mean": np.ndarray[D], "std": np.ndarray[D]},
            }
    """
    all_actions = []
    all_proprio = []

    for key in train_demo_keys:
        grp = unified_hdf5[f"data/{key}"]
        all_actions.append(grp["actions"][:])
        all_proprio.append(grp["proprio"][:])

    all_actions = np.concatenate(all_actions, axis=0)  # [N_total, 7]
    all_proprio = np.concatenate(all_proprio, axis=0)  # [N_total, D_prop]

    stats = {
        "actions": {
            "mean": all_actions.mean(axis=0).astype(np.float32),
            "std": np.clip(all_actions.std(axis=0), STD_MIN, None).astype(np.float32),
        },
        "proprio": {
            "mean": all_proprio.mean(axis=0).astype(np.float32),
            "std": np.clip(all_proprio.std(axis=0), STD_MIN, None).astype(np.float32),
        },
    }

    # Write to /norm_stats/ group (overwrite if exists)
    if "norm_stats" in unified_hdf5:
        del unified_hdf5["norm_stats"]
    ns = unified_hdf5.create_group("norm_stats")

    for field, field_stats in stats.items():
        grp = ns.create_group(field)
        grp.create_dataset("mean", data=field_stats["mean"])
        grp.create_dataset("std", data=field_stats["std"])

    return stats


def load_norm_stats(unified_hdf5_path: str) -> dict:
    """Load norm stats from a unified HDF5 file.

    Returns:
        dict with structure:
            {
              "actions": {"mean": np.ndarray[7], "std": np.ndarray[7]},
              "proprio": {"mean": np.ndarray[D], "std": np.ndarray[D]},
            }
    """
    with h5py.File(unified_hdf5_path, "r") as f:
        ns = f["norm_stats"]
        return {
            "actions": {
                "mean": ns["actions/mean"][:],
                "std": ns["actions/std"][:],
            },
            "proprio": {
                "mean": ns["proprio/mean"][:],
                "std": ns["proprio/std"][:],
            },
        }
