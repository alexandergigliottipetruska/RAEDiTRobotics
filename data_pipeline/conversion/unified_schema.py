"""Unified HDF5 schema. SINGLE SOURCE OF TRUTH for data format.
Both conversion scripts and the dataset class import from here.

Schema per demo group (data/<demo_key>/):
  images      uint8  [T, K, H, W, 3]   [0,255] for all benchmarks
                base_dataset.py converts to float32 and ImageNet-normalizes at load time.
  view_present bool    [K]                  True if camera slot has real data
  actions     float32  [T, 7]               delta-EE: [dx,dy,dz,drx,dry,drz,gripper]
  proprio     float32  [T, D_prop]          benchmark-specific (dim stored in attrs)
  states      float32  [T, D_state]         (optional) simulator state for GT replay/eval

File-level attrs:
  benchmark       str    e.g. "robomimic"
  task            str    e.g. "lift"
  proprio_dim     int    D_prop for this file
  action_dim      int    always 7
  image_size      int    always 224
  num_cam_slots   int    always 4

Norm stats saved under /norm_stats/ group:
  norm_stats/actions/{mean,std,min,max}   float32  [action_dim]
  norm_stats/proprio/{mean,std,min,max}   float32  [D_prop]
"""

import h5py
import numpy as np

NUM_CAMERA_SLOTS = 4
IMAGE_SIZE = (224, 224)
ACTION_DIM = 7
VALID_BENCHMARKS = ["robomimic", "rlbench"]


def create_unified_hdf5(
    output_path: str,
    benchmark: str,
    task: str,
    proprio_dim: int,
    action_dim: int = ACTION_DIM,
) -> h5py.File:
    """Open a new unified HDF5 file and write file-level metadata attrs.

    Returns an open h5py.File in write mode. Caller must close it.
    """
    assert benchmark in VALID_BENCHMARKS, f"Unknown benchmark: {benchmark}"
    f = h5py.File(output_path, "w")
    f.attrs["benchmark"] = benchmark
    f.attrs["task"] = task
    f.attrs["proprio_dim"] = proprio_dim
    f.attrs["action_dim"] = action_dim
    f.attrs["image_size"] = IMAGE_SIZE[0]
    f.attrs["num_cam_slots"] = NUM_CAMERA_SLOTS
    # Create mask group to hold train/valid demo key lists
    f.create_group("mask")
    f.create_group("data")
    return f


def create_demo_group(
    hdf5_file: h5py.File,
    demo_key: str,
    T: int,
    D_prop: int,
    compress: bool = True,
    action_dim: int = ACTION_DIM,
    image_dtype: np.dtype = np.float32,
    state_dim: int | None = None,
) -> h5py.Group:
    """Create a demo group with pre-allocated datasets of the correct shapes.

    Args:
        hdf5_file:   Open h5py.File in write mode.
        demo_key:    e.g. "demo_0"
        T:           Number of timesteps in this demo.
        D_prop:      Proprio vector dimension.
        compress:    Whether to use gzip compression (recommended for images).
        image_dtype: np.float32 (robomimic/maniskill) or np.uint8 (rlbench).
        state_dim:   If not None, create a states dataset [T, state_dim] for GT replay.

    Returns:
        The newly created h5py.Group at data/<demo_key>.
    """
    H, W = IMAGE_SIZE
    K = NUM_CAMERA_SLOTS

    grp = hdf5_file["data"].create_group(demo_key)

    kwargs = {"compression": "gzip", "compression_opts": 4} if compress else {}

    # Images: float32 [0,1] for robomimic/maniskill; uint8 [0,255] for rlbench
    grp.create_dataset(
        "images",
        shape=(T, K, H, W, 3),
        dtype=image_dtype,
        **kwargs,
    )

    # view_present: one bool per camera slot (constant for a given benchmark)
    grp.create_dataset(
        "view_present",
        shape=(K,),
        dtype=bool,
    )

    # Actions: 7D delta-EE (or 14D for two-arm tasks)
    grp.create_dataset(
        "actions",
        shape=(T, action_dim),
        dtype=np.float32,
        **kwargs,
    )

    # Proprio
    grp.create_dataset(
        "proprio",
        shape=(T, D_prop),
        dtype=np.float32,
        **kwargs,
    )

    # States (optional): simulator state for GT replay and deterministic eval
    if state_dim is not None:
        grp.create_dataset(
            "states",
            shape=(T, state_dim),
            dtype=np.float32,
        )

    return grp


def write_mask(hdf5_file: h5py.File, split: str, demo_keys: list[str]) -> None:
    """Write a list of demo keys into mask/<split> as variable-length ASCII strings."""
    encoded = [k.encode("ascii") for k in demo_keys]
    dt = h5py.special_dtype(vlen=str)
    ds = hdf5_file["mask"].create_dataset(split, shape=(len(encoded),), dtype=dt)
    ds[:] = encoded


def read_mask(hdf5_file: h5py.File, split: str) -> list[str]:
    """Read demo keys from mask/<split>."""
    return [
        k.decode() if isinstance(k, bytes) else k
        for k in hdf5_file["mask"][split][:]
    ]
