"""Convert robomimic HDF5 demos to unified schema.

Camera mapping:
  agentview            -> slot 0 (front)
  robot0_eye_in_hand   -> slot 3 (wrist)
  slots 1, 2           -> zeros, view_present=False

Proprio: [eef_pos(3), eef_quat(4), gripper_qpos(2)] = 9D
Actions: copied directly (already 7D delta-EE from OSC_POSE controller).
Images:  resized 84x84 -> 224x224, stored as uint8 [0,255].

Usage:
  python -m data_pipeline.conversion.convert_robomimic \
      --task lift --variant ph --config data_pipeline/configs/paths.yaml
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

from data_pipeline.conversion.unified_schema import (
    NUM_CAMERA_SLOTS,
    IMAGE_SIZE,
    create_unified_hdf5,
    create_demo_group,
    write_mask,
)
from data_pipeline.conversion.compute_norm_stats import compute_and_save_norm_stats

# Default single-arm config
_SINGLE_ARM_PROPRIO_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
_SINGLE_ARM_SLOT_MAP = {
    "agentview_image": 0,
    "robot0_eye_in_hand_image": 3,
}

# Per-task overrides (tasks not listed use single-arm defaults)
TASK_CONFIG = {
    "transport": {
        "action_dim": 14,  # [robot0_7D | robot1_7D]
        "proprio_keys": [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
            "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos",
        ],
        "proprio_dim": 18,  # 9D × 2 robots
        "slot_map": _SINGLE_ARM_SLOT_MAP,  # same 2 cameras as single-arm
    },
}

_DEFAULT_CONFIG = {
    "action_dim": 7,
    "proprio_keys": _SINGLE_ARM_PROPRIO_KEYS,
    "proprio_dim": 9,
    "slot_map": _SINGLE_ARM_SLOT_MAP,
}


def _get_task_config(task: str) -> dict:
    return TASK_CONFIG.get(task, _DEFAULT_CONFIG)


def load_paths(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resize_image(img_uint8: np.ndarray) -> np.ndarray:
    """Resize a single HxWx3 uint8 image to IMAGE_SIZE, return uint8."""
    H, W = IMAGE_SIZE
    return cv2.resize(img_uint8, (W, H), interpolation=cv2.INTER_LINEAR)


def _convert_demo(
    src_grp: h5py.Group,
    dst_grp: h5py.Group,
    cfg: dict,
) -> None:
    """Convert a single demo from robomimic format into a pre-created unified group."""
    T = src_grp["actions"].shape[0]
    H, W = IMAGE_SIZE
    K = NUM_CAMERA_SLOTS
    slot_map = cfg["slot_map"]

    # --- Images ---
    # dst_grp["images"] already allocated as (T, K, H, W, 3) uint8
    img_buf = np.zeros((T, K, H, W, 3), dtype=np.uint8)
    for src_key, slot in slot_map.items():
        raw = src_grp[f"obs/{src_key}"][:]  # (T, 84, 84, 3) uint8
        for t in range(T):
            img_buf[t, slot] = _resize_image(raw[t])
    dst_grp["images"][:] = img_buf

    # --- view_present ---
    view_present = np.zeros(K, dtype=bool)
    for slot in slot_map.values():
        view_present[slot] = True
    dst_grp["view_present"][:] = view_present

    # --- Actions (direct copy) ---
    dst_grp["actions"][:] = src_grp["actions"][:].astype(np.float32)

    # --- Proprio ---
    parts = [src_grp[f"obs/{k}"][:].astype(np.float32) for k in cfg["proprio_keys"]]
    dst_grp["proprio"][:] = np.concatenate(parts, axis=1)


def convert_task(raw_hdf5_path: str, output_path: str, task: str = "lift") -> None:
    """Convert a single robomimic task HDF5 to unified format.

    Args:
        raw_hdf5_path: Path to the source robomimic image.hdf5
        output_path:   Path for the output unified HDF5
        task:          Task name string stored in metadata (e.g. "lift")
    """
    cfg = _get_task_config(task)
    proprio_dim = cfg["proprio_dim"]
    action_dim = cfg["action_dim"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(raw_hdf5_path, "r") as src:
        # Read train/valid splits from source mask groups
        train_keys = [
            k.decode() if isinstance(k, bytes) else k
            for k in src["mask"]["train"][:]
        ]
        valid_keys = [
            k.decode() if isinstance(k, bytes) else k
            for k in src["mask"]["valid"][:]
        ]
        all_keys = train_keys + valid_keys

        print(f"Converting {len(train_keys)} train + {len(valid_keys)} valid demos")
        print(f"  Task:    {task} (action_dim={action_dim}, proprio_dim={proprio_dim})")
        print(f"  Source:  {raw_hdf5_path}")
        print(f"  Output:  {output_path}")

        with create_unified_hdf5(str(output_path), "robomimic", task, proprio_dim, action_dim) as dst:
            for i, demo_key in enumerate(all_keys):
                src_grp = src[f"data/{demo_key}"]
                T = src_grp["actions"].shape[0]
                dst_grp = create_demo_group(dst, demo_key, T, proprio_dim, action_dim=action_dim, image_dtype=np.uint8)
                _convert_demo(src_grp, dst_grp, cfg)
                if (i + 1) % 20 == 0 or (i + 1) == len(all_keys):
                    print(f"  [{i+1}/{len(all_keys)}] done")

            # Write mask groups
            write_mask(dst, "train", train_keys)
            write_mask(dst, "valid", valid_keys)

            # Compute and save norm stats from training demos only
            print("Computing normalization stats from train split...")
            stats = compute_and_save_norm_stats(dst, train_keys)

    print("Conversion complete.")
    print(f"  Action mean: {stats['actions']['mean'].round(4)}")
    print(f"  Action std:  {stats['actions']['std'].round(4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert robomimic HDF5 to unified schema")
    parser.add_argument("--task", default="lift", help="Task name (e.g. lift, can, square)")
    parser.add_argument("--variant", default="ph", help="Dataset variant (ph or mh)")
    parser.add_argument("--config", default="data_pipeline/configs/paths.yaml")
    args = parser.parse_args()

    paths = load_paths(args.config)
    raw_hdf5 = Path(paths["robomimic_raw"]) / args.task / args.variant / "image.hdf5"
    output_hdf5 = Path(paths["unified_data_dir"]) / "robomimic" / args.task / f"{args.variant}.hdf5"

    convert_task(str(raw_hdf5), str(output_hdf5), task=args.task)
