"""Convert robomimic unified HDF5 to joint-space action format.

Reads the existing unified HDF5 (which has `states` containing simulator qpos)
and creates a new HDF5 with:
  - actions:  [T, 8]  = joint_positions(7) + gripper_command(1)
              Action at time t = target joint positions at t+1
  - proprio:  [T, 9]  = joint_positions(7) + gripper_qpos(2)
  - images, view_present, states: copied unchanged

The gripper command is kept as the original [-1, 1] from the EE actions
(not the raw gripper qpos) since that's what robosuite's controller expects.

Usage:
    python -m data_pipeline.conversion.convert_robomimic_joints \
        --input data/robomimic/lift/ph_abs_v15.hdf5 \
        --output data/robomimic/lift/ph_joint.hdf5
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from data_pipeline.conversion.unified_schema import (
    create_unified_hdf5,
    create_demo_group,
    write_mask,
    read_mask,
)
from data_pipeline.conversion.compute_norm_stats import compute_and_save_norm_stats


JOINT_ACTION_DIM = 8   # joint_pos(7) + gripper(1)
JOINT_PROPRIO_DIM = 9  # joint_pos(7) + gripper_qpos(2)


def _extract_joint_actions(states: np.ndarray, ee_actions: np.ndarray) -> np.ndarray:
    """Extract joint-space actions from simulator states.

    The flattened robosuite sim state layout is:
      [time(1), qpos(N), qvel(N-1)]
    For Panda: qpos = [arm_joints(7), gripper(2), object_joints...]
    So arm joints are at states[:, 1:8].

    Args:
        states:     (T, D_state) flattened simulator states
        ee_actions: (T, 7) original EE actions, last dim = gripper command [-1, 1]

    Returns:
        (T, 8) joint actions [joint_pos_target(7), gripper(1)]
    """
    T = states.shape[0]
    joint_pos = states[:, 1:8]   # (T, 7) arm joint positions (skip time at index 0)
    grip_cmd = ee_actions[:, -1:]  # (T, 1) gripper command from original actions

    # Action at time t = joint positions at t+1 (where we want to go)
    # Last action: repeat final joint position (no next state to target)
    target_joints = np.zeros((T, 7), dtype=np.float32)
    target_joints[:-1] = joint_pos[1:]     # t → t+1 for t=0..T-2
    target_joints[-1] = joint_pos[-1]      # repeat last

    return np.concatenate([target_joints, grip_cmd], axis=-1).astype(np.float32)


def _extract_joint_proprio(states: np.ndarray) -> np.ndarray:
    """Extract joint-space proprio from simulator states.

    Flattened state layout: [time(1), qpos(N), qvel(N-1)]
    Panda qpos: [arm(7), gripper(2), object...]
    So arm joints at 1:8, gripper at 8:10.

    Args:
        states: (T, D_state) flattened simulator states

    Returns:
        (T, 9) proprio [joint_pos(7), gripper_qpos(2)]
    """
    joint_pos = states[:, 1:8]     # (T, 7) arm joints (skip time)
    gripper_qpos = states[:, 8:10] # (T, 2) gripper joints
    return np.concatenate([joint_pos, gripper_qpos], axis=-1).astype(np.float32)


def convert_to_joints(input_path: str, output_path: str) -> None:
    """Convert a robomimic unified HDF5 from EE actions to joint-space actions."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src:
        task = src.attrs.get("task", "lift")
        if isinstance(task, bytes):
            task = task.decode()

        train_keys = read_mask(src, "train")
        valid_keys = read_mask(src, "valid")
        all_keys = train_keys + valid_keys

        # Verify states exist
        first = src[f"data/{all_keys[0]}"]
        assert "states" in first, "Source HDF5 missing 'states' — cannot extract joint positions"
        state_dim = first["states"].shape[1]

        print(f"Converting {len(all_keys)} demos to joint-space actions")
        print(f"  Task:    {task}")
        print(f"  Action:  {JOINT_ACTION_DIM}D (joint_pos7 + grip1)")
        print(f"  Proprio: {JOINT_PROPRIO_DIM}D (joint_pos7 + grip_qpos2)")
        print(f"  Input:   {input_path}")
        print(f"  Output:  {output_path}")

        with create_unified_hdf5(
            str(output_path), "robomimic", task,
            proprio_dim=JOINT_PROPRIO_DIM,
            action_dim=JOINT_ACTION_DIM,
        ) as dst:
            # Copy env_args if present
            if "data" in src and "env_args" in src["data"].attrs:
                dst["data"].attrs["env_args"] = src["data"].attrs["env_args"]

            # Mark as joint-space for downstream code
            dst.attrs["action_space"] = "joint"

            for i, demo_key in enumerate(all_keys):
                src_grp = src[f"data/{demo_key}"]
                T = src_grp["actions"].shape[0]

                dst_grp = create_demo_group(
                    dst, demo_key, T,
                    D_prop=JOINT_PROPRIO_DIM,
                    action_dim=JOINT_ACTION_DIM,
                    image_dtype=np.uint8,
                    state_dim=state_dim,
                )

                states = src_grp["states"][:].astype(np.float32)
                ee_actions = src_grp["actions"][:].astype(np.float32)

                # Joint-space actions and proprio
                dst_grp["actions"][:] = _extract_joint_actions(states, ee_actions)
                dst_grp["proprio"][:] = _extract_joint_proprio(states)

                # Copy images, view_present, states unchanged
                dst_grp["images"][:] = src_grp["images"][:]
                dst_grp["view_present"][:] = src_grp["view_present"][:]
                dst_grp["states"][:] = states

                if (i + 1) % 20 == 0 or (i + 1) == len(all_keys):
                    print(f"  [{i+1}/{len(all_keys)}] done")

            write_mask(dst, "train", train_keys)
            write_mask(dst, "valid", valid_keys)

            print("Computing normalization stats...")
            stats = compute_and_save_norm_stats(dst, train_keys)

    print("Conversion complete.")
    print(f"  Action mean: {stats['actions']['mean'].round(4)}")
    print(f"  Action std:  {stats['actions']['std'].round(4)}")
    print(f"  Action min:  {stats['actions']['min'].round(4)}")
    print(f"  Action max:  {stats['actions']['max'].round(4)}")
    print(f"  Proprio mean: {stats['proprio']['mean'].round(4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert robomimic to joint-space actions")
    parser.add_argument("--input", required=True, help="Input unified HDF5 (with states)")
    parser.add_argument("--output", required=True, help="Output joint-space HDF5")
    args = parser.parse_args()

    convert_to_joints(args.input, args.output)
