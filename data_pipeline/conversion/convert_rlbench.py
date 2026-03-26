"""Convert RLBench PerAct demos to unified schema.

Camera mapping (slot assignment matches spec section 5):
  front_rgb          -> slot 0
  left_shoulder_rgb  -> slot 1
  right_shoulder_rgb -> slot 2
  wrist_rgb          -> slot 3
  (overhead_rgb is ignored — not in our 4-slot schema)

Actions: 8D absolute EE poses [position(3), quaternion_xyzw(4), gripper(1)].
  Action at time t = target pose at time t+1 (next EE position to reach),
  with gripper state from time t (current gripper command).
  T timesteps -> T-1 actions (last observation has no "next" target).

  Changed in v6: Previously stored 7D delta actions (position delta + rotation
  axis-angle delta + gripper). Dense deltas compound IK tracking errors over
  ~170 steps. CoA/ACT/DP all use absolute EE poses on RLBench. See spec v6
  Section 5.2 for full rationale.

Quaternion: gripper_pose stores [x, y, z, qx, qy, qz, qw]  (xyzw — scipy-native).
  Verified from RVT codebase: utils.quaternion_to_discrete_euler calls
  scipy Rotation.from_quat(obs.gripper_pose[3:]) directly with no reordering.

Proprio: joint_positions (7) + gripper_open (1) = 8D.

Images stored as uint8 [0, 255] to avoid ~3x disk inflation vs float32.
  Resized to 224x224 (spec IMAGE_SIZE) using PIL LANCZOS.

Episode layout (HQfang/rlbench-18-tasks):
  <task>/all_variations/episodes/episode<N>/
    low_dim_obs.pkl            list[Observation], length T  (requires rlbench pkg)
    front_rgb/0.png ... T-1.png
    left_shoulder_rgb/0.png ...
    right_shoulder_rgb/0.png ...
    wrist_rgb/0.png ...

Prerequisite:
  pip install rlbench  (needed to unpickle Observation objects)

Usage (official PerAct splits — recommended):
  python data_pipeline/conversion/convert_rlbench.py \\
      --task close_jar \\
      --input     data/raw/rlbench/data/train/close_jar \\
      --val-input data/raw/rlbench/data/valid/close_jar \\
      --output    data/unified/rlbench/close_jar.hdf5

Usage (legacy auto-split):
  python data_pipeline/conversion/convert_rlbench.py \\
      --task close_jar \\
      --input  data/raw/rlbench/data/train/close_jar \\
      --output data/unified/rlbench/close_jar.hdf5

Run from repo root with .venv active.
"""

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Register rlbench stub BEFORE any pickle.load — allows unpickling Observation
# objects without installing the full rlbench+PyRep+CoppeliaSim stack.
from data_pipeline.conversion.rlbench_obs_stub import register_stub
register_stub()

from data_pipeline.conversion.unified_schema import (
    NUM_CAMERA_SLOTS,
    IMAGE_SIZE,
    create_unified_hdf5,
    create_demo_group,
    write_mask,
)
from data_pipeline.conversion.compute_norm_stats import compute_and_save_norm_stats


# ---------------------------------------------------------------------------
# Camera slot assignment
# ---------------------------------------------------------------------------

_CAMERAS = ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]
_VIEW_PRESENT = np.array([True, True, True, True], dtype=bool)  # all 4 real cameras

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROPRIO_DIM = 8   # joint_positions (7) + gripper_open (1)
ACTION_DIM = 8    # position (3) + quaternion_xyzw (4) + gripper (1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_absolute_actions(
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    grippers: np.ndarray,
) -> np.ndarray:
    """Extract absolute EE pose actions from demo observations.

    Action at time t = target pose at time t+1, with gripper from time t.
    This follows the convention that the action is the target the robot
    should reach from the current state.

    Args:
        positions:   [T, 3]  absolute xyz positions.
        quats_xyzw:  [T, 4]  absolute orientations in xyzw order (scipy-native).
        grippers:    [T]     gripper_open values (0 or 1).

    Returns:
        [T-1, 8] float32 absolute actions:
          [position(3), quaternion_xyzw(4), gripper(1)]
    """
    T = positions.shape[0]
    assert T >= 2, "Need at least 2 timesteps"

    # Target pose = next timestep's pose
    target_pos = positions[1:]          # [T-1, 3]
    target_quat = quats_xyzw[1:]       # [T-1, 4]

    # Gripper command at current timestep
    gripper_action = grippers[:-1, None].astype(np.float32)  # [T-1, 1]

    return np.concatenate(
        [target_pos.astype(np.float32),
         target_quat.astype(np.float32),
         gripper_action],
        axis=1,
    )  # [T-1, 8]


def load_images_for_episode(ep_dir: Path, T: int) -> np.ndarray:
    """Load and resize all camera images for one episode.

    Images are stored as <camera_name>/<timestep>.png (0-indexed).

    Returns:
        uint8 array [T, K, H, W, 3] where H=W=224.
    """
    H, W = IMAGE_SIZE
    K = NUM_CAMERA_SLOTS
    out = np.zeros((T, K, H, W, 3), dtype=np.uint8)

    for slot, cam in enumerate(_CAMERAS):
        cam_dir = ep_dir / cam
        if not cam_dir.exists():
            continue  # slot stays zero (unexpected — dataset should have all 4)
        for t in range(T):
            img_path = cam_dir / f"{t}.png"
            if not img_path.exists():
                break
            frame = np.array(Image.open(img_path).convert("RGB"))
            if frame.shape[:2] != (H, W):
                frame = np.array(
                    Image.fromarray(frame).resize((W, H), Image.LANCZOS)
                )
            out[t, slot] = frame

    return out  # [T, K, H, W, 3] uint8


def load_low_dim(ep_dir: Path):
    """Load low_dim_obs.pkl. Returns list of Observation objects."""
    with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
        return pickle.load(fh)


def extract_proprio_and_pose(
    obs_list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract arrays from a list of RLBench Observation objects.

    gripper_pose layout: [x, y, z, qx, qy, qz, qw]  (xyzw — no reordering needed)

    Returns:
        positions:  [T, 3]  gripper xyz
        quats_xyzw: [T, 4]  gripper quaternion in xyzw order (scipy-native)
        grippers:   [T]     gripper_open (float)
        proprio:    [T, 8]  joint_positions(7) + gripper_open(1)
    """
    T = len(obs_list)
    positions  = np.zeros((T, 3), dtype=np.float32)
    quats_xyzw = np.zeros((T, 4), dtype=np.float32)
    grippers   = np.zeros(T, dtype=np.float32)
    joint_pos  = np.zeros((T, 7), dtype=np.float32)

    for t, obs in enumerate(obs_list):
        pose = obs.gripper_pose  # [7]: x y z qx qy qz qw
        positions[t]  = pose[:3]
        quats_xyzw[t] = pose[3:]   # already xyzw
        grippers[t]   = float(obs.gripper_open)
        joint_pos[t]  = np.array(obs.joint_positions, dtype=np.float32)

    proprio = np.concatenate([joint_pos, grippers[:, None]], axis=1)  # [T, 8]
    return positions, quats_xyzw, grippers, proprio


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------

def convert_episode(
    ep_dir: Path,
    hdf5_file: h5py.File,
    demo_key: str,
) -> bool:
    """Convert one RLBench episode. Returns False and skips if too short.

    Stores T-1 timesteps: at each t we have obs[t] paired with action
    targeting pose at t+1.
    """
    obs_list = load_low_dim(ep_dir)
    T = len(obs_list)
    if T < 2:
        print(f"  [SKIP] {demo_key}: only {T} timestep(s)")
        return False

    positions, quats_xyzw, grippers, proprio = extract_proprio_and_pose(obs_list)

    # Absolute actions: [T-1, 8]
    actions = extract_absolute_actions(positions, quats_xyzw, grippers)

    # Images: [T, K, H, W, 3] uint8 -> keep only first T-1 (paired with actions)
    imgs_all = load_images_for_episode(ep_dir, T)
    imgs            = imgs_all[:-1]    # [T-1, K, H, W, 3]
    proprio_trimmed = proprio[:-1]     # [T-1, 8]

    T_out = T - 1
    grp = create_demo_group(
        hdf5_file, demo_key, T=T_out, D_prop=PROPRIO_DIM,
        action_dim=ACTION_DIM, image_dtype=np.uint8,
    )
    grp["images"][:] = imgs
    grp["actions"][:] = actions
    grp["proprio"][:] = proprio_trimmed
    grp["view_present"][:] = _VIEW_PRESENT

    return True


# ---------------------------------------------------------------------------
# Task conversion
# ---------------------------------------------------------------------------

def _collect_episodes(task_root: Path) -> list[Path]:
    """Find all episode directories under task_root/all_variations/episodes/."""
    ep_parent = task_root / "all_variations" / "episodes"
    if not ep_parent.exists():
        raise FileNotFoundError(f"Expected episodes at {ep_parent}")
    return sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )


def convert_task(
    task_dir: str,
    output_path: str,
    train_frac: float = 0.9,
    val_dir: str | None = None,
) -> None:
    """Convert all episodes for one task.

    If val_dir is provided, train episodes come from task_dir and valid
    episodes come from val_dir (official PerAct splits). train_frac is
    ignored in this case.

    If val_dir is None, all episodes are loaded from task_dir and split
    by train_frac (legacy behavior).

    Args:
        task_dir:    Path to e.g. data/raw/rlbench/data/train/close_jar/
        output_path: Path for output unified HDF5.
        train_frac:  Fraction for training (ignored when val_dir is set).
        val_dir:     Path to e.g. data/raw/rlbench/data/valid/close_jar/
    """
    task_root   = Path(task_dir)
    task_name   = task_root.name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if val_dir is not None:
        # Official split mode: separate train and valid folders
        val_root = Path(val_dir)
        train_ep_dirs = _collect_episodes(task_root)
        val_ep_dirs   = _collect_episodes(val_root)
        print(f"Task: {task_name} (official split)")
        print(f"Train episodes: {len(train_ep_dirs)} (from {task_root})")
        print(f"Valid episodes: {len(val_ep_dirs)} (from {val_root})")
    else:
        # Legacy mode: single folder, split by fraction
        all_ep_dirs = _collect_episodes(task_root)
        n_total = len(all_ep_dirs)
        n_train = max(1, int(n_total * train_frac))
        train_ep_dirs = all_ep_dirs[:n_train]
        val_ep_dirs   = all_ep_dirs[n_train:]
        print(f"Task: {task_name} (auto-split {train_frac:.0%})")
        print(f"Total episodes: {n_total}")
        print(f"Train: {len(train_ep_dirs)} | Valid: {len(val_ep_dirs)}")

    with create_unified_hdf5(
        str(output_path),
        benchmark="rlbench",
        task=task_name,
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
    ) as f:
        train_keys = []
        valid_keys = []
        demo_idx = 0

        for ep_dir in train_ep_dirs:
            demo_key = f"demo_{demo_idx}"
            print(f"  [train] {demo_key} <- {ep_dir.name}")
            ok = convert_episode(ep_dir, f, demo_key)
            if ok:
                train_keys.append(demo_key)
            demo_idx += 1

        for ep_dir in val_ep_dirs:
            demo_key = f"demo_{demo_idx}"
            print(f"  [valid] {demo_key} <- {ep_dir.name}")
            ok = convert_episode(ep_dir, f, demo_key)
            if ok:
                valid_keys.append(demo_key)
            demo_idx += 1

        write_mask(f, "train", train_keys)
        write_mask(f, "valid", valid_keys)

        print("\nComputing norm stats from training demos...")
        stats = compute_and_save_norm_stats(f, train_keys)
        print(f"  action mean: {stats['actions']['mean']}")
        print(f"  action std:  {stats['actions']['std']}")

    print(f"\nSaved: {output_path}")
    print(f"Train demos: {len(train_keys)} | Valid demos: {len(valid_keys)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert one RLBench task to unified HDF5."
    )
    parser.add_argument(
        "--task", required=True,
        choices=[
            # Proposal tasks
            "reach_target", "push_button", "pick_and_lift",
            "slide_block_to_target", "put_item_in_drawer",
            # Additional PerAct tasks
            "close_jar", "open_drawer", "slide_block_to_color_target",
            "stack_cups", "place_shape_in_shape_sorter",
            "meat_off_grill", "turn_tap", "push_buttons", "reach_and_drag",
            "place_wine_at_rack_location", "sweep_to_dustpan_of_size",
        ],
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to train task dir (e.g. data/raw/rlbench/data/train/close_jar).",
    )
    parser.add_argument(
        "--val-input", default=None,
        help="Path to valid task dir (e.g. data/raw/rlbench/data/valid/close_jar). "
             "When provided, uses official PerAct splits instead of --train-frac.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output unified HDF5 path (e.g. data/unified/rlbench/close_jar.hdf5).",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.9,
        help="Fraction of episodes for training split (default: 0.9). "
             "Ignored when --val-input is provided.",
    )
    args = parser.parse_args()
    convert_task(args.input, args.output, train_frac=args.train_frac,
                 val_dir=args.val_input)


if __name__ == "__main__":
    main()
