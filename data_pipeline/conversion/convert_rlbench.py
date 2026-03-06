"""Convert RLBench PerAct demos to unified schema.

Camera mapping (slot assignment matches spec section 4):
  front_rgb          -> slot 0
  left_shoulder_rgb  -> slot 1
  right_shoulder_rgb -> slot 2
  wrist_rgb          -> slot 3
  (overhead_rgb is ignored — not in our 4-slot schema)

Actions: absolute EE pose -> 7D delta [dx, dy, dz, drx, dry, drz, gripper].
  delta_pos (3): pos[t+1] - pos[t]
  delta_rot (3): axis-angle of R[t+1] * R[t].inv()  (world frame)
  gripper   (1): gripper_open at timestep t (0=closed, 1=open)

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

Usage:
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
from scipy.spatial.transform import Rotation as R

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
# Proprio dimensionality: joint_positions (7) + gripper_open (1)
# ---------------------------------------------------------------------------

PROPRIO_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_delta_actions(
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    grippers: np.ndarray,
) -> np.ndarray:
    """Convert absolute EE poses to delta actions.

    Args:
        positions:   [T, 3]  absolute xyz positions.
        quats_xyzw:  [T, 4]  absolute orientations in xyzw order (scipy-native).
                             Taken directly from obs.gripper_pose[3:].
        grippers:    [T]     gripper_open values (0 or 1).

    Returns:
        [T-1, 7] float32 delta actions:
          [delta_pos(3), delta_rotvec(3), gripper(1)]
        gripper at step t uses the value AT timestep t (current state).
    """
    T = positions.shape[0]
    assert T >= 2, "Need at least 2 timesteps to compute deltas"

    rots = R.from_quat(quats_xyzw)                       # [T] Rotation objects

    delta_pos = np.diff(positions, axis=0)                # [T-1, 3]
    delta_rot = (rots[1:] * rots[:-1].inv()).as_rotvec()  # [T-1, 3] world frame

    # Use gripper at current timestep t (not t+1) as the action signal.
    gripper_action = grippers[:-1, None].astype(np.float32)  # [T-1, 1]

    return np.concatenate(
        [delta_pos.astype(np.float32),
         delta_rot.astype(np.float32),
         gripper_action],
        axis=1,
    )  # [T-1, 7]


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
    """Load low_dim_obs.pkl. Returns list of Observation objects.

    Requires `pip install rlbench` so pickle can reconstruct the class.
    """
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

    Stores T-1 timesteps: at each t we have obs[t] paired with delta to t+1.
    """
    obs_list = load_low_dim(ep_dir)
    T = len(obs_list)
    if T < 2:
        print(f"  [SKIP] {demo_key}: only {T} timestep(s)")
        return False

    positions, quats_xyzw, grippers, proprio = extract_proprio_and_pose(obs_list)

    # Delta actions: [T-1, 7]
    actions = compute_delta_actions(positions, quats_xyzw, grippers)

    # Images: [T, K, H, W, 3] uint8 -> keep only first T-1 (paired with actions)
    imgs_all = load_images_for_episode(ep_dir, T)
    imgs            = imgs_all[:-1]    # [T-1, K, H, W, 3]
    proprio_trimmed = proprio[:-1]     # [T-1, 8]

    T_out = T - 1
    grp = create_demo_group(
        hdf5_file, demo_key, T=T_out, D_prop=PROPRIO_DIM, image_dtype=np.uint8
    )
    grp["images"][:] = imgs
    grp["actions"][:] = actions
    grp["proprio"][:] = proprio_trimmed
    grp["view_present"][:] = _VIEW_PRESENT

    return True


# ---------------------------------------------------------------------------
# Task conversion
# ---------------------------------------------------------------------------

def convert_task(task_dir: str, output_path: str, train_frac: float = 0.9) -> None:
    """Convert all episodes for one task.

    Iterates over <task>/all_variations/episodes/episode<N> directories.
    Splits demos into train/valid by train_frac (ordered, not shuffled).

    Args:
        task_dir:    Path to e.g. data/raw/rlbench/data/train/close_jar/
        output_path: Path for output unified HDF5.
        train_frac:  Fraction of episodes for training (default 0.9).
    """
    task_root   = Path(task_dir)
    task_name   = task_root.name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Episodes are flat under all_variations/episodes/
    ep_parent = task_root / "all_variations" / "episodes"
    if not ep_parent.exists():
        raise FileNotFoundError(f"Expected episodes at {ep_parent}")

    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )

    n_total = len(ep_dirs)
    n_train = max(1, int(n_total * train_frac))
    print(f"Task: {task_name}")
    print(f"Total episodes found: {n_total}")
    print(f"Train: {n_train} | Valid: {n_total - n_train}")

    with create_unified_hdf5(
        str(output_path),
        benchmark="rlbench",
        task=task_name,
        proprio_dim=PROPRIO_DIM,
    ) as f:
        train_keys = []
        valid_keys = []

        for i, ep_dir in enumerate(ep_dirs):
            demo_key = f"demo_{i}"
            split    = "train" if i < n_train else "valid"
            print(f"  [{split:5s}] {demo_key} <- {ep_dir.name}")
            ok = convert_episode(ep_dir, f, demo_key)
            if not ok:
                continue
            if split == "train":
                train_keys.append(demo_key)
            else:
                valid_keys.append(demo_key)

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
        choices=["close_jar", "open_drawer", "slide_block_to_color_target"],
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to task dir (e.g. data/raw/rlbench/data/train/close_jar).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output unified HDF5 path (e.g. data/unified/rlbench/close_jar.hdf5).",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.9,
        help="Fraction of episodes for training split (default: 0.9).",
    )
    args = parser.parse_args()
    convert_task(args.input, args.output, train_frac=args.train_frac)


if __name__ == "__main__":
    main()
