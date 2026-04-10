"""Convert RLBench PerAct demos to unified schema.

Camera mapping (slot assignment matches spec section 5):
  front_rgb          -> slot 0
  left_shoulder_rgb  -> slot 1
  right_shoulder_rgb -> slot 2
  wrist_rgb          -> slot 3
  (overhead_rgb is ignored — not in our 4-slot schema)

Actions: 8D absolute EE poses [position(3), quaternion_xyzw(4), gripper_centered(1)].
  Action at time t = target pose at time t+1 (next EE position to reach),
  with gripper state from time t (current gripper command).
  Gripper centered: {0,1} -> {-1,1} via grip*2-1 (symmetric for DDPM).
  T timesteps -> T-1 actions (last observation has no "next" target).

  Changed in v6: Previously stored 7D delta actions (position delta + rotation
  axis-angle delta + gripper). Dense deltas compound IK tracking errors over
  ~170 steps. CoA/ACT/DP all use absolute EE poses on RLBench. See spec v6
  Section 5.2 for full rationale.

Quaternion: gripper_pose stores [x, y, z, qx, qy, qz, qw]  (xyzw — scipy-native).
  Verified from RVT codebase: utils.quaternion_to_discrete_euler calls
  scipy Rotation.from_quat(obs.gripper_pose[3:]) directly with no reordering.

Proprio: eef_pos (3) + eef_quat_xyzw (4) + gripper_centered (1) = 8D.
  Matches action space (EE-centric) for direct proprio→action correspondence.
  Changed from joint_positions(7)+gripper_open(1) in v35 update.

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

import cv2
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

PROPRIO_DIM = 8   # eef_pos (3) + eef_quat_xyzw (4) + gripper_centered (1)
ACTION_DIM = 8    # position (3) + quaternion_xyzw (4) + gripper (1)


# ---------------------------------------------------------------------------
# Keyframe extraction (PerAct heuristic + commanded gripper fix)
# ---------------------------------------------------------------------------

def _get_grip_cmd(obs):
    """Get commanded gripper state from joint_position_action if available.

    obs.gripper_open is unreliable for grasping tasks — the fingers don't
    fully close around objects, so get_open_amount() reads 1.0 even when
    the gripper was commanded to close. joint_position_action[-1] is the
    actual commanded target.
    """
    jpa = obs.misc.get("joint_position_action", None)
    if jpa is not None:
        return jpa[-1]
    return float(obs.gripper_open)


def _is_stopped(obs_list, i, obs, stopped_buffer, delta=0.1):
    """Check if robot is stopped at timestep i (PerAct heuristic)."""
    next_is_not_final = i == (len(obs_list) - 2)
    grip_i = _get_grip_cmd(obs)
    grip_next = _get_grip_cmd(obs_list[i + 1]) if i < len(obs_list) - 1 else grip_i
    grip_prev = _get_grip_cmd(obs_list[i - 1]) if i > 0 else grip_i
    grip_prev2 = _get_grip_cmd(obs_list[i - 2]) if i > 1 else grip_prev
    gripper_state_no_change = (
        i < (len(obs_list) - 2) and
        (grip_i == grip_next and grip_i == grip_prev and grip_prev2 == grip_prev))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(obs_list, stopping_delta=0.1):
    """Discover keypoints using PerAct's heuristic + commanded gripper fix.

    Finds timesteps where:
    - Gripper command changes (from joint_position_action[-1])
    - Robot is stopped (joint velocities near zero)
    - Episode ends

    Returns:
        List of keypoint timestep indices.
    """
    episode_keypoints = []
    prev_grip = _get_grip_cmd(obs_list[0])
    stopped_buffer = 0
    for i, obs in enumerate(obs_list):
        stopped = _is_stopped(obs_list, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        curr_grip = _get_grip_cmd(obs)
        last = i == (len(obs_list) - 1)
        if i != 0 and (curr_grip != prev_grip or last or stopped):
            episode_keypoints.append(i)
        prev_grip = curr_grip
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    return episode_keypoints


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

    Gripper is centered: {0,1} → {-1,1} via grip*2-1 for DDPM symmetry.

    Args:
        positions:   [T, 3]  absolute xyz positions.
        quats_xyzw:  [T, 4]  absolute orientations in xyzw order (scipy-native).
        grippers:    [T]     gripper_open values (0 or 1).

    Returns:
        [T-1, 8] float32 absolute actions:
          [position(3), quaternion_xyzw(4), gripper_centered(1)]
    """
    T = positions.shape[0]
    assert T >= 2, "Need at least 2 timesteps"

    # Target pose = next timestep's pose
    target_pos = positions[1:]          # [T-1, 3]
    target_quat = quats_xyzw[1:].copy()  # [T-1, 4] xyzw order

    # Canonical quaternion: ensure w > 0 (eliminates double-cover)
    # quat is [qx, qy, qz, qw], so qw is at index 3
    neg_mask = target_quat[:, 3] < 0
    target_quat[neg_mask] = -target_quat[neg_mask]

    # Gripper command at current timestep, centered {0,1} → {-1,1}
    gripper_action = (grippers[:-1] * 2 - 1)[:, None].astype(np.float32)  # [T-1, 1]

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
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
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

    Proprio is EE-centric: [eef_pos(3), eef_quat_xyzw(4), gripper_centered(1)] = 8D.
    This matches the action space for direct proprio→action correspondence
    (same pattern as robomimic's eef_pos+eef_quat+gripper_qpos).
    Gripper centered: {0,1} → {-1,1} via grip*2-1.

    Returns:
        positions:  [T, 3]  gripper xyz
        quats_xyzw: [T, 4]  gripper quaternion in xyzw order (scipy-native)
        grippers:   [T]     gripper_open (float, raw 0/1)
        proprio:    [T, 8]  eef_pos(3) + eef_quat_xyzw(4) + gripper_centered(1)
    """
    T = len(obs_list)
    positions  = np.zeros((T, 3), dtype=np.float32)
    quats_xyzw = np.zeros((T, 4), dtype=np.float32)
    grippers   = np.zeros(T, dtype=np.float32)

    for t, obs in enumerate(obs_list):
        pose = obs.gripper_pose  # [7]: x y z qx qy qz qw
        positions[t]  = pose[:3]
        quats_xyzw[t] = pose[3:]   # already xyzw
        grippers[t]   = _get_grip_cmd(obs)

    # Canonical quaternion for proprio too: ensure w > 0
    neg_mask = quats_xyzw[:, 3] < 0
    quats_xyzw[neg_mask] = -quats_xyzw[neg_mask]

    # EE-centric proprio: [eef_pos(3), eef_quat_canonical(4), gripper_centered(1)]
    grip_centered = (grippers * 2 - 1)[:, None].astype(np.float32)  # {0,1} → {-1,1}
    proprio = np.concatenate([positions, quats_xyzw, grip_centered], axis=1)  # [T, 8]
    return positions, quats_xyzw, grippers, proprio


# ---------------------------------------------------------------------------
# Episode conversion
# ---------------------------------------------------------------------------

def convert_episode_nbp(
    ep_dir: Path,
    hdf5_file: h5py.File,
    demo_key_prefix: str,
    augment_every_n: int = 5,
) -> list[str]:
    """Convert one episode into multiple NBP (Next-Best-Pose) sub-demos.

    Matches robobase's get_nbp_demos: for each starting point (every N steps),
    create a mini-demo containing [start_obs, keypoint1, keypoint2, ...].
    Action at each step = absolute EE pose at the next step in the mini-demo
    (= the next keyframe target, a large movement).

    Returns list of demo keys created.
    """
    obs_list = load_low_dim(ep_dir)
    T = len(obs_list)
    if T < 2:
        return []

    positions, quats_xyzw, grippers, proprio = extract_proprio_and_pose(obs_list)
    imgs_all = load_images_for_episode(ep_dir, T)

    kf_indices = keypoint_discovery(obs_list)
    if len(kf_indices) < 1:
        return []

    created_keys = []
    sub_idx = 0

    for start in range(0, T, augment_every_n):
        # Find keyframes that come after this starting point
        remaining_kf = [k for k in kf_indices if k > start]
        if len(remaining_kf) == 0:
            continue

        # Build mini-demo indices: [start, kf1, kf2, ...]
        indices = [start] + remaining_kf
        K = len(indices)

        # Observations at these indices
        idx_arr = np.array(indices)
        imgs = imgs_all[idx_arr]
        proprio_out = proprio[idx_arr]

        # Actions: target pose at next index in mini-demo
        # For last step, hold current pose
        actions = np.zeros((K, ACTION_DIM), dtype=np.float32)
        for i in range(K - 1):
            next_t = indices[i + 1]
            curr_t = indices[i]
            actions[i, :3] = positions[next_t]
            actions[i, 3:7] = quats_xyzw[next_t]
            grip = _get_grip_cmd(obs_list[curr_t])
            actions[i, 7] = grip * 2 - 1
        # Last: hold pose
        last_t = indices[-1]
        actions[-1, :3] = positions[last_t]
        actions[-1, 3:7] = quats_xyzw[last_t]
        actions[-1, 7] = _get_grip_cmd(obs_list[last_t]) * 2 - 1

        demo_key = f"{demo_key_prefix}_s{sub_idx}"
        grp = create_demo_group(
            hdf5_file, demo_key, T=K, D_prop=PROPRIO_DIM,
            action_dim=ACTION_DIM, image_dtype=np.uint8,
        )
        grp["images"][:] = imgs
        grp["actions"][:] = actions
        grp["proprio"][:] = proprio_out
        grp["view_present"][:] = _VIEW_PRESENT

        created_keys.append(demo_key)
        sub_idx += 1

    return created_keys


def convert_episode(
    ep_dir: Path,
    hdf5_file: h5py.File,
    demo_key: str,
    keyframes: bool = False,
) -> bool:
    """Convert one RLBench episode. Returns False and skips if too short.

    Dense mode (keyframes=False):
        Stores T-1 timesteps: at each t we have obs[t] paired with action
        targeting pose at t+1.

    Keyframe mode (keyframes=True):
        Detects keypoints (gripper changes, velocity stops, endpoint).
        Stores only keyframe observations. Action at keyframe i = EE pose
        at keyframe i+1, with gripper command from keyframe i.
        Last keyframe's action = its own pose (hold position).
    """
    obs_list = load_low_dim(ep_dir)
    T = len(obs_list)
    if T < 2:
        print(f"  [SKIP] {demo_key}: only {T} timestep(s)")
        return False

    positions, quats_xyzw, grippers, proprio = extract_proprio_and_pose(obs_list)
    imgs_all = load_images_for_episode(ep_dir, T)

    if not keyframes:
        # Dense mode: T-1 actions
        actions = extract_absolute_actions(positions, quats_xyzw, grippers)
        imgs = imgs_all[:-1]
        proprio_out = proprio[:-1]
    else:
        # Keyframe mode: subsample to keypoints only
        kf_indices = keypoint_discovery(obs_list)
        K = len(kf_indices)
        if K < 2:
            print(f"  [SKIP] {demo_key}: only {K} keyframe(s)")
            return False

        # Observations at keyframe timesteps
        kf_idx = np.array(kf_indices)
        imgs = imgs_all[kf_idx]         # [K, K_cam, H, W, 3]
        proprio_out = proprio[kf_idx]   # [K, 8]

        # Actions: target EE pose at next keyframe, gripper from current
        # For the last keyframe, action = hold current pose
        actions = np.zeros((K, ACTION_DIM), dtype=np.float32)
        for i in range(K - 1):
            next_idx = kf_indices[i + 1]
            curr_idx = kf_indices[i]
            actions[i, :3] = positions[next_idx]
            actions[i, 3:7] = quats_xyzw[next_idx]
            # Use commanded gripper, centered {0,1} → {-1,1}
            grip = _get_grip_cmd(obs_list[curr_idx])
            actions[i, 7] = grip * 2 - 1
        # Last keyframe: hold pose, use its own gripper command
        last_idx = kf_indices[-1]
        actions[-1, :3] = positions[last_idx]
        actions[-1, 3:7] = quats_xyzw[last_idx]
        actions[-1, 7] = _get_grip_cmd(obs_list[last_idx]) * 2 - 1

        print(f"    {demo_key}: {T} steps -> {K} keyframes (indices: {kf_indices})")

    T_out = actions.shape[0]
    grp = create_demo_group(
        hdf5_file, demo_key, T=T_out, D_prop=PROPRIO_DIM,
        action_dim=ACTION_DIM, image_dtype=np.uint8,
    )
    grp["images"][:] = imgs
    grp["actions"][:] = actions
    grp["proprio"][:] = proprio_out
    grp["view_present"][:] = _VIEW_PRESENT

    return True


# ---------------------------------------------------------------------------
# Task conversion
# ---------------------------------------------------------------------------

def _collect_episodes(task_root: Path) -> list[Path]:
    """Find all episode directories under task_root/{all_variations,variation0}/episodes/."""
    ep_parent = task_root / "all_variations" / "episodes"
    if not ep_parent.exists():
        # stepjam dataset_generator uses variation0/ instead of all_variations/
        ep_parent = task_root / "variation0" / "episodes"
    if not ep_parent.exists():
        raise FileNotFoundError(f"Expected episodes at {task_root}/{{all_variations,variation0}}/episodes")
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
    keyframes: bool = False,
    nbp: bool = False,
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

        if nbp:
            print(f"Mode: NBP (Next-Best-Pose sub-demos, augment every 5 steps)")
        elif keyframes:
            print(f"Mode: KEYFRAME (extracting keypoints only)")

        for ep_dir in train_ep_dirs:
            if nbp:
                prefix = f"demo_{demo_idx}"
                print(f"  [train] {prefix}_s* <- {ep_dir.name}")
                keys = convert_episode_nbp(ep_dir, f, prefix)
                train_keys.extend(keys)
                print(f"    -> {len(keys)} sub-demos")
            else:
                demo_key = f"demo_{demo_idx}"
                print(f"  [train] {demo_key} <- {ep_dir.name}")
                ok = convert_episode(ep_dir, f, demo_key, keyframes=keyframes)
                if ok:
                    train_keys.append(demo_key)
            demo_idx += 1

        for ep_dir in val_ep_dirs:
            if nbp:
                prefix = f"demo_{demo_idx}"
                print(f"  [valid] {prefix}_s* <- {ep_dir.name}")
                keys = convert_episode_nbp(ep_dir, f, prefix)
                valid_keys.extend(keys)
                print(f"    -> {len(keys)} sub-demos")
            else:
                demo_key = f"demo_{demo_idx}"
                print(f"  [valid] {demo_key} <- {ep_dir.name}")
                ok = convert_episode(ep_dir, f, demo_key, keyframes=keyframes)
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
        choices=["close_jar", "open_drawer", "slide_block_to_color_target",
                 "put_item_in_drawer", "stack_cups", "place_shape_in_shape_sorter",
                 "meat_off_grill", "turn_tap", "push_buttons", "reach_and_drag",
                 "place_wine_at_rack_location", "sweep_to_dustpan_of_size",
                 "sweep_to_dustpan", "reach_target"],
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
    parser.add_argument(
        "--keyframes", action="store_true",
        help="Extract keyframes only (PerAct heuristic: gripper changes + "
             "velocity stops). Reduces ~170 steps to ~5-7 keyframes per demo.",
    )
    parser.add_argument(
        "--nbp", action="store_true",
        help="NBP (Next-Best-Pose) mode: create multiple sub-demos per episode, "
             "each starting at a different point with keyframe targets. "
             "Matches robobase's get_nbp_demos for END_EFFECTOR_POSE training.",
    )
    args = parser.parse_args()
    convert_task(args.input, args.output, train_frac=args.train_frac,
                 val_dir=args.val_input, keyframes=args.keyframes,
                 nbp=args.nbp)


if __name__ == "__main__":
    main()
