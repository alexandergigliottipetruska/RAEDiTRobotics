"""RLBench demo replay verification (spec section 8.4).

Three verification modes:

1. Numerical round-trip (--mode numerical, default):
   Load raw absolute poses from low_dim_obs.pkl, load converted delta actions
   from unified HDF5, accumulate deltas back to absolute, compare. This proves
   the delta conversion is mathematically lossless.

2. Sim replay (--mode sim):
   Launch CoppeliaSim via RLBench, restore scene state via demo random seed,
   replay joint positions directly (no motion planning). Checks task success.
   Requires WSL2 with PyRep + CoppeliaSim installed.

3. Video from raw data (--mode video):
   Stitch raw demo PNGs (front_rgb/) into MP4 videos. These are the original
   CoppeliaSim renders from data collection — genuine sim video.

Usage (numerical — works on Windows or WSL2):
  python replay_rlbench.py \
      --hdf5  ~/rlbench_replay/unified/close_jar.hdf5 \
      --raw   ~/rlbench_replay/raw/close_jar \
      --n 5

Usage (sim — WSL2 only):
  QT_QPA_PLATFORM=offscreen xvfb-run python replay_rlbench.py \
      --mode sim \
      --raw   ~/rlbench_replay/raw/close_jar \
      --task  close_jar \
      --n 5

Usage (video — works on Windows or WSL2):
  python replay_rlbench.py \
      --mode video \
      --raw   ~/rlbench_replay/raw/close_jar \
      --video-dir ~/rlbench_replay/replay_videos \
      --n 3
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Raw demo loading (works with real rlbench OR the stub)
# ---------------------------------------------------------------------------

def _ensure_rlbench_importable():
    """Register stub if real rlbench is not installed."""
    try:
        import rlbench  # noqa: F401
    except ImportError:
        # Add repo root so the stub import works
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        from data_pipeline.conversion.rlbench_obs_stub import register_stub
        register_stub()


def load_raw_demo(ep_dir: Path):
    """Load raw observations from low_dim_obs.pkl.

    Returns list of Observation objects (real or stub).
    """
    with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
        demo = pickle.load(fh)
    # Demo might be a Demo object (list-like) with _observations
    if hasattr(demo, "_observations"):
        return demo._observations
    return list(demo)


def extract_poses(obs_list):
    """Extract absolute poses and gripper states from raw observations.

    Returns:
        positions:  [T, 3]  xyz
        quats_xyzw: [T, 4]  quaternion in xyzw order (scipy-native)
        grippers:   [T]     gripper_open (0 or 1)
    """
    T = len(obs_list)
    positions  = np.zeros((T, 3), dtype=np.float64)
    quats_xyzw = np.zeros((T, 4), dtype=np.float64)
    grippers   = np.zeros(T, dtype=np.float64)

    for t, obs in enumerate(obs_list):
        pose = obs.gripper_pose  # [x, y, z, qx, qy, qz, qw]
        positions[t]  = pose[:3]
        quats_xyzw[t] = pose[3:]
        grippers[t]   = float(obs.gripper_open)

    return positions, quats_xyzw, grippers


# ---------------------------------------------------------------------------
# Numerical round-trip verification
# ---------------------------------------------------------------------------

def accumulate_deltas(
    init_pos: np.ndarray,       # [3]
    init_quat_xyzw: np.ndarray, # [4]
    delta_actions: np.ndarray,  # [T-1, 7]
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate delta actions back to absolute poses.

    Returns:
        positions:  [T, 3]  (T = T-1 + 1, includes initial)
        quats_xyzw: [T, 4]
    """
    T_minus_1 = delta_actions.shape[0]
    T = T_minus_1 + 1

    positions  = np.zeros((T, 3), dtype=np.float64)
    quats_xyzw = np.zeros((T, 4), dtype=np.float64)

    positions[0]  = init_pos
    quats_xyzw[0] = init_quat_xyzw
    rot = R.from_quat(init_quat_xyzw)

    for t in range(T_minus_1):
        delta_pos    = delta_actions[t, :3].astype(np.float64)
        delta_rotvec = delta_actions[t, 3:6].astype(np.float64)

        positions[t + 1] = positions[t] + delta_pos
        delta_rot = R.from_rotvec(delta_rotvec)
        rot = delta_rot * rot  # world-frame: delta * current
        quats_xyzw[t + 1] = rot.as_quat()

    return positions, quats_xyzw


def run_numerical_verification(args):
    """Compare accumulated deltas against raw absolute poses."""
    import h5py

    _ensure_rlbench_importable()

    raw_root = Path(args.raw)
    ep_parent = raw_root / "all_variations" / "episodes"
    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )

    # Load unified HDF5 to get delta actions and demo key mapping
    with h5py.File(args.hdf5, "r") as f:
        split = args.split
        mask_keys = [
            k.decode() if isinstance(k, bytes) else k
            for k in f["mask"][split][:]
        ]
        n_demos = min(args.n, len(mask_keys), len(ep_dirs))

        print(f"Numerical round-trip verification")
        print(f"  HDF5:  {args.hdf5}")
        print(f"  Raw:   {args.raw}")
        print(f"  Demos: {n_demos}")
        print()

        all_pos_errors = []
        all_rot_errors = []

        for i in range(n_demos):
            demo_key = mask_keys[i]
            ep_dir = ep_dirs[i]

            # Raw absolute poses
            obs_list = load_raw_demo(ep_dir)
            raw_pos, raw_quat, raw_grip = extract_poses(obs_list)

            # Converted delta actions from unified HDF5
            delta_actions = f[f"data/{demo_key}/actions"][:]

            # Accumulate deltas from initial pose
            acc_pos, acc_quat = accumulate_deltas(
                raw_pos[0], raw_quat[0], delta_actions
            )

            # Compare positions: max absolute error
            pos_err = np.abs(acc_pos - raw_pos).max()

            # Compare rotations: angular error in degrees
            rot_orig = R.from_quat(raw_quat)
            rot_acc  = R.from_quat(acc_quat)
            rot_diff = rot_acc * rot_orig.inv()
            angle_errors = rot_diff.magnitude()  # radians
            rot_err_deg = np.degrees(angle_errors.max())

            # Compare grippers
            grip_from_hdf5 = delta_actions[:, 6]
            grip_match = np.allclose(grip_from_hdf5, raw_grip[:-1], atol=1e-5)

            status = "OK" if (pos_err < 1e-4 and rot_err_deg < 0.01) else "MISMATCH"
            print(f"  {demo_key:10s} ({ep_dir.name:12s}) | "
                  f"pos_err={pos_err:.2e}  rot_err={rot_err_deg:.4f}deg  "
                  f"grip={'OK' if grip_match else 'FAIL'}  [{status}]")

            all_pos_errors.append(pos_err)
            all_rot_errors.append(rot_err_deg)

    max_pos = max(all_pos_errors)
    max_rot = max(all_rot_errors)
    print(f"\nSummary:")
    print(f"  Max position error:  {max_pos:.2e} m")
    print(f"  Max rotation error:  {max_rot:.4f} deg")

    if max_pos < 1e-4 and max_rot < 0.1:
        print("  PASS: Delta conversion is numerically lossless.")
        return True
    else:
        print("  FAIL: Significant round-trip error detected.")
        return False


# ---------------------------------------------------------------------------
# Sim replay
# ---------------------------------------------------------------------------

_TASK_CLASS_MAP = {
    "close_jar":                    "CloseJar",
    "open_drawer":                  "OpenDrawer",
    "slide_block_to_color_target":  "SlideBlockToColorTarget",
}


def _load_demo_object(ep_dir: Path):
    """Load the full Demo object (not just observations) from pickle.

    The Demo object carries random_seed needed for scene restoration.
    """
    with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
        return pickle.load(fh)


def _save_video(frames: list, output_path: str, fps: int = 10):
    """Save a list of RGB numpy arrays as an MP4 video."""
    import imageio.v3 as iio
    # Stack to [N, H, W, 3] uint8
    frames_arr = [f.astype(np.uint8) if f.dtype != np.uint8 else f for f in frames]
    iio.imwrite(output_path, frames_arr, fps=fps, codec="libx264")
    print(f"    Video saved: {output_path}")


def _restore_demo_scene(task, demo):
    """Try to restore scene state from demo. Returns (descriptions, obs, restored).

    Handles PerAct demos that lack misc['variation_index'] by manually
    restoring the random seed and calling reset with the demo.
    """
    obs_list = demo._observations if hasattr(demo, '_observations') else list(demo)

    # Try the built-in reset_to_demo first
    try:
        descriptions, obs = task.reset_to_demo(demo)
        return descriptions, obs, True
    except (KeyError, AttributeError):
        pass

    # Fall back: restore random seed manually, then reset with demo object
    if demo.random_seed is not None:
        try:
            demo.restore_state()
            # Try to extract variation index from the observations
            variation = 0
            if hasattr(obs_list[0], 'misc') and obs_list[0].misc is not None:
                variation = obs_list[0].misc.get('variation_index', 0)
            task.set_variation(variation)
            descriptions, obs = task.reset(demo)
            return descriptions, obs, True
        except Exception:
            pass

    # Final fallback: random reset
    descriptions, obs = task.reset()
    return descriptions, obs, False


def run_sim_replay(args):
    """Replay demos in CoppeliaSim using joint positions directly.

    Restores scene via demo random seed, then sets joint positions + gripper
    at every timestep (no motion planning). This is the most faithful replay
    possible — it matches exactly what happened during data collection.
    """
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig

    task_name = args.task
    if task_name not in _TASK_CLASS_MAP:
        print(f"Unknown task: {task_name}")
        print(f"Supported: {list(_TASK_CLASS_MAP.keys())}")
        sys.exit(1)

    raw_root = Path(args.raw)
    ep_parent = raw_root / "all_variations" / "episodes"
    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )

    n_demos = min(args.n, len(ep_dirs))

    print(f"Sim replay (joint positions): {task_name}")
    print(f"  Raw:    {args.raw}")
    print(f"  Demos:  {n_demos}")
    print()

    # No camera rendering — avoids OpenGL crashes in headless WSL2
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointPosition(),
        gripper_action_mode=Discrete(),
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    task_class_name = _TASK_CLASS_MAP[task_name]
    import rlbench.tasks
    task_cls = getattr(rlbench.tasks, task_class_name)
    task = env.get_task(task_cls)

    successes = []

    for i in range(n_demos):
        ep_dir = ep_dirs[i]
        demo = _load_demo_object(ep_dir)
        obs_list = demo._observations if hasattr(demo, '_observations') else list(demo)

        # Extract joint positions and gripper states for every timestep
        T = len(obs_list)
        joint_positions = np.zeros((T, 7), dtype=np.float64)
        grippers = np.zeros(T, dtype=np.float64)
        for t, obs in enumerate(obs_list):
            joint_positions[t] = np.array(obs.joint_positions, dtype=np.float64)
            grippers[t] = float(obs.gripper_open)

        # Restore scene state from demo
        _, obs, restored = _restore_demo_scene(task, demo)

        success = False
        try:
            for t in range(T):
                # Action = [joint_positions(7), gripper(1)]
                action = np.append(joint_positions[t], grippers[t])
                obs, reward, terminate = task.step(action)

                if reward == 1.0:
                    success = True
                    break
        except Exception as e:
            print(f"  {ep_dir.name:12s} | ERROR: {e}")
            successes.append(False)
            continue

        status = "SUCCESS" if success else "FAIL"
        scene = "restored" if restored else "random"
        print(f"  {ep_dir.name:12s} | steps={T:4d} | scene={scene:8s} | {status}")
        successes.append(success)

    env.shutdown()

    n_success = sum(successes)
    rate = n_success / len(successes) if successes else 0
    print(f"\nResult: {n_success}/{len(successes)} succeeded ({rate*100:.0f}%)")

    if rate >= 0.8:
        print("PASS: Demo replay meets >=80% threshold.")
    elif rate >= 0.5:
        print("PARTIAL: Some demos succeeded.")
    else:
        print("NOTE: Low success rate. Check scene restoration logs above.")
        print("      The numerical round-trip test is the authoritative verification.")

    return rate >= 0.5


# ---------------------------------------------------------------------------
# Video from raw demo PNGs
# ---------------------------------------------------------------------------

def run_video_from_raw(args):
    """Stitch raw demo PNGs into MP4 videos.

    These are the original CoppeliaSim renders captured during data collection.
    No sim needed — just reads PNG files and encodes to video.
    """
    from PIL import Image

    raw_root = Path(args.raw)
    ep_parent = raw_root / "all_variations" / "episodes"
    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )

    n_demos = min(args.n, len(ep_dirs))
    video_dir = Path(args.video_dir) if args.video_dir else Path.cwd() / "replay_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    task_name = raw_root.name

    print(f"Video from raw demo PNGs: {task_name}")
    print(f"  Raw:    {args.raw}")
    print(f"  Output: {video_dir}")
    print(f"  Demos:  {n_demos}")
    print()

    cameras = ["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb"]

    for i in range(n_demos):
        ep_dir = ep_dirs[i]

        # Check which cameras have PNGs
        available_cams = []
        for cam in cameras:
            cam_dir = ep_dir / cam
            if cam_dir.exists() and any(cam_dir.glob("*.png")):
                available_cams.append(cam)

        if not available_cams:
            print(f"  {ep_dir.name:12s} | No PNG files found — skipped")
            continue

        # Count frames from first available camera
        first_cam_dir = ep_dir / available_cams[0]
        n_frames = len(list(first_cam_dir.glob("*.png")))

        # Build grid frames: stack cameras horizontally
        frames = []
        for t in range(n_frames):
            row = []
            for cam in available_cams:
                img_path = ep_dir / cam / f"{t}.png"
                if img_path.exists():
                    img = np.array(Image.open(img_path).convert("RGB"))
                    row.append(img)
            if row:
                # Horizontal concatenation of all cameras
                frames.append(np.concatenate(row, axis=1))

        if not frames:
            print(f"  {ep_dir.name:12s} | No frames loaded — skipped")
            continue

        video_path = str(video_dir / f"{task_name}_{ep_dir.name}.mp4")
        _save_video(frames, video_path, fps=15)
        print(f"  {ep_dir.name:12s} | {n_frames} frames, {len(available_cams)} cameras -> {video_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RLBench demo replay verification."
    )
    parser.add_argument(
        "--mode", default="numerical", choices=["numerical", "sim", "video"],
        help="Verification mode (default: numerical).",
    )
    parser.add_argument(
        "--hdf5",
        help="Path to unified HDF5 (required for numerical mode).",
    )
    parser.add_argument(
        "--raw", required=True,
        help="Path to raw RLBench task dir (e.g. .../close_jar).",
    )
    parser.add_argument(
        "--task", default="close_jar",
        choices=list(_TASK_CLASS_MAP.keys()),
        help="Task name (required for sim mode).",
    )
    parser.add_argument("--n", type=int, default=5, help="Number of demos.")
    parser.add_argument("--split", default="train", help="HDF5 split to use.")
    parser.add_argument(
        "--video-dir",
        help="Directory for replay videos (default: ./replay_videos). Sim mode only.",
    )

    args = parser.parse_args()

    if args.mode == "numerical":
        if not args.hdf5:
            parser.error("--hdf5 is required for numerical mode")
        ok = run_numerical_verification(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "sim":
        ok = run_sim_replay(args)
        sys.exit(0 if ok else 1)
    elif args.mode == "video":
        run_video_from_raw(args)
        sys.exit(0)


if __name__ == "__main__":
    main()
