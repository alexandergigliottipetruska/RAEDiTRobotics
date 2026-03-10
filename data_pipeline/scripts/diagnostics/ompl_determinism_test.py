"""OMPL determinism test: record-then-replay free-space motion.

Phase 1 (Record): Move the arm along a straight line in free space
  (no object contact). Save the resulting EE poses at each timestep.
  This is our synthetic "ground-truth demo".

Phase 2 (Replay x N): Reset the scene, then replay the EXACT recorded
  EE poses through EndEffectorPoseViaPlanning (OMPL). Compare the
  resulting joint trajectories and final EE positions against the
  recorded demo.

This mirrors the GT replay pipeline (replay_rlbench.py --mode sim)
but with zero contact dynamics, isolating OMPL path planning as the
only variable.

If replays match the recording:
  OMPL is deterministic for free-space motion.
  GT replay failures (~50%) come from contact dynamics, not OMPL.

If replays diverge:
  OMPL planner is inherently stochastic (randomized sampling).
  Path divergence contributes to GT replay non-determinism.

Uses close_jar as a host task (just for robot + scene setup) but
does NOT interact with any task objects.

Usage (WSL2 only):
  QT_QPA_PLATFORM=offscreen python ompl_determinism_test.py --replays 5
  QT_QPA_PLATFORM=offscreen python ompl_determinism_test.py --replays 10 --steps 20
  QT_QPA_PLATFORM=offscreen python ompl_determinism_test.py --axis z --delta -0.01

  # With trajectory plots:
  QT_QPA_PLATFORM=offscreen python ompl_determinism_test.py --replays 5 --plot
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _grab_frame(obs):
    """Extract front camera RGB from observation as uint8."""
    frame = obs.front_rgb
    if frame is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    return frame.copy()


def _save_video(frames, path, fps=5):
    """Save list of RGB arrays as MP4."""
    import imageio.v3 as iio
    iio.imwrite(str(path), frames, fps=fps, codec="libx264")


def plot_trajectories(recorded_joints, recorded_ee, all_replay_joints,
                      all_replay_ee, out_dir):
    """Generate trajectory comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_replays = len(all_replay_joints)
    n_joints = recorded_joints.shape[1]
    steps = np.arange(recorded_joints.shape[0])

    # --- Plot 1: Joint trajectories (all 7 joints) ---
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.5 * n_joints),
                             sharex=True)
    fig.suptitle("Joint Trajectories: Recording vs Replays", fontsize=14)

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(steps, recorded_joints[:, j], "k-", linewidth=2,
                label="Recording", zorder=10)
        for r in range(n_replays):
            ax.plot(steps, all_replay_joints[r][:, j], "--", alpha=0.7,
                    label=f"Replay {r}")
        ax.set_ylabel(f"Joint {j} (rad)")
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    p = out_dir / "joint_trajectories.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Plot 2: Joint deviations from recording ---
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.5 * n_joints),
                             sharex=True)
    fig.suptitle("Joint Deviation from Recording (rad)", fontsize=14)

    for j in range(n_joints):
        ax = axes[j]
        for r in range(n_replays):
            diff = all_replay_joints[r][:, j] - recorded_joints[:, j]
            ax.plot(steps, diff, "-", alpha=0.7, label=f"Replay {r}")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"Joint {j} Δ")
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    p = out_dir / "joint_deviations.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Plot 3: EE position (x, y, z) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("End-Effector Position: Recording vs Replays", fontsize=14)
    labels = ["X", "Y", "Z"]

    for dim in range(3):
        ax = axes[dim]
        ax.plot(steps, recorded_ee[:, dim], "k-", linewidth=2,
                label="Recording", zorder=10)
        for r in range(n_replays):
            ax.plot(steps, all_replay_ee[r][:, dim], "--", alpha=0.7,
                    label=f"Replay {r}")
        ax.set_ylabel(f"EE {labels[dim]} (m)")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    p = out_dir / "ee_trajectories.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Plot 4: EE deviation from recording ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("End-Effector Deviation from Recording (m)", fontsize=14)

    for dim in range(3):
        ax = axes[dim]
        for r in range(n_replays):
            diff = all_replay_ee[r][:, dim] - recorded_ee[:, dim]
            ax.plot(steps, diff, "-", alpha=0.7, label=f"Replay {r}")
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"EE {labels[dim]} Δ")
        if dim == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    p = out_dir / "ee_deviations.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

    # --- Plot 5: Summary — max deviation per step ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Max Deviation Across Replays per Step", fontsize=14)

    joint_max_per_step = np.zeros(len(steps))
    ee_max_per_step = np.zeros(len(steps))
    for r in range(n_replays):
        jd = np.nan_to_num(np.abs(all_replay_joints[r] - recorded_joints), nan=0.0)
        ed = np.nan_to_num(np.abs(all_replay_ee[r] - recorded_ee), nan=0.0)
        joint_max_per_step = np.maximum(joint_max_per_step, jd.max(axis=1))
        ee_max_per_step = np.maximum(ee_max_per_step, ed.max(axis=1))

    ax1.bar(steps, joint_max_per_step, color="steelblue", alpha=0.8)
    ax1.set_ylabel("Max joint dev (rad)")
    ax1.axhline(1e-2, color="r", linestyle="--", alpha=0.5, label="1e-2 threshold")
    ax1.legend()

    ax2.bar(steps, ee_max_per_step * 1000, color="coral", alpha=0.8)
    ax2.set_ylabel("Max EE dev (mm)")
    ax2.axhline(1.0, color="r", linestyle="--", alpha=0.5, label="1mm threshold")
    ax2.legend()
    ax2.set_xlabel("Step")

    fig.tight_layout()
    p = out_dir / "deviation_summary.png"
    fig.savefig(str(p), dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")


def run_test(args):
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks

    n_replays = args.replays
    n_steps = args.steps
    delta = args.delta
    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    axis_name = args.axis
    record_video = args.video

    # Obs config — enable front camera if recording video
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)
    if record_video:
        obs_config.front_camera.rgb = True

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=not record_video,  # headless=False needed for vision sensors
    )
    env.launch()

    # Use close_jar as host task (just for the robot + scene)
    task = env.get_task(rlbench.tasks.CloseJar)

    # Video output dir
    if record_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

    print("OMPL Determinism Test (Record-then-Replay)")
    print(f"  Replays:    {n_replays}")
    print(f"  Steps:      {n_steps}")
    print(f"  Axis:       {axis_name}")
    print(f"  Delta:      {delta:.3f} m/step ({delta * n_steps:.3f} m total)")
    if record_video:
        print(f"  Video dir:  {video_dir}")
    print()

    # ===================================================================
    # PHASE 1: Record a free-space trajectory
    # ===================================================================
    print("=" * 60)
    print("PHASE 1: RECORDING free-space trajectory")
    print("=" * 60)

    descriptions, obs = task.reset()
    start_pose = obs.gripper_pose.copy()  # [x, y, z, qx, qy, qz, qw]
    start_joints = obs.joint_positions.copy()

    print(f"  Start EE:     [{start_pose[0]:.4f}, {start_pose[1]:.4f}, {start_pose[2]:.4f}]")
    print(f"  Start joints: [{', '.join(f'{j:.4f}' for j in start_joints)}]")

    # Build target poses: straight line along chosen axis
    target_actions = []
    for step in range(n_steps):
        target_pose = start_pose.copy()
        target_pose[axis_idx] += delta * (step + 1)
        target_actions.append(np.append(target_pose, 1.0))  # gripper open

    # Execute and record — stop at first failure (clean trajectory only)
    recorded_joints = [start_joints.copy()]
    recorded_ee = [start_pose[:3].copy()]
    good_actions = []
    record_frames = []

    if record_video:
        record_frames.append(_grab_frame(obs))

    for step, action in enumerate(target_actions):
        try:
            obs, reward, terminate = task.step(action)
            recorded_joints.append(obs.joint_positions.copy())
            recorded_ee.append(obs.gripper_pose[:3].copy())
            good_actions.append(action)
            if record_video:
                record_frames.append(_grab_frame(obs))
        except Exception as e:
            print(f"  Step {step} FAILED (stopping here): {e}")
            break

    recorded_joints = np.array(recorded_joints)
    recorded_ee = np.array(recorded_ee)
    target_actions = good_actions  # only replay what succeeded
    actual_steps = len(good_actions)

    if actual_steps == 0:
        print("\n  First step failed. Target poses are unreachable.")
        print("  Try a different --axis or --delta value.")
        env.shutdown()
        return float("inf"), float("inf")

    final_rec = recorded_ee[-1]
    print(f"  Recorded:     [{final_rec[0]:.4f}, {final_rec[1]:.4f}, {final_rec[2]:.4f}]")
    print(f"  Good steps:   {actual_steps}/{n_steps}")
    total_dist = np.linalg.norm(recorded_ee[-1] - recorded_ee[0])
    print(f"  Total travel: {total_dist:.4f} m")

    if record_video and record_frames:
        p = video_dir / "ompl_test_recording.mp4"
        _save_video(record_frames, p)
        print(f"  Video saved:  {p}")
    print()

    # ===================================================================
    # PHASE 2: Replay the exact same target poses N times
    # ===================================================================
    print("=" * 60)
    print(f"PHASE 2: REPLAYING recorded poses {n_replays} times")
    print("=" * 60)

    all_replay_joints = []
    all_replay_ee = []

    for replay_idx in range(n_replays):
        # Reset to clean state
        descriptions, obs = task.reset()

        joint_traj = [obs.joint_positions.copy()]
        ee_traj = [obs.gripper_pose[:3].copy()]
        failures = 0
        replay_frames = []

        if record_video:
            replay_frames.append(_grab_frame(obs))

        # Replay the EXACT SAME target actions from Phase 1
        for step, action in enumerate(target_actions):
            try:
                obs, reward, terminate = task.step(action)
                joint_traj.append(obs.joint_positions.copy())
                ee_traj.append(obs.gripper_pose[:3].copy())
                if record_video:
                    replay_frames.append(_grab_frame(obs))
            except Exception as e:
                failures += 1
                joint_traj.append(np.full_like(start_joints, np.nan))
                ee_traj.append(np.full(3, np.nan))

        all_replay_joints.append(np.array(joint_traj))
        all_replay_ee.append(np.array(ee_traj))

        final_pos = ee_traj[-1]
        ee_err = np.linalg.norm(final_pos - final_rec) if not np.any(np.isnan(final_pos)) else float("nan")
        print(f"  Replay {replay_idx}: final=[{final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f}]"
              f"  err={ee_err:.6f} m  fails={failures}")

        if record_video and replay_frames:
            p = video_dir / f"ompl_test_replay{replay_idx}.mp4"
            _save_video(replay_frames, p)
            print(f"    Video: {p}")

    env.shutdown()

    # ===================================================================
    # ANALYSIS
    # ===================================================================
    print(f"\n{'='*60}")
    print("TRAJECTORY COMPARISON (each replay vs recording)")
    print(f"{'='*60}\n")

    max_joint_dev = 0.0
    max_ee_dev = 0.0
    final_ee_errors = []

    for replay_idx in range(n_replays):
        joint_diff = np.abs(all_replay_joints[replay_idx] - recorded_joints)
        ee_diff = np.abs(all_replay_ee[replay_idx] - recorded_ee)

        # NaN entries (failed steps) -> 0 for stats
        joint_diff = np.nan_to_num(joint_diff, nan=0.0)
        ee_diff = np.nan_to_num(ee_diff, nan=0.0)

        run_max_joint = joint_diff.max()
        run_max_ee = ee_diff.max()
        run_mean_joint = joint_diff.mean()
        run_mean_ee = ee_diff.mean()

        max_joint_dev = max(max_joint_dev, run_max_joint)
        max_ee_dev = max(max_ee_dev, run_max_ee)

        # Final position error
        final_err = np.linalg.norm(
            all_replay_ee[replay_idx][-1] - recorded_ee[-1]
        ) if not np.any(np.isnan(all_replay_ee[replay_idx][-1])) else float("nan")
        final_ee_errors.append(final_err)

        print(f"  Replay {replay_idx} vs Recording:")
        print(f"    Joint:     max={run_max_joint:.6e} rad  mean={run_mean_joint:.6e} rad")
        print(f"    EE pos:    max={run_max_ee:.6e} m    mean={run_mean_ee:.6e} m")
        print(f"    Final err: {final_err:.6e} m")

    # Also compare replays against each other
    print(f"\n{'='*60}")
    print("REPLAY-VS-REPLAY COMPARISON")
    print(f"{'='*60}\n")

    max_replay_joint_dev = 0.0
    max_replay_ee_dev = 0.0

    for i in range(n_replays):
        for j in range(i + 1, n_replays):
            jd = np.nan_to_num(np.abs(all_replay_joints[i] - all_replay_joints[j]), nan=0.0)
            ed = np.nan_to_num(np.abs(all_replay_ee[i] - all_replay_ee[j]), nan=0.0)
            max_replay_joint_dev = max(max_replay_joint_dev, jd.max())
            max_replay_ee_dev = max(max_replay_ee_dev, ed.max())
            print(f"  Replay {i} vs Replay {j}: "
                  f"joint max={jd.max():.6e} rad  EE max={ed.max():.6e} m")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Max deviation (replay vs recording):")
    print(f"    Joint:  {max_joint_dev:.6e} rad")
    print(f"    EE pos: {max_ee_dev:.6e} m")
    print(f"  Max deviation (replay vs replay):")
    print(f"    Joint:  {max_replay_joint_dev:.6e} rad")
    print(f"    EE pos: {max_replay_ee_dev:.6e} m")
    valid_errors = [e for e in final_ee_errors if not np.isnan(e)]
    if valid_errors:
        print(f"  Final EE error: mean={np.mean(valid_errors):.6e} m  "
              f"max={np.max(valid_errors):.6e} m")
    print(f"{'='*60}\n")

    # Thresholds
    JOINT_TOL = 1e-6  # rad — essentially zero
    EE_TOL = 1e-6     # m — essentially zero

    if max_joint_dev < JOINT_TOL and max_ee_dev < EE_TOL:
        print("RESULT: DETERMINISTIC")
        print("  OMPL produces identical trajectories for free-space motion.")
        print("  GT replay failures come from contact dynamics, not planner randomness.")
    elif max_joint_dev < 1e-2 and max_ee_dev < 1e-3:
        print("RESULT: NEAR-DETERMINISTIC (small deviations)")
        print(f"  Joint dev {max_joint_dev:.2e} rad, EE dev {max_ee_dev:.2e} m")
        print("  Deviations exist but small. May compound over long contact sequences.")
    else:
        print("RESULT: NON-DETERMINISTIC")
        print(f"  Joint dev {max_joint_dev:.2e} rad, EE dev {max_ee_dev:.2e} m")
        print("  OMPL planner finds different paths each run.")
        print("  This is a contributing factor to GT replay non-determinism.")

    # --- Plots ---
    if args.plot:
        print(f"\n{'='*60}")
        print("GENERATING PLOTS")
        print(f"{'='*60}")
        plot_trajectories(recorded_joints, recorded_ee,
                          all_replay_joints, all_replay_ee,
                          args.plot_dir)

    return max_joint_dev, max_ee_dev


def main():
    parser = argparse.ArgumentParser(
        description="Test OMPL determinism: record free-space motion, replay N times.",
    )
    parser.add_argument(
        "--replays", type=int, default=5,
        help="Number of times to replay the recorded trajectory (default 5).",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Number of waypoints in the straight line (default 20).",
    )
    parser.add_argument(
        "--axis", default="x", choices=["x", "y", "z"],
        help="Axis to move along (default x = lateral).",
    )
    parser.add_argument(
        "--delta", type=float, default=0.002,
        help="Displacement per step in meters (default 0.002 = 2mm). "
             "Use negative values to move in the opposite direction.",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save trajectory comparison plots (matplotlib, no display needed).",
    )
    parser.add_argument(
        "--plot-dir", default="ompl_test_plots",
        help="Directory for plot output (default: ompl_test_plots/).",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Save front-camera MP4 videos. Requires a display (real or Xvfb) "
             "and headless=False — won't work on WSL2 with QT_QPA_PLATFORM=offscreen.",
    )
    parser.add_argument(
        "--video-dir", default="ompl_test_videos",
        help="Directory for video output (default: ompl_test_videos/).",
    )

    args = parser.parse_args()
    max_j, max_ee = run_test(args)

    # Exit code: 0 if deterministic, 1 if not
    sys.exit(0 if max_j < 1e-2 and max_ee < 1e-3 else 1)


if __name__ == "__main__":
    main()
