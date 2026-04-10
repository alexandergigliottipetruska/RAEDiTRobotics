"""Ground-truth replay for RLBench: verify data chain is lossless.

Three modes:

  joint:
    Replay obs.misc["joint_position_action"] from raw demo pickles through
    JointPosition(absolute_mode=True). Matches RLBench's own test
    (test_environment.py:test_executed_jp_action). Expected >=90% success.
    Requires demos collected via RLBench's get_demo() (has joint_position_action).

  keyframe:
    Replay only gripper-transition + endpoint EE poses through OMPL.
    Reduces ~170 OMPL calls to ~3-7. Works with PerAct-style demos.
    Expected ~70%+ success.

  ompl:
    Feed ALL recorded absolute EE poses from HDF5 through OMPL planning.
    Expected ~50% success (CoppeliaSim + OMPL non-determinism ceiling).

Usage (joint — if demos have joint_position_action):
    PYTHONPATH=. python training/gt_replay_rlbench.py \
        --task open_drawer --mode joint \
        --pickles /virtual/csc415user/data/rlbench/train/open_drawer/all_variations/episodes \
        --num_demos 20

Usage (keyframe — recommended for PerAct-style demos):
    PYTHONPATH=. python training/gt_replay_rlbench.py \
        --task open_drawer --mode keyframe \
        --pickles /virtual/csc415user/data/rlbench/train/open_drawer/all_variations/episodes \
        --num_demos 20

Usage (ompl — dense replay from HDF5):
    PYTHONPATH=. python training/gt_replay_rlbench.py \
        --task close_jar --mode ompl \
        --hdf5 /virtual/csc415user/data/rlbench/close_jar.hdf5 \
        --pickles /virtual/csc415user/data/rlbench/valid/close_jar/all_variations/episodes \
        --num_demos 20
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import h5py

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# Task name -> RLBench task class name
TASK_CLASS_MAP = {
    "close_jar": "CloseJar",
    "open_drawer": "OpenDrawer",
    "slide_block_to_color_target": "SlideBlockToColorTarget",
    "put_item_in_drawer": "PutItemInDrawer",
    "stack_cups": "StackCups",
    "place_shape_in_shape_sorter": "PlaceShapeInShapeSorter",
    "meat_off_grill": "MeatOffGrill",
    "turn_tap": "TurnTap",
    "push_buttons": "PushButtons",
    "reach_and_drag": "ReachAndDrag",
    "place_wine_at_rack_location": "PlaceWineAtRackLocation",
    "sweep_to_dustpan_of_size": "SweepToDustpanOfSize",
    "sweep_to_dustpan": "SweepToDustpan",
    "reach_target": "ReachTarget",
}


def get_episode_dirs(pickles_root):
    """Get sorted episode directories from raw episodes folder."""
    ep_root = Path(pickles_root)
    return sorted(
        [d for d in ep_root.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )


def get_obs_list(demo):
    """Extract observation list from Demo object."""
    if hasattr(demo, '_observations'):
        return demo._observations
    return list(demo)


def safe_reset_to_demo(task_env, demo):
    """Reset scene to demo initial state, handling PerAct-style demos.

    RLBench's reset_to_demo expects obs.misc["variation_index"], which
    PerAct-style demos don't have. Falls back to demo.variation_number.
    """
    # Try standard reset_to_demo first
    try:
        return task_env.reset_to_demo(demo)
    except (KeyError, AttributeError):
        pass

    # Fallback: PerAct demos have variation_number on the Demo object
    demo.restore_state()
    variation = getattr(demo, 'variation_number', 0)
    task_env.set_variation(variation)
    return task_env.reset(demo)


# ---------------------------------------------------------------------------
# Mode: joint — replays joint_position_action from pickles
# ---------------------------------------------------------------------------

def gt_replay_joint(task_env, demo):
    """Replay one demo using joint positions.

    Uses obs.misc["joint_position_action"] if available (stepjam RLBench),
    otherwise reconstructs from obs.joint_positions (PerAct-style demos).

    Action format: [joint_positions(7), gripper_open(1)] — matches
    MoveArmThenGripper(JointPosition(absolute_mode=True), Discrete()).

    Returns:
        dict with 'success', 'steps'.
    """
    obs_list = get_obs_list(demo)

    # Try obs.misc["joint_position_action"] first (stepjam RLBench demos)
    jp_actions = []
    if len(obs_list) > 1 and "joint_position_action" in obs_list[1].misc:
        for t, obs in enumerate(obs_list):
            if t == 0:
                continue
            jp_actions.append(obs.misc["joint_position_action"])
        log.info("  Using joint_position_action from obs.misc (%d actions)", len(jp_actions))
    else:
        # Reconstruct from obs.joint_positions (PerAct-style demos)
        # Action[t] = [joint_positions[t+1], gripper_open[t]]
        for t in range(len(obs_list) - 1):
            jp_next = obs_list[t + 1].joint_positions
            grip = float(obs_list[t].gripper_open)
            jp_actions.append(np.append(jp_next, grip))
        log.info("  Reconstructed from obs.joint_positions (%d actions)", len(jp_actions))

    if not jp_actions:
        log.warning("  No actions to replay")
        return {"success": False, "steps": 0}

    for t, action in enumerate(jp_actions):
        try:
            obs, reward, terminate = task_env.step(action)
        except Exception as e:
            log.warning("  Step %d failed: %s", t, e)
            return {"success": False, "steps": t}
        if reward == 1.0:
            return {"success": True, "steps": t + 1}
        if terminate:
            break

    return {"success": False, "steps": len(jp_actions)}


def run_joint_mode(args):
    """Run GT replay in joint mode using JointPosition(absolute_mode=True)."""
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks

    ep_dirs = get_episode_dirs(args.pickles)
    ep_dirs = ep_dirs[:args.num_demos]
    log.info("Found %d episode pickles in %s", len(ep_dirs), args.pickles)

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=JointPosition(absolute_mode=True),
        gripper_action_mode=Discrete(),
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    task_cls = getattr(rlbench.tasks, TASK_CLASS_MAP[args.task])
    task_env = env.get_task(task_cls)

    results = []
    for i, ep_dir in enumerate(ep_dirs):
        with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)

        try:
            descriptions, obs = safe_reset_to_demo(task_env, demo)
        except Exception as e:
            log.warning("Demo %d (%s): reset failed, skipping: %s",
                        i, ep_dir.name, e)
            results.append({"success": False, "steps": 0})
            continue

        result = gt_replay_joint(task_env, demo)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        n_success = sum(r["success"] for r in results)
        log.info("Demo %d/%d (%s): %s (%d steps) | Running: %d/%d (%.0f%%)",
                 i + 1, len(ep_dirs), ep_dir.name, status, result["steps"],
                 n_success, len(results), 100 * n_success / len(results))

    env.shutdown()

    n_success = sum(r["success"] for r in results)
    log.info("=== %s GT replay (joint): %d/%d (%.1f%%) ===",
             args.task, n_success, len(results), 100 * n_success / len(results))


# ---------------------------------------------------------------------------
# Mode: keyframe — replays gripper-transition + endpoint EE poses via OMPL
# ---------------------------------------------------------------------------

def _get_grip_cmd(obs):
    """Get commanded gripper state: from joint_position_action if available,
    else from obs.gripper_open.

    obs.gripper_open can be wrong for grasping tasks (e.g. close_jar) —
    the fingers don't fully close around an object, so get_open_amount()
    still reads 1.0 even though the gripper was commanded to close.
    joint_position_action[-1] is the actual commanded gripper target.
    """
    jpa = obs.misc.get("joint_position_action", None)
    if jpa is not None:
        return jpa[-1]  # commanded gripper: 0.0=close, 1.0=open
    return float(obs.gripper_open)


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    """Check if robot is stopped at timestep i (from PerAct heuristic)."""
    next_is_not_final = i == (len(demo) - 2)
    grip_i = _get_grip_cmd(obs)
    grip_next = _get_grip_cmd(demo[i + 1]) if i < len(demo) - 1 else grip_i
    grip_prev = _get_grip_cmd(demo[i - 1]) if i > 0 else grip_i
    grip_prev2 = _get_grip_cmd(demo[i - 2]) if i > 1 else grip_prev
    gripper_state_no_change = (
        i < (len(demo) - 2) and
        (grip_i == grip_next and grip_i == grip_prev and grip_prev2 == grip_prev))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(obs_list, stopping_delta=0.1):
    """Discover keypoints using PerAct's heuristic.

    Finds timesteps where:
    - Gripper command changes (from joint_position_action, not obs.gripper_open)
    - Robot is stopped (joint velocities near zero)
    - Episode ends

    Uses joint_position_action[-1] for gripper state when available,
    since obs.gripper_open is unreliable for grasping tasks.

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


def extract_keyframe_actions(obs_list):
    """Extract keyframe actions using PerAct's keypoint discovery.

    Instead of replaying all ~170 EE poses through OMPL, only replay the
    keyframe poses (gripper changes + velocity stops + endpoint).

    Uses joint_position_action[-1] for correct gripper command.

    Returns:
        list of 8D actions [pose(7), gripper(1)] at keyframe timesteps.
    """
    kf_indices = keypoint_discovery(obs_list)

    actions = []
    for idx in kf_indices:
        pose = obs_list[idx].gripper_pose  # [x, y, z, qx, qy, qz, qw]
        grip = 1.0 if _get_grip_cmd(obs_list[idx]) > 0.5 else 0.0
        actions.append(np.append(pose, grip))

    return actions, kf_indices


def gt_replay_keyframe(task_env, obs_list):
    """Replay keyframe EE poses through OMPL.

    Returns:
        dict with 'success', 'steps'.
    """
    actions, kf_indices = extract_keyframe_actions(obs_list)
    log.info("  Keyframes: %d actions (indices: %s)", len(actions), kf_indices)

    for t, action in enumerate(actions):
        try:
            obs, reward, terminate = task_env.step(action)
        except Exception as e:
            log.warning("  Keyframe %d failed: %s", t, e)
            continue  # Skip failed keyframes, try next
        if reward == 1.0:
            return {"success": True, "steps": t + 1}
        if terminate:
            break

    return {"success": False, "steps": len(actions)}


def run_keyframe_mode(args):
    """Run GT replay in keyframe mode (OMPL with fewer planning calls)."""
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks

    ep_dirs = get_episode_dirs(args.pickles)
    ep_dirs = ep_dirs[:args.num_demos]
    log.info("Found %d episode pickles in %s", len(ep_dirs), args.pickles)

    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    task_cls = getattr(rlbench.tasks, TASK_CLASS_MAP[args.task])
    task_env = env.get_task(task_cls)

    results = []
    for i, ep_dir in enumerate(ep_dirs):
        with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)

        obs_list = get_obs_list(demo)

        try:
            descriptions, obs = safe_reset_to_demo(task_env, demo)
        except Exception as e:
            log.warning("Demo %d (%s): reset failed, skipping: %s",
                        i, ep_dir.name, e)
            results.append({"success": False, "steps": 0})
            continue

        result = gt_replay_keyframe(task_env, obs_list)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        n_success = sum(r["success"] for r in results)
        log.info("Demo %d/%d (%s): %s (%d steps) | Running: %d/%d (%.0f%%)",
                 i + 1, len(ep_dirs), ep_dir.name, status, result["steps"],
                 n_success, len(results), 100 * n_success / len(results))

    env.shutdown()

    n_success = sum(r["success"] for r in results)
    log.info("=== %s GT replay (keyframe): %d/%d (%.1f%%) ===",
             args.task, n_success, len(results), 100 * n_success / len(results))


# ---------------------------------------------------------------------------
# Mode: ompl (legacy) — replays ALL absolute EE poses from HDF5 through OMPL
# ---------------------------------------------------------------------------

def gt_replay_demo(env, actions):
    """Replay one demo's actions through the environment via OMPL.

    Args:
        env: RLBenchWrapper instance (already reset via reset_to_demo).
        actions: (T, 8) float32 actions [pos(3), quat(4), grip_centered(1)].

    Returns:
        dict with 'success', 'steps'.
    """
    for t in range(len(actions)):
        try:
            obs, reward, done, info = env.step(actions[t])
        except Exception as e:
            log.warning("  OMPL failed at step %d: %s", t, e)
            return {"success": False, "steps": t}
        if info["success"]:
            return {"success": True, "steps": t + 1}
        if done:
            break

    return {"success": False, "steps": t + 1}


def run_ompl_mode(args):
    """Run GT replay in OMPL mode (legacy, dense)."""
    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

    if not args.hdf5:
        raise ValueError("--hdf5 is required for ompl mode")

    # Load actions from HDF5
    log.info("Loading %s from %s", args.task, args.hdf5)
    with h5py.File(args.hdf5, "r") as f:
        demo_keys = [k.decode() for k in f[f"mask/{args.split}"][:]]
        demo_keys = demo_keys[:args.num_demos]
        all_actions = {}
        for dk in demo_keys:
            all_actions[dk] = f[f"data/{dk}/actions"][:]
    log.info("Loaded %d demos (%s split)", len(demo_keys), args.split)

    # Map episode directories
    ep_dirs = get_episode_dirs(args.pickles)
    log.info("Found %d episode pickles in %s", len(ep_dirs), args.pickles)

    log.info("Creating RLBenchWrapper for %s...", args.task)
    env = RLBenchWrapper(task_name=args.task, headless=True, cameras=False)

    results = []
    for i, dk in enumerate(demo_keys):
        actions = all_actions[dk]

        ep_dir = ep_dirs[i]
        with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)

        # Restore scene to demo initial state
        try:
            descriptions, obs = safe_reset_to_demo(env._task, demo)
        except Exception as e:
            log.warning("Demo %d (%s, %s): reset failed, skipping: %s",
                        i, dk, ep_dir.name, e)
            results.append({"success": False, "steps": 0})
            continue

        env._last_obs = obs

        result = gt_replay_demo(env, actions)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        n_success = sum(r["success"] for r in results)
        log.info("Demo %d/%d (%s, %s): %s (%d steps) | Running: %d/%d (%.0f%%)",
                 i + 1, len(demo_keys), dk, ep_dir.name, status, result["steps"],
                 n_success, len(results), 100 * n_success / len(results))

    env.close()

    n_success = sum(r["success"] for r in results)
    log.info("=== %s GT replay (ompl): %d/%d (%.1f%%) ===",
             args.task, n_success, len(results), 100 * n_success / len(results))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_ik_mode(args):
    """Run GT replay in IK mode — direct IK, no OMPL planning (closest to Robomimic OSC)."""
    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

    ep_dirs = get_episode_dirs(args.pickles)
    ep_dirs = ep_dirs[:args.num_demos]
    log.info("Found %d episode pickles in %s", len(ep_dirs), args.pickles)

    log.info("Creating RLBenchWrapper for %s (IK mode)...", args.task)
    env = RLBenchWrapper(task_name=args.task, headless=True, cameras=False, use_ik=True)

    results = []
    for i, ep_dir in enumerate(ep_dirs):
        with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)

        try:
            descriptions, obs = safe_reset_to_demo(env._task, demo)
        except Exception as e:
            log.warning("Demo %d (%s): reset failed: %s", i, ep_dir.name, e)
            results.append({"success": False, "steps": 0})
            continue

        env._last_obs = obs

        # Extract dense EE pose actions from demo observations
        # Use joint_position_action[-1] for gripper command (not gripper_open,
        # which reports observed state and lags behind for tasks like close_jar)
        actions = []
        for t in range(1, len(demo)):
            obs_t = demo[t]
            pose = obs_t.gripper_pose  # [x, y, z, qx, qy, qz, qw]
            jpa = obs_t.misc.get("joint_position_action", None)
            if jpa is not None:
                grip = 1.0 if jpa[-1] > 0.5 else -1.0
            else:
                grip = 1.0 if obs_t.gripper_open else -1.0
            actions.append(np.concatenate([pose, [grip]]))

        actions = np.array(actions, dtype=np.float32)
        log.info("  IK replay: %d actions from demo observations", len(actions))

        result = gt_replay_demo(env, actions)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        n_success = sum(r["success"] for r in results)
        log.info("Demo %d/%d (%s): %s (%d steps) | Running: %d/%d (%.0f%%)",
                 i + 1, len(ep_dirs), ep_dir.name, status, result["steps"],
                 n_success, len(results), 100 * n_success / len(results))

    env.close()
    n_success = sum(r["success"] for r in results)
    log.info("=== %s GT replay (ik): %d/%d (%.1f%%) ===",
             args.task, n_success, len(results), 100 * n_success / len(results))


def main():
    parser = argparse.ArgumentParser(description="GT replay for RLBench")
    parser.add_argument("--task", required=True, choices=list(TASK_CLASS_MAP.keys()))
    parser.add_argument("--mode", default="keyframe",
                        choices=["joint", "keyframe", "ompl", "ik"],
                        help="joint: replay joint_position_action (>=90%%, needs RLBench demos). "
                             "keyframe: replay gripper-transition EE poses via OMPL (~70%%). "
                             "ompl: replay ALL EE poses via OMPL (~50%%). "
                             "ik: replay EE poses via direct IK, no planning (closest to Robomimic).")
    parser.add_argument("--hdf5", help="HDF5 path (required for ompl mode)")
    parser.add_argument("--pickles", required=True,
                        help="Path to raw episodes dir with low_dim_obs.pkl files")
    parser.add_argument("--num_demos", type=int, default=20)
    parser.add_argument("--split", default="train", help="HDF5 split (ompl mode)")
    args = parser.parse_args()

    if args.mode == "joint":
        run_joint_mode(args)
    elif args.mode == "keyframe":
        run_keyframe_mode(args)
    elif args.mode == "ik":
        run_ik_mode(args)
    else:
        run_ompl_mode(args)


if __name__ == "__main__":
    main()
