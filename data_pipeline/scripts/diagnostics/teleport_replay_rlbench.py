"""Direct joint-position teleport replay for RLBench.

Bypasses OMPL motion planning entirely. For each demo timestep,
directly sets robot arm joint positions to the recorded values,
then steps the physics engine.

Purpose: Diagnose whether GT replay failures (~50%) come from:
  - OMPL planner non-determinism   (teleport >> 50% -> OMPL is the cause)
  - CoppeliaSim physics engine     (teleport ~= 50% -> deeper issue)

Key caveat: teleportation skips intermediate contact dynamics. Between
recorded observations the robot originally moved smoothly, generating
contact forces (pushing, grasping). Teleporting skips those intermediate
forces, so tasks requiring sustained contact (pushing, pulling) may fail
even with a deterministic physics engine. The --substeps flag adds extra
physics steps per observation to let contacts settle.

Interpretation guide:
  ~100%  -> OMPL planner is the sole noise source.
  ~50%   -> Physics engine itself is non-deterministic, OR teleportation
            breaks contact dynamics. Run with --substeps 10 to distinguish.
  <50%   -> Teleportation actively breaks the task (expected for pushing
            tasks). Compare with --substeps to see if settling helps.

Usage (WSL2 only):
  QT_QPA_PLATFORM=offscreen python teleport_replay_rlbench.py \
      --raw ~/rlbench_replay/raw/close_jar \
      --task close_jar --n 20

  # With extra physics settling steps:
  QT_QPA_PLATFORM=offscreen python teleport_replay_rlbench.py \
      --raw ~/rlbench_replay/raw/close_jar \
      --task close_jar --n 20 --substeps 5

  # Also teleport gripper joints (full teleport, no dynamics at all):
  QT_QPA_PLATFORM=offscreen python teleport_replay_rlbench.py \
      --raw ~/rlbench_replay/raw/close_jar \
      --task close_jar --n 20 --teleport-gripper
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# All 8 RLBench tasks
TASK_MAP = {
    "close_jar": "CloseJar",
    "open_drawer": "OpenDrawer",
    "slide_block_to_color_target": "SlideBlockToColorTarget",
    "meat_off_grill": "MeatOffGrill",
    "turn_tap": "TurnTap",
    "push_buttons": "PushButtons",
    "place_wine_at_rack_location": "PlaceWineAtRackLocation",
    "sweep_to_dustpan_of_size": "SweepToDustpanOfSize",
}


# ---------------------------------------------------------------------------
# Helpers (shared with replay_rlbench.py)
# ---------------------------------------------------------------------------

def get_episode_dirs(raw_root: Path, n: int):
    """Get sorted episode directories from raw task folder."""
    ep_parent = raw_root / "all_variations" / "episodes"
    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )
    return ep_dirs[:n]


def load_demo_pickle(ep_dir: Path):
    """Load Demo object from low_dim_obs.pkl."""
    with open(ep_dir / "low_dim_obs.pkl", "rb") as f:
        return pickle.load(f)


def get_observations(demo):
    """Extract observation list from Demo object."""
    if hasattr(demo, "_observations"):
        return demo._observations
    return list(demo)


# ---------------------------------------------------------------------------
# Teleport replay
# ---------------------------------------------------------------------------

def run_teleport(args):
    """Replay by directly teleporting arm joints (bypassing OMPL)."""
    from rlbench.environment import Environment
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.observation_config import ObservationConfig
    import rlbench.tasks

    task_name = args.task
    if task_name not in TASK_MAP:
        sys.exit(f"Unknown task: {task_name}. Known: {list(TASK_MAP.keys())}")

    raw_root = Path(args.raw).resolve()
    ep_dirs = get_episode_dirs(raw_root, args.n)

    # Minimal obs config (no cameras — faster)
    obs_config = ObservationConfig()
    obs_config.set_all_low_dim(True)
    obs_config.set_all_high_dim(False)

    # Need an action mode for Environment init, but we won't use task.step()
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

    task_cls = getattr(rlbench.tasks, TASK_MAP[task_name])
    task_env = env.get_task(task_cls)

    # Access internals for direct joint control
    robot = task_env._robot
    pyrep = task_env._pyrep

    substeps = args.substeps

    print(f"Teleport replay: {task_name}")
    print(f"  Raw:      {raw_root}")
    print(f"  Episodes: {len(ep_dirs)}")
    print(f"  Substeps: {substeps}")
    print(f"  Gripper:  {'teleport' if args.teleport_gripper else 'actuate'}")
    print()

    successes = []
    ompl_baseline = {
        "close_jar": "10/20 (50%)",
        "open_drawer": "10/20 (50%)",
        "slide_block_to_color_target": "9/20 (45%)",
        "meat_off_grill": "5/20 (25%)",
        "turn_tap": "11/20 (55%)",
        "push_buttons": "11/20 (55%)",
        "place_wine_at_rack_location": "6/20 (30%)",
        "sweep_to_dustpan_of_size": "11/20 (55%)",
    }

    for i, ep_dir in enumerate(ep_dirs):
        demo = load_demo_pickle(ep_dir)
        obs_list = get_observations(demo)
        T = len(obs_list)

        # Restore scene to demo initial state
        task_env.set_variation(demo.variation_number)
        descriptions, obs = task_env.reset_to_demo(demo)

        success = False
        for t in range(1, T):
            # --- Teleport arm joints ---
            jp = obs_list[t].joint_positions
            robot.arm.set_joint_positions(jp)
            robot.arm.set_joint_target_positions(jp)

            # --- Gripper ---
            if args.teleport_gripper:
                # Full teleport: also set gripper joint positions directly
                if hasattr(obs_list[t], "gripper_joint_positions"):
                    gjp = obs_list[t].gripper_joint_positions
                    robot.gripper.set_joint_positions(gjp)
                    robot.gripper.set_joint_target_positions(gjp)
                else:
                    # Fallback: actuate
                    grip = 1.0 if obs_list[t].gripper_open > 0.5 else 0.0
                    robot.gripper.actuate(grip, velocity=0.04)
            else:
                # Normal actuation (lets physics handle grasping)
                grip = 1.0 if obs_list[t].gripper_open > 0.5 else 0.0
                robot.gripper.actuate(grip, velocity=0.04)

            # --- Step physics ---
            for _ in range(substeps):
                pyrep.step()

            # --- Check success ---
            success, _ = task_env._task.success()
            if success:
                break

        successes.append(success)
        print(f"  episode{i:<4d} | steps={T-1:4d} | "
              f"{'SUCCESS' if success else 'FAIL'}")

    env.shutdown()

    n_ok = sum(successes)
    n_total = len(successes)
    rate = n_ok / n_total if n_total else 0

    print(f"\n{'='*60}")
    print(f"Teleport result:  {n_ok}/{n_total} ({rate * 100:.0f}%)")
    if task_name in ompl_baseline:
        print(f"OMPL baseline:    {ompl_baseline[task_name]}")
    print(f"{'='*60}")

    # Interpretation
    ompl_rates = {
        "close_jar": 0.50, "open_drawer": 0.50,
        "slide_block_to_color_target": 0.45, "meat_off_grill": 0.25,
        "turn_tap": 0.55, "push_buttons": 0.55,
        "place_wine_at_rack_location": 0.30, "sweep_to_dustpan_of_size": 0.55,
    }
    ompl_rate = ompl_rates.get(task_name, 0.50)

    if rate >= ompl_rate + 0.25:
        print("INTERPRETATION: Teleport >> OMPL -> OMPL planner is the noise source.")
    elif abs(rate - ompl_rate) <= 0.15:
        print("INTERPRETATION: Teleport ~= OMPL -> physics engine non-determinism")
        print("  (or teleportation breaks contact dynamics for this task).")
    elif rate < ompl_rate - 0.15:
        print("INTERPRETATION: Teleport < OMPL -> teleportation breaks this task's")
        print("  contact dynamics. Try --substeps 10 for more settling time.")
    else:
        print("INTERPRETATION: Marginal difference. Run more episodes (--n 50).")

    return rate


def main():
    parser = argparse.ArgumentParser(
        description="Direct joint teleport replay for RLBench (bypasses OMPL).",
    )
    parser.add_argument(
        "--raw", required=True,
        help="Raw RLBench task dir (e.g. ~/rlbench_replay/raw/close_jar).",
    )
    parser.add_argument(
        "--task", required=True,
        choices=list(TASK_MAP.keys()),
        help="Task name.",
    )
    parser.add_argument("--n", type=int, default=20, help="Number of episodes.")
    parser.add_argument(
        "--substeps", type=int, default=1,
        help="Physics steps per observation (default 1). Increase to let "
             "contacts settle after teleportation.",
    )
    parser.add_argument(
        "--teleport-gripper", action="store_true",
        help="Also teleport gripper joint positions (default: actuate via "
             "physics, which is needed for grasping).",
    )

    args = parser.parse_args()
    rate = run_teleport(args)
    sys.exit(0 if rate >= 0.4 else 1)


if __name__ == "__main__":
    main()
