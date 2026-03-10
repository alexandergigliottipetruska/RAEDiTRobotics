"""Diagnose scene restoration for RLBench demos.

Checks what info the demo objects contain and whether the restored
scene matches the demo's initial state.

Usage (WSL2):
  QT_QPA_PLATFORM=offscreen python -m data_pipeline.scripts.diagnose_scene_restore \
      --raw ../data/raw/rlbench/data/train/close_jar --task close_jar --n 5
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    raw_root = Path(args.raw)
    ep_parent = raw_root / "all_variations" / "episodes"
    ep_dirs = sorted(
        [d for d in ep_parent.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )

    n = min(args.n, len(ep_dirs))

    print(f"=== Scene Restoration Diagnostic: {args.task} ===\n")

    for i in range(n):
        ep_dir = ep_dirs[i]
        with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)

        print(f"--- {ep_dir.name} ---")
        print(f"  type(demo):        {type(demo).__name__}")
        print(f"  has _observations:  {hasattr(demo, '_observations')}")
        print(f"  has random_seed:    {hasattr(demo, 'random_seed')}")

        if hasattr(demo, 'random_seed'):
            rs = demo.random_seed
            print(f"  random_seed:        {type(rs).__name__}, is None: {rs is None}")
            if rs is not None and not callable(rs):
                print(f"  random_seed value:  {rs}")

        print(f"  has variation_number: {hasattr(demo, 'variation_number')}")
        if hasattr(demo, 'variation_number'):
            print(f"  variation_number:   {demo.variation_number}")

        obs_list = demo._observations if hasattr(demo, '_observations') else list(demo)
        print(f"  num observations:   {len(obs_list)}")

        obs0 = obs_list[0]
        print(f"  has misc:           {hasattr(obs0, 'misc')}")
        if hasattr(obs0, 'misc') and obs0.misc is not None:
            print(f"  misc keys:          {list(obs0.misc.keys())}")
            if 'variation_index' in obs0.misc:
                print(f"  variation_index:    {obs0.misc['variation_index']}")
            else:
                print(f"  variation_index:    NOT PRESENT")
        else:
            print(f"  misc:               None or missing")

        print(f"  obs0 gripper_pose:  {obs0.gripper_pose}")
        print(f"  obs0 joint_pos:     {np.array(obs0.joint_positions)}")
        print(f"  obs0 gripper_open:  {obs0.gripper_open}")

        # Check if restore_state is callable
        if hasattr(demo, 'restore_state'):
            print(f"  has restore_state:  True (callable: {callable(demo.restore_state)})")
        else:
            print(f"  has restore_state:  False")

        print()

    # Now try actual scene restoration with CoppeliaSim
    print("=" * 60)
    print("Attempting actual scene restoration in CoppeliaSim...")
    print("=" * 60)

    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointPosition
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig

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

        task_map = {
            "close_jar": "CloseJar",
            "open_drawer": "OpenDrawer",
            "slide_block_to_color_target": "SlideBlockToColorTarget",
        }
        import rlbench.tasks
        task_cls = getattr(rlbench.tasks, task_map[args.task])
        task = env.get_task(task_cls)

        for i in range(n):
            ep_dir = ep_dirs[i]
            with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
                demo = pickle.load(fh)

            obs_list = demo._observations if hasattr(demo, '_observations') else list(demo)
            expected_pose = obs_list[0].gripper_pose
            expected_joints = np.array(obs_list[0].joint_positions)

            # Try reset_to_demo first
            method = "unknown"
            try:
                desc, obs = task.reset_to_demo(demo)
                method = "reset_to_demo"
            except Exception as e1:
                # Fallback: manual restoration
                try:
                    if hasattr(demo, 'restore_state') and callable(demo.restore_state):
                        demo.restore_state()
                    variation = 0
                    if (hasattr(obs_list[0], 'misc') and obs_list[0].misc is not None
                            and 'variation_index' in obs_list[0].misc):
                        variation = obs_list[0].misc['variation_index']
                    task.set_variation(variation)
                    desc, obs = task.reset()
                    method = f"manual (variation={variation})"
                except Exception as e2:
                    print(f"\n  {ep_dir.name}: BOTH methods failed")
                    print(f"    reset_to_demo error: {e1}")
                    print(f"    manual error: {e2}")
                    continue

            actual_pose = obs.gripper_pose
            actual_joints = np.array(obs.joint_positions)

            pose_err = np.abs(actual_pose - expected_pose).max()
            joint_err = np.abs(actual_joints - expected_joints).max()

            print(f"\n  {ep_dir.name} (via {method}):")
            print(f"    gripper pose error: {pose_err:.6f}")
            print(f"    joint pos error:    {joint_err:.6f}")
            print(f"    expected pose: {expected_pose}")
            print(f"    actual pose:   {actual_pose}")
            if pose_err > 0.01:
                print(f"    *** SCENE NOT RESTORED — pose mismatch > 1cm ***")
            else:
                print(f"    Scene restored correctly.")

        env.shutdown()

    except ImportError:
        print("CoppeliaSim not available — skipping live restoration test.")
    except Exception as e:
        print(f"Error during live test: {e}")


if __name__ == "__main__":
    main()
