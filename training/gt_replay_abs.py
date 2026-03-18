"""GT replay test for absolute actions through the RobomimicWrapper.

Loads absolute actions + initial states from an HDF5, replays them through
the wrapper with abs_action=True, and reports success rate. This isolates
the controller config from the policy — if GT replay fails, the controller
is misconfigured.

Usage:
  # Using Chi's raw image_abs.hdf5 (has states):
  python training/gt_replay_abs.py \
    --hdf5 data/raw/robomimic/lift/liftfromdp/ph/image_abs.hdf5 \
    --num_demos 10

  # Using unified ph_abs.hdf5 + separate raw HDF5 for states:
  python training/gt_replay_abs.py \
    --hdf5 data/unified/robomimic/lift/ph_abs.hdf5 \
    --states_hdf5 data/raw/robomimic/lift/liftfromdp/ph/image_abs.hdf5 \
    --num_demos 10

  # With 10D rot6d actions (auto-converts to 7D axis-angle):
  python training/gt_replay_abs.py \
    --hdf5 data/unified/robomimic/lift/ph_abs_rot6d.hdf5 \
    --states_hdf5 data/raw/robomimic/lift/liftfromdp/ph/image_abs.hdf5 \
    --num_demos 10
"""

import argparse
import logging
import os
import sys
import warnings

import h5py
import numpy as np

warnings.filterwarnings("ignore", module="robosuite")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GT replay with absolute actions")
    parser.add_argument("--hdf5", required=True,
                        help="HDF5 with absolute actions (raw or unified)")
    parser.add_argument("--states_hdf5", default=None,
                        help="Separate HDF5 for initial states (if not in --hdf5)")
    parser.add_argument("--task", default="lift")
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

    # Open HDF5 files
    action_file = h5py.File(args.hdf5, "r")

    # Check if the main HDF5 has states; if not, require --states_hdf5
    first_demo = sorted(action_file["data"].keys())[0]
    has_states = "states" in action_file[f"data/{first_demo}"]
    if has_states:
        states_file = action_file
        log.info("States found in main HDF5")
    elif args.states_hdf5:
        states_file = h5py.File(args.states_hdf5, "r")
        log.info("States loaded from %s", args.states_hdf5)
    else:
        log.error("No states in HDF5 and no --states_hdf5 provided. "
                   "Cannot do GT replay without initial states.")
        action_file.close()
        return

    # Find demo keys
    demo_keys = sorted(action_file["data"].keys())[:args.num_demos]
    log.info("Found %d demos, testing %d", len(action_file["data"].keys()), len(demo_keys))

    # Check action dimension
    sample_actions = action_file[f"data/{demo_keys[0]}/actions"][:]
    ac_dim = sample_actions.shape[-1]
    log.info("Action dimension: %d", ac_dim)
    log.info("Sample action[0]: %s", np.array2string(sample_actions[0], precision=4))

    rot6d = ac_dim == 10
    if rot6d:
        log.info("10D actions detected — will convert rot6d → axis_angle")
        from data_pipeline.utils.rotation import convert_actions_from_rot6d

    # Create env with absolute action mode
    log.info("Creating %s env with abs_action=True", args.task)
    env = RobomimicWrapper(task=args.task, seed=args.seed, abs_action=True)

    successes = 0
    for demo_key in demo_keys:
        actions = action_file[f"data/{demo_key}/actions"][:]
        initial_state = states_file[f"data/{demo_key}/states"][0]

        # Convert 10D rot6d → 7D axis-angle if needed
        if rot6d:
            actions = convert_actions_from_rot6d(actions)

        # Reset env to initial state
        env.reset()
        env._env.sim.set_state_from_flattened(initial_state)
        env._env.sim.forward()
        env._last_obs = env._env._get_observations()

        success = False
        for t in range(len(actions)):
            _, _, _, info = env.step(actions[t])
            if info["success"]:
                success = True
                break

        successes += int(success)
        status = "SUCCESS" if success else "FAIL"
        log.info("  %s: %s (step %d/%d)", demo_key, status, t + 1, len(actions))

    env.close()
    if states_file is not action_file:
        states_file.close()
    action_file.close()

    log.info("=" * 50)
    log.info("GT replay (absolute): %d/%d (%.1f%%)",
             successes, len(demo_keys), 100 * successes / len(demo_keys))
    log.info("=" * 50)

    if successes >= len(demo_keys) * 0.9:
        log.info("PASSED — controller config is correct")
    else:
        log.warning("FAILED — controller config may be wrong, or data mismatch")


if __name__ == "__main__":
    main()
