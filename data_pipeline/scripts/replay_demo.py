"""Demo replay test (spec section 8.4).

Loads raw actions from a unified HDF5 and replays them in robosuite.
At least 4/5 demos must succeed (>=80%). Failure = conversion bug.

Usage:
  python data_pipeline/scripts/replay_demo.py \
      --hdf5 "path/to/unified/robomimic/lift/ph.hdf5" \
      --raw  "path/to/raw/robomimic/lift/ph/image.hdf5" \
      --task lift --n 5

Notes:
  - Actions must be fed RAW (not normalized). The unified HDF5 stores raw
    actions; normalization only happens inside the dataset class at train time.
  - env.reset() restores the initial state from the raw HDF5 (via model_xml
    + init_state). Without exact state restoration, success < 100% is normal.
  - Run from the repo root with the venv active.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# Add repo root to path so data_pipeline imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_pipeline.conversion.unified_schema import read_mask


# Map task name -> robosuite env name
_TASK_TO_ENV = {
    "lift":      "Lift",
    "can":       "PickPlaceCan",
    "square":    "NutAssemblySquare",
    "tool_hang": "ToolHang",
}


def make_env(task: str):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name=_TASK_TO_ENV[task],
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        controller_configs=controller_config,
        ignore_done=True,
    )
    return env


def replay_demo(env, actions: np.ndarray, init_state=None) -> bool:
    """Replay one demo. Returns True if the episode succeeds."""
    env.reset()
    if init_state is not None:
        env.sim.set_state_from_flattened(init_state)
        env.sim.forward()

    for t, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        if env._check_success():
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", required=True,
                        help="Path to unified HDF5 (e.g. data/unified/robomimic/lift/ph.hdf5)")
    parser.add_argument("--raw", required=True,
                        help="Path to raw robomimic image.hdf5 (for initial states)")
    parser.add_argument("--task", default="lift", choices=list(_TASK_TO_ENV.keys()))
    parser.add_argument("--n", type=int, default=5, help="Number of demos to replay")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"Task: {args.task}")
    print(f"Unified HDF5: {args.hdf5}")
    print(f"Raw HDF5:     {args.raw}")
    print()

    # Load demo keys and actions from unified HDF5
    with h5py.File(args.hdf5, "r") as f:
        keys = read_mask(f, args.split)[: args.n]
        actions_list = [f[f"data/{k}/actions"][:] for k in keys]

    # Load initial states from raw HDF5 (needed for deterministic reset)
    with h5py.File(args.raw, "r") as f:
        init_states = []
        for k in keys:
            if "states" in f[f"data/{k}"]:
                init_states.append(f[f"data/{k}/states"][0])
            else:
                init_states.append(None)

    env = make_env(args.task)
    successes = []

    for i, (key, actions, init_state) in enumerate(zip(keys, actions_list, init_states)):
        success = replay_demo(env, actions, init_state)
        successes.append(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"  Demo {key:10s} | T={len(actions):3d} | {status}")

    env.close()

    n_success = sum(successes)
    rate = n_success / len(successes)
    print(f"\nResult: {n_success}/{len(successes)} succeeded ({rate*100:.0f}%)")

    if rate < 0.8:
        print("FAIL: Success rate below 80% threshold — conversion bug likely.")
        sys.exit(1)
    else:
        print("PASS: Demo replay meets >=80% threshold.")


if __name__ == "__main__":
    main()
