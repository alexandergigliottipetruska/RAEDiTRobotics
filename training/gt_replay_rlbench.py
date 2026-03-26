"""Ground-truth replay for RLBench: feed recorded actions through OMPL.

Verifies the regenerated HDF5 data chain is lossless.
Expected: ~50% success (CoppeliaSim non-determinism ceiling).

Usage:
    PYTHONPATH=. python training/gt_replay_rlbench.py \
        --task close_jar \
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


def gt_replay_demo(env, actions):
    """Replay one demo's actions through the environment.

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--pickles", required=True,
                        help="Path to raw episodes dir with low_dim_obs.pkl files")
    parser.add_argument("--num_demos", type=int, default=20)
    parser.add_argument("--split", default="valid", help="Use valid split for GT replay")
    args = parser.parse_args()

    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

    # Load actions from HDF5
    log.info("Loading %s from %s", args.task, args.hdf5)
    with h5py.File(args.hdf5, "r") as f:
        demo_keys = [k.decode() for k in f[f"mask/{args.split}"][:]]
        demo_keys = demo_keys[:args.num_demos]
        all_actions = {}
        for dk in demo_keys:
            all_actions[dk] = f[f"data/{dk}/actions"][:]
    log.info("Loaded %d demos (%s split)", len(demo_keys), args.split)

    # Map episode directories (sorted by index)
    ep_root = Path(args.pickles)
    ep_dirs = sorted(
        [d for d in ep_root.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )
    log.info("Found %d episode pickles in %s", len(ep_dirs), ep_root)

    log.info("Creating RLBenchWrapper for %s...", args.task)
    env = RLBenchWrapper(task_name=args.task, headless=True, cameras=False)

    results = []
    for i, dk in enumerate(demo_keys):
        actions = all_actions[dk]

        # Load demo pickle for scene restoration
        # Mapping: demo_100 -> episode0, demo_101 -> episode1, etc.
        ep_dir = ep_dirs[i]
        with open(ep_dir / "low_dim_obs.pkl", "rb") as fh:
            demo = pickle.load(fh)

        # Restore scene to demo initial state
        try:
            descriptions, obs = env._task.reset_to_demo(demo)
        except (KeyError, AttributeError):
            env._task.set_variation(demo.variation_number)
            descriptions, obs = env._task.reset()

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
    log.info("=== %s GT replay: %d/%d (%.1f%%) ===",
             args.task, n_success, len(results), 100 * n_success / len(results))


if __name__ == "__main__":
    main()
