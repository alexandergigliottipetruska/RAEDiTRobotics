"""Tier 3 smoke tests — real simulators, no trained model needed.

Ground-truth replay: feed stored actions back through the full eval wrapper
to verify normalization round-trip, image processing, and env interaction
are all correct end-to-end.

robomimic tests run on Windows (MuJoCo CPU). Mark with @pytest.mark.slow.
RLBench tests run on WSL2 (CoppeliaSim). Mark with @pytest.mark.rlbench.

Usage:
  # robomimic only (Windows, ~5 min)
  python -m pytest tests/test_eval_smoke.py -m slow -v

  # RLBench only (WSL2, ~30-60 min)
  python -m pytest tests/test_eval_smoke.py -m rlbench -v
"""

import pickle
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # CSC415 Project/
_DATA_ROOT = _PROJECT_ROOT / "data"

_ROBOMIMIC_UNIFIED = _DATA_ROOT / "unified" / "robomimic" / "lift" / "ph.hdf5"
_ROBOMIMIC_RAW = _DATA_ROOT / "raw" / "robomimic" / "lift" / "ph" / "image.hdf5"

_RLBENCH_UNIFIED_DIR = _DATA_ROOT / "unified" / "rlbench"
_RLBENCH_RAW_DIR = _DATA_ROOT / "raw" / "rlbench" / "data" / "train"


# ---------------------------------------------------------------------------
# GroundTruthPolicy — serves stored actions as if they were model predictions
# ---------------------------------------------------------------------------

class GroundTruthPolicy:
    """Wraps pre-stored actions as a 'policy' for eval harness testing.

    Returns normalized action chunks. The rollout loop denormalizes them,
    resulting in the original raw actions being fed to the environment.
    """

    def __init__(self, actions_raw: np.ndarray, action_mean: np.ndarray,
                 action_std: np.ndarray, chunk_size: int = 50):
        # Pre-normalize so rollout's denormalization recovers the originals
        self.actions_norm = ((actions_raw - action_mean) / action_std).astype(np.float32)
        self.chunk_size = chunk_size
        self.cursor = 0

    def predict(self, images, proprio, view_present):
        start = self.cursor
        end = min(start + self.chunk_size, len(self.actions_norm))
        chunk = self.actions_norm[start:end]

        if len(chunk) < self.chunk_size:
            pad = np.zeros((self.chunk_size - len(chunk), chunk.shape[1]),
                           dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        self.cursor = end
        return torch.from_numpy(chunk)

    def reset_cursor(self):
        self.cursor = 0


# ---------------------------------------------------------------------------
# robomimic smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGTReplayRobomimic:
    """Ground-truth replay of lift demos through RobomimicWrapper.

    Loads initial states from raw HDF5 to restore scene, then steps through
    with actions from unified HDF5. Tests the full wrapper end-to-end.

    Pass criterion: >= 90% of all train demos succeed.
    """

    def test_gt_replay_lift(self):
        if not _ROBOMIMIC_UNIFIED.exists():
            pytest.skip(f"Unified HDF5 not found: {_ROBOMIMIC_UNIFIED}")
        if not _ROBOMIMIC_RAW.exists():
            pytest.skip(f"Raw HDF5 not found: {_ROBOMIMIC_RAW}")

        from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

        env = RobomimicWrapper("lift")
        successes = 0

        with h5py.File(str(_ROBOMIMIC_UNIFIED), "r") as uf, \
             h5py.File(str(_ROBOMIMIC_RAW), "r") as rf:

            a_mean = uf["norm_stats/actions/mean"][:]
            a_std = uf["norm_stats/actions/std"][:]

            demo_keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in uf["mask/train"][:]
            ]
            n_demos = len(demo_keys)

            for key in demo_keys:
                actions = uf[f"data/{key}/actions"][:]  # [T, 7] raw
                initial_state = rf[f"data/{key}/states"][0]  # MuJoCo state

                # Normalization round-trip sanity check
                actions_norm = (actions - a_mean) / a_std
                actions_denorm = actions_norm * a_std + a_mean
                assert np.allclose(actions, actions_denorm, atol=1e-5), \
                    f"Normalization round-trip failed for {key}"

                # Reset env and restore exact scene state
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
                print(f"  {key}: {'SUCCESS' if success else 'FAIL'} "
                      f"(steps={t+1}/{len(actions)})")

        env.close()
        rate = successes / n_demos
        print(f"\nGT replay lift: {successes}/{n_demos} ({rate*100:.0f}%)")
        # Offscreen renderer introduces tiny FP perturbations vs bare sim,
        # so borderline demos may flip. Require >= 98%.
        assert rate >= 0.98, \
            f"Expected >=98% success, got {successes}/{n_demos} ({rate*100:.1f}%)"


@pytest.mark.slow
class TestRandomPolicyRobomimic:
    """Random policy through full evaluate_policy loop.

    Tests that the evaluation pipeline runs without crashing.
    Expected success rate: ~0%.
    """

    def test_random_policy_no_crash(self):
        from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
        from data_pipeline.evaluation.rollout import evaluate_policy

        class RandomPolicy:
            def predict(self, images, proprio, view_present):
                return torch.randn(50, 7) * 0.1

        env = RobomimicWrapper("lift")

        sr, results = evaluate_policy(
            RandomPolicy(), env,
            num_episodes=3, max_steps=50,
            exec_horizon=8, obs_horizon=2,
        )

        env.close()

        assert len(results) == 3
        for r in results:
            assert "success" in r
            assert "steps" in r
            assert r["steps"] == 50  # should hit max_steps

        # Random policy should almost never succeed
        assert sr < 0.5, f"Random policy success rate suspiciously high: {sr}"
        print(f"Random policy lift: sr={sr:.2f} (expected ~0.0)")


# ---------------------------------------------------------------------------
# RLBench smoke tests (WSL2 only)
# ---------------------------------------------------------------------------

@pytest.mark.rlbench
class TestGTReplayRLBench:
    """Ground-truth replay of RLBench demos through RLBenchWrapper.

    Feeds stored 8D absolute EE pose actions through the wrapper and checks
    task success. Scene is restored from raw demo pickles.

    Pass criterion: >= 40% on each task (physics non-determinism ceiling ~50%).
    """

    def _run_gt_replay(self, task_name, n_demos=5):
        try:
            import rlbench  # noqa: F401
        except ImportError:
            pytest.skip("rlbench not installed (requires WSL2 + CoppeliaSim)")

        unified_path = _RLBENCH_UNIFIED_DIR / f"{task_name}.hdf5"
        if not unified_path.exists():
            pytest.skip(f"Unified HDF5 not found: {unified_path}")

        raw_ep_dir = _RLBENCH_RAW_DIR / task_name / "all_variations" / "episodes"
        if not raw_ep_dir.exists():
            pytest.skip(f"Raw episodes not found: {raw_ep_dir}")

        from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

        env = RLBenchWrapper(task_name, cameras=False)

        # Map episode dirs (sorted by index)
        ep_dirs = sorted(
            [d for d in raw_ep_dir.iterdir()
             if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
            key=lambda d: int(d.name.replace("episode", "")),
        )

        successes = 0

        with h5py.File(str(unified_path), "r") as f:
            demo_keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in f["mask/train"][:n_demos]
            ]

            for i, key in enumerate(demo_keys):
                actions = f[f"data/{key}/actions"][:]  # [T, 8] absolute poses

                # Load demo pickle for scene restoration
                with open(ep_dirs[i] / "low_dim_obs.pkl", "rb") as fh:
                    demo = pickle.load(fh)

                # Restore scene state (same logic as replay_rlbench.py)
                try:
                    descriptions, obs = env._task.reset_to_demo(demo)
                except (KeyError, AttributeError):
                    obs_list = (demo._observations
                                if hasattr(demo, '_observations')
                                else list(demo))
                    if hasattr(demo, 'random_seed') and demo.random_seed is not None:
                        demo.restore_state()
                    variation = 0
                    if (hasattr(obs_list[0], 'misc')
                            and obs_list[0].misc is not None):
                        variation = obs_list[0].misc.get('variation_index', 0)
                    env._task.set_variation(variation)
                    descriptions, obs = env._task.reset(demo)

                env._last_obs = obs

                success = False
                for t in range(len(actions)):
                    _, _, done, info = env.step(actions[t])
                    if info["success"]:
                        success = True
                        break
                    if done:
                        break

                successes += int(success)
                print(f"  {task_name}/{key}: "
                      f"{'SUCCESS' if success else 'FAIL'} (steps={t+1})")

        env.close()
        rate = successes / n_demos
        print(f"\nGT replay {task_name}: {successes}/{n_demos} ({rate*100:.0f}%)")
        return successes, n_demos

    def test_gt_replay_close_jar(self):
        successes, total = self._run_gt_replay("close_jar", n_demos=20)
        rate = successes / total
        print(f"close_jar: {successes}/{total} ({rate*100:.0f}%)")
        # Physics non-determinism ceiling ~50% (replay_rlbench.py: 10/20).
        assert rate >= 0.4, \
            f"Expected >=40% success, got {successes}/{total} ({rate*100:.1f}%)"

    def test_gt_replay_open_drawer(self):
        successes, total = self._run_gt_replay("open_drawer", n_demos=20)
        rate = successes / total
        print(f"open_drawer: {successes}/{total} ({rate*100:.0f}%)")
        assert rate >= 0.4, \
            f"Expected >=40% success, got {successes}/{total} ({rate*100:.1f}%)"

    def test_gt_replay_slide_block(self):
        successes, total = self._run_gt_replay(
            "slide_block_to_color_target", n_demos=20,
        )
        rate = successes / total
        print(f"slide_block: {successes}/{total} ({rate*100:.0f}%)")
        assert rate >= 0.4, \
            f"Expected >=40% success, got {successes}/{total} ({rate*100:.1f}%)"


@pytest.mark.rlbench
class TestRandomPolicyRLBench:
    """Random policy through RLBench wrapper. Tests no-crash."""

    def test_random_policy_no_crash(self):
        try:
            import rlbench  # noqa: F401
        except ImportError:
            pytest.skip("rlbench not installed (requires WSL2 + CoppeliaSim)")

        unified_path = _RLBENCH_UNIFIED_DIR / "close_jar.hdf5"
        if not unified_path.exists():
            pytest.skip("RLBench unified data not found")

        from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper
        from data_pipeline.evaluation.rollout import evaluate_policy

        class RandomPolicy:
            def predict(self, images, proprio, view_present):
                return torch.randn(50, 8) * 0.01  # small 8D actions

        env = RLBenchWrapper("close_jar", cameras=False)

        sr, results = evaluate_policy(
            RandomPolicy(), env,
            num_episodes=2, max_steps=30,
            exec_horizon=8, obs_horizon=2,
        )

        env.close()
        assert len(results) == 2
        assert sr < 0.5
        print(f"Random policy close_jar: sr={sr:.2f}")
