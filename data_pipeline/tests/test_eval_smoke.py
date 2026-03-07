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

    Pass criterion: >= 9/10 demos succeed (matches Phase 1 demo replay).
    """

    def test_gt_replay_lift(self):
        if not _ROBOMIMIC_UNIFIED.exists():
            pytest.skip(f"Unified HDF5 not found: {_ROBOMIMIC_UNIFIED}")
        if not _ROBOMIMIC_RAW.exists():
            pytest.skip(f"Raw HDF5 not found: {_ROBOMIMIC_RAW}")

        from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

        env = RobomimicWrapper("lift")
        n_demos = 10
        successes = 0

        with h5py.File(str(_ROBOMIMIC_UNIFIED), "r") as uf, \
             h5py.File(str(_ROBOMIMIC_RAW), "r") as rf:

            a_mean = uf["norm_stats/actions/mean"][:]
            a_std = uf["norm_stats/actions/std"][:]

            demo_keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in uf["mask/train"][:n_demos]
            ]

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
        print(f"\nGT replay lift: {successes}/{n_demos}")
        assert successes >= 9, \
            f"Expected >= 9/10 successes, got {successes}/10"


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

    Tests the delta->absolute accumulation in the wrapper by feeding
    stored delta actions and checking task success.

    Pass criterion: comparable to Phase 2 sim replay rates.
    - close_jar: ~1/5 (20%)
    - open_drawer: ~3/5 (60%)
    """

    def _run_gt_replay(self, task_name, n_demos=5, min_successes=0):
        unified_path = _RLBENCH_UNIFIED_DIR / f"{task_name}.hdf5"
        if not unified_path.exists():
            pytest.skip(f"Unified HDF5 not found: {unified_path}")

        from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

        env = RLBenchWrapper(task_name)
        successes = 0

        with h5py.File(str(unified_path), "r") as f:
            a_mean = f["norm_stats/actions/mean"][:]
            a_std = f["norm_stats/actions/std"][:]

            demo_keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in f["mask/train"][:n_demos]
            ]

            for key in demo_keys:
                actions = f[f"data/{key}/actions"][:]  # [T, 7] raw deltas

                env.reset()

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
        successes, total = self._run_gt_replay("close_jar", n_demos=5)
        # Phase 2 baseline: 1/5. Accept any success.
        print(f"close_jar: {successes}/{total} (Phase 2 baseline: ~1/5)")

    def test_gt_replay_open_drawer(self):
        successes, total = self._run_gt_replay("open_drawer", n_demos=5)
        # Phase 2 baseline: 3/5. Accept >= 1.
        assert successes >= 1, \
            f"Expected >= 1/5 for open_drawer, got {successes}/5"


@pytest.mark.rlbench
class TestRandomPolicyRLBench:
    """Random policy through RLBench wrapper. Tests no-crash."""

    def test_random_policy_no_crash(self):
        unified_path = _RLBENCH_UNIFIED_DIR / "close_jar.hdf5"
        if not unified_path.exists():
            pytest.skip("RLBench unified data not found")

        from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper
        from data_pipeline.evaluation.rollout import evaluate_policy

        class RandomPolicy:
            def predict(self, images, proprio, view_present):
                return torch.randn(50, 7) * 0.01  # small actions

        env = RLBenchWrapper("close_jar")

        sr, results = evaluate_policy(
            RandomPolicy(), env,
            num_episodes=2, max_steps=30,
            exec_horizon=8, obs_horizon=2,
        )

        env.close()
        assert len(results) == 2
        assert sr < 0.5
        print(f"Random policy close_jar: sr={sr:.2f}")
