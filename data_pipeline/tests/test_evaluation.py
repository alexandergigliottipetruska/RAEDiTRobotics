"""Tier 1 unit tests for evaluation harness components.

Tests metrics (Wilson CI, success rate, aggregate_seeds) and
verifies the base_env abstract interface contract.
All local, no simulator needed.
"""

import numpy as np
import pytest

from data_pipeline.evaluation.metrics import (
    aggregate_seeds,
    success_rate,
    wilson_ci,
)
from data_pipeline.envs.base_env import BaseManipulationEnv


# ---------------------------------------------------------------------------
# success_rate
# ---------------------------------------------------------------------------

class TestSuccessRate:
    def test_all_success(self):
        assert success_rate([True, True, True]) == 1.0

    def test_all_failure(self):
        assert success_rate([False, False, False]) == 0.0

    def test_mixed(self):
        assert success_rate([True, False, True, False]) == 0.5

    def test_empty(self):
        assert success_rate([]) == 0.0

    def test_numpy_input(self):
        arr = np.array([True, True, False, False, True])
        assert success_rate(arr) == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

class TestWilsonCI:
    def test_perfect_success(self):
        lo, hi = wilson_ci(10, 10)
        # 10/10 → CI should have lower bound around 0.69
        assert lo >= 0.5
        assert hi == pytest.approx(1.0, abs=0.01)

    def test_perfect_failure(self):
        lo, hi = wilson_ci(0, 10)
        # 0/10 → CI should have upper bound around 0.31
        assert lo == pytest.approx(0.0, abs=0.01)
        assert hi <= 0.5

    def test_half(self):
        lo, hi = wilson_ci(5, 10)
        # 5/10 → symmetric around 0.5
        assert lo < 0.5
        assert hi > 0.5
        assert pytest.approx(0.5, abs=0.05) == (lo + hi) / 2

    def test_empty(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0
        assert hi == 0.0

    def test_ci_contains_point_estimate(self):
        lo, hi = wilson_ci(7, 20)
        p_hat = 7 / 20
        assert lo <= p_hat <= hi

    def test_larger_sample_tighter_ci(self):
        _, hi_small = wilson_ci(5, 10)
        lo_small, _ = wilson_ci(5, 10)
        lo_large, hi_large = wilson_ci(50, 100)
        # Same proportion but larger sample → tighter interval
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_ci_bounds_valid(self):
        for n_success in range(11):
            lo, hi = wilson_ci(n_success, 10)
            assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# aggregate_seeds
# ---------------------------------------------------------------------------

class TestAggregateSeeds:
    def test_single_seed(self):
        result = aggregate_seeds([[True, True, False]])
        assert result["success_rate"] == pytest.approx(2 / 3)
        assert result["n_successes"] == 2
        assert result["n_total"] == 3
        assert len(result["per_seed_rates"]) == 1

    def test_multiple_seeds(self):
        outcomes = [
            [True, True, True],   # 1.0
            [False, False, False], # 0.0
            [True, False, True],   # 0.667
        ]
        result = aggregate_seeds(outcomes)
        assert result["n_total"] == 9
        assert result["n_successes"] == 5
        assert result["success_rate"] == pytest.approx(5 / 9)
        assert result["ci_lower"] <= result["success_rate"]
        assert result["ci_upper"] >= result["success_rate"]
        assert len(result["per_seed_rates"]) == 3

    def test_empty_seeds(self):
        result = aggregate_seeds([])
        assert result["success_rate"] == 0.0
        assert result["n_total"] == 0


# ---------------------------------------------------------------------------
# BaseManipulationEnv interface
# ---------------------------------------------------------------------------

class TestBaseEnvInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseManipulationEnv()

    def test_concrete_subclass_works(self):
        """A minimal concrete implementation should be instantiable."""
        class DummyEnv(BaseManipulationEnv):
            def reset(self): return {}
            def step(self, action): return {}, 0.0, False, {"success": False}
            def get_multiview_images(self):
                return np.zeros((1, 4, 3, 224, 224), dtype=np.float32)
            def get_proprio(self):
                return np.zeros((1, 9), dtype=np.float32)
            def get_view_present(self):
                return np.array([True, False, False, True])
            def close(self): pass
            @property
            def proprio_dim(self): return 9
            @property
            def num_cameras(self): return 4

        env = DummyEnv()
        assert env.proprio_dim == 9
        assert env.num_cameras == 4
        assert env.get_multiview_images().shape == (1, 4, 3, 224, 224)
        assert env.get_proprio().shape == (1, 9)
        assert env.get_view_present().shape == (4,)
        obs, reward, done, info = env.step(np.zeros(7))
        assert "success" in info
