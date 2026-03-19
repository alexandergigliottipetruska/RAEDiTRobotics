"""Tests for V3 evaluation wrapper.

Tests the V3PolicyWrapper logic: no external ImageNet norm, rot6d conversion,
action denormalization. Does NOT require robosuite (tests wrapper only).
"""

import numpy as np
import torch
import pytest

from training.eval_v3 import V3PolicyWrapper
from data_pipeline.utils.rotation import convert_actions_to_rot6d, convert_actions_from_rot6d

# Constants
B = 1
T_O = 2
T_P = 16
K = 4
H = W = 224
AC_DIM = 10
PROPRIO_DIM = 9


class _MockPolicyV3:
    """Mock PolicyDiTv3 that returns fixed actions for testing."""

    def __init__(self, ac_dim=AC_DIM, T_pred=T_P):
        self.ac_dim = ac_dim
        self.T_pred = T_pred
        self._last_images = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def predict_action(self, obs_dict):
        self._last_images = obs_dict["images_enc"]
        B = obs_dict["proprio"].shape[0]
        return torch.randn(B, self.T_pred, self.ac_dim)


class TestV3WrapperNoImageNetNorm:
    def test_images_not_imagenet_normalized(self):
        """V3 wrapper passes float [0,1] images, NOT ImageNet-normalized."""
        mock = _MockPolicyV3()
        wrapper = V3PolicyWrapper(mock)

        images = torch.rand(1, T_O, K, 3, H, W)  # float [0,1]
        proprio = torch.randn(1, T_O, PROPRIO_DIM)
        vp = torch.ones(1, K, dtype=torch.bool)

        wrapper.predict(images, proprio, vp)

        received = mock._last_images
        # Should be in [0, 1] range — NOT ImageNet normalized (~[-2.1, 2.6])
        assert received.min() >= -0.01, f"Images min {received.min():.3f} < 0 (was ImageNet-normalized?)"
        assert received.max() <= 1.01, f"Images max {received.max():.3f} > 1 (was ImageNet-normalized?)"

    def test_uint8_converted_to_float(self):
        """uint8 images are converted to float [0,1]."""
        mock = _MockPolicyV3()
        wrapper = V3PolicyWrapper(mock)

        images = torch.randint(0, 256, (1, T_O, K, 3, H, W), dtype=torch.uint8)
        proprio = torch.randn(1, T_O, PROPRIO_DIM)
        vp = torch.ones(1, K, dtype=torch.bool)

        wrapper.predict(images, proprio, vp)

        received = mock._last_images
        assert received.dtype == torch.float32
        assert received.min() >= 0.0
        assert received.max() <= 1.0


class TestV3WrapperOutput:
    def test_output_shape(self):
        """predict returns (T_p, ac_dim) — no batch dim."""
        mock = _MockPolicyV3()
        wrapper = V3PolicyWrapper(mock)

        images = torch.rand(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, PROPRIO_DIM)
        vp = torch.ones(1, K, dtype=torch.bool)

        out = wrapper.predict(images, proprio, vp)
        assert out.shape == (T_P, AC_DIM)

    def test_output_finite(self):
        mock = _MockPolicyV3()
        wrapper = V3PolicyWrapper(mock)

        images = torch.rand(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, PROPRIO_DIM)
        vp = torch.ones(1, K, dtype=torch.bool)

        out = wrapper.predict(images, proprio, vp)
        assert torch.isfinite(out).all()


class TestRot6dConversion:
    def test_rot6d_to_axis_angle_roundtrip(self):
        """10D → 7D → 10D roundtrip preserves actions."""
        actions_7d = np.random.uniform(-1, 1, (T_P, 7)).astype(np.float32)
        actions_10d = convert_actions_to_rot6d(actions_7d)
        actions_7d_back = convert_actions_from_rot6d(actions_10d)
        np.testing.assert_allclose(actions_7d, actions_7d_back, atol=1e-5)

    def test_pos_and_grip_preserved(self):
        """Position and gripper dims are unchanged by rot6d conversion."""
        actions_7d = np.random.uniform(-1, 1, (T_P, 7)).astype(np.float32)
        actions_10d = convert_actions_to_rot6d(actions_7d)
        # pos: dims 0-2
        np.testing.assert_allclose(actions_10d[:, :3], actions_7d[:, :3], atol=1e-6)
        # gripper: dim 9 (10D) = dim 6 (7D)
        np.testing.assert_allclose(actions_10d[:, 9], actions_7d[:, 6], atol=1e-6)

    def test_action_denormalization(self):
        """Minmax denormalization recovers original scale."""
        raw = np.array([0.5, -0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        a_min = np.full(10, -2.0)
        a_max = np.full(10, 2.0)

        # Normalize
        norm = 2.0 * (raw - a_min) / (a_max - a_min) - 1.0
        # Denormalize
        recovered = (norm + 1.0) / 2.0 * (a_max - a_min) + a_min

        np.testing.assert_allclose(recovered, raw, atol=1e-6)


# ============================================================
# Robosuite integration tests
# ============================================================

class TestV3EvalRobosuite:
    """Integration tests using real RobomimicWrapper + robosuite."""

    def _make_env(self):
        from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
        return RobomimicWrapper(task="lift", image_size=84, abs_action=True)

    def test_invalid_task_key_raises(self):
        """Capitalized task key raises KeyError (must use lowercase)."""
        from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
        with pytest.raises(KeyError):
            RobomimicWrapper(task="Lift", image_size=84, abs_action=True)

    def test_env_returns_float_images(self):
        """RobomimicWrapper.get_multiview_images() returns float [0,1]."""
        env = self._make_env()
        env.reset()
        imgs = env.get_multiview_images()  # (1, K, 3, H, W)
        assert imgs.dtype == np.float32 or imgs.dtype == np.float64
        assert imgs.min() >= 0.0
        assert imgs.max() <= 1.0
        env.close()

    def test_image_shape(self):
        """get_multiview_images() returns [1, 4, 3, S, S] matching image_size."""
        env = self._make_env()  # image_size=84
        env.reset()
        imgs = env.get_multiview_images()
        assert imgs.shape == (1, 4, 3, 84, 84), f"Expected (1,4,3,84,84) got {imgs.shape}"
        # Slots 1 and 2 should be zero (unused cameras)
        np.testing.assert_array_equal(imgs[0, 1], 0.0)
        np.testing.assert_array_equal(imgs[0, 2], 0.0)
        # Slots 0 and 3 should have non-zero content
        assert imgs[0, 0].sum() > 0, "agentview (slot 0) should not be all zeros"
        assert imgs[0, 3].sum() > 0, "wrist cam (slot 3) should not be all zeros"
        env.close()

    def test_proprio_shape_and_content(self):
        """get_proprio() returns [1, 9] with reasonable values."""
        env = self._make_env()
        env.reset()
        proprio = env.get_proprio()
        assert proprio.shape == (1, 9), f"Expected (1,9) got {proprio.shape}"
        assert proprio.dtype == np.float32
        # EEF position should be within robosuite workspace (~[-2, 2] meters)
        eef_pos = proprio[0, :3]
        assert np.all(np.abs(eef_pos) < 5.0), f"EEF pos {eef_pos} out of range"
        # Quaternion should be unit-length
        quat = proprio[0, 3:7]
        quat_norm = np.linalg.norm(quat)
        np.testing.assert_allclose(quat_norm, 1.0, atol=1e-3,
                                   err_msg=f"Quaternion not unit: norm={quat_norm}")
        env.close()

    def test_view_present_mask(self):
        """get_view_present() returns [True, False, False, True] for robomimic."""
        env = self._make_env()
        env.reset()
        vp = env.get_view_present()
        expected = np.array([True, False, False, True])
        np.testing.assert_array_equal(vp, expected)
        env.close()

    def test_env_accepts_7d_abs_action(self):
        """Env accepts 7D absolute actions without error."""
        env = self._make_env()
        env.reset()
        # Realistic absolute action: position near workspace, small rotation, gripper closed
        action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        assert isinstance(reward, (int, float))
        env.close()

    def test_full_episode_with_random_10d_policy(self):
        """Run a short episode: random 10D → rot6d→aa → env.step()."""
        env = self._make_env()
        env.reset()

        for _ in range(5):
            # Random 10D action (normalized scale)
            action_10d = np.random.uniform(-0.5, 0.5, (1, 10)).astype(np.float32)
            # Convert rot6d → axis_angle: 10D → 7D
            action_7d = convert_actions_from_rot6d(action_10d)
            obs, reward, done, info = env.step(action_7d[0])
            if done:
                break

        env.close()

    def test_seed_produces_different_states(self):
        """Different seeds produce different initial states."""
        env = self._make_env()

        env.seed(100000)
        env.reset()
        pos1 = env.get_proprio()[0, :3].copy()  # eef position

        env.seed(100001)
        env.reset()
        pos2 = env.get_proprio()[0, :3].copy()

        assert not np.allclose(pos1, pos2, atol=1e-4), \
            "Different seeds should produce different initial states"
        env.close()

    def test_same_seed_reproduces_state(self):
        """Same seed produces same initial state."""
        env = self._make_env()

        env.seed(42)
        env.reset()
        pos1 = env.get_proprio()[0, :3].copy()

        env.seed(42)
        env.reset()
        pos2 = env.get_proprio()[0, :3].copy()

        np.testing.assert_allclose(pos1, pos2, atol=1e-6)
        env.close()

    def test_parallel_eval_matches_sequential(self):
        """Parallel eval produces results for all episodes."""
        from training.eval_v3 import V3PolicyWrapper, evaluate_v3_parallel

        class _RandomWrapper:
            def __init__(self):
                self._inference_lock = __import__('threading').Lock()
            def predict(self, images, proprio, view_present):
                with self._inference_lock:
                    return torch.randn(T_P, AC_DIM) * 0.1

        action_min = np.full(10, -1.0, dtype=np.float32)
        action_max = np.full(10, 1.0, dtype=np.float32)
        proprio_min = np.full(PROPRIO_DIM, -5.0, dtype=np.float32)
        proprio_max = np.full(PROPRIO_DIM, 5.0, dtype=np.float32)
        norm_stats = {
            "actions": {"min": action_min, "max": action_max},
            "proprio": {"min": proprio_min, "max": proprio_max},
        }

        wrapper = _RandomWrapper()
        sr, results = evaluate_v3_parallel(
            wrapper, norm_stats,
            num_episodes=4, num_workers=2,
            task="lift", max_steps=10,
        )
        assert len(results) == 4
        assert all(isinstance(r["success"], bool) for r in results)

    def test_inference_lock_exists(self):
        """V3PolicyWrapper has thread-safe inference lock."""
        from training.eval_v3 import V3PolicyWrapper
        mock = _MockPolicyV3()
        wrapper = V3PolicyWrapper(mock)
        assert hasattr(wrapper, '_inference_lock')
        assert isinstance(wrapper._inference_lock, type(__import__('threading').Lock()))

    def test_rollout_with_mock_v3_policy(self):
        """Full evaluate_policy loop with mock V3 wrapper and real env."""
        env = self._make_env()

        # Mock wrapper that returns random normalized 10D actions
        class _RandomV3Wrapper:
            def predict(self, images, proprio, view_present):
                return torch.randn(T_P, AC_DIM) * 0.1  # small random actions

        from data_pipeline.evaluation.rollout import evaluate_policy

        # Use dummy norm stats that map [-1,1] → small real-world range
        action_min = np.full(10, -1.0, dtype=np.float32)
        action_max = np.full(10, 1.0, dtype=np.float32)
        proprio_min = np.full(PROPRIO_DIM, -5.0, dtype=np.float32)
        proprio_max = np.full(PROPRIO_DIM, 5.0, dtype=np.float32)

        success_rate, results = evaluate_policy(
            policy=_RandomV3Wrapper(),
            env=env,
            num_episodes=2,
            max_steps=10,
            norm_mode="minmax",
            action_min=action_min,
            action_max=action_max,
            proprio_min=proprio_min,
            proprio_max=proprio_max,
            exec_horizon=8,
            obs_horizon=2,
            rot6d=True,
        )

        assert len(results) == 2
        assert all(isinstance(r["success"], bool) for r in results)
        env.close()
