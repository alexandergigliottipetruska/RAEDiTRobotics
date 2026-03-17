"""Tests for C.8 Stage3PolicyWrapper.

Verifies the eval wrapper correctly interfaces with the rollout harness.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from data_pipeline.evaluation.stage3_eval import Stage3PolicyWrapper
from models.ema import EMA
from models.policy_dit import PolicyDiT
from models.stage1_bridge import Stage1Bridge


# Test constants
T_O = 2
T_P = 16
K = 4
H, W = 224, 224
AC_DIM = 7
D_PROP = 9


def _make_policy():
    """Create a PolicyDiT with mock encoder."""
    bridge = Stage1Bridge(pretrained_encoder=False)
    return PolicyDiT(
        bridge=bridge,
        ac_dim=AC_DIM,
        proprio_dim=D_PROP,
        hidden_dim=256,
        T_obs=T_O,
        T_pred=T_P,
        num_blocks=2,
        nhead=8,
        num_views=K,
        train_diffusion_steps=100,
        eval_diffusion_steps=5,
        p_view_drop=0.0,
        policy_type="ddpm",
    )


def _make_wrapper(with_ema=False):
    """Create a Stage3PolicyWrapper."""
    policy = _make_policy()
    ema = None
    if with_ema:
        ema = EMA(policy, decay=0.999, warmup_steps=0)
    return Stage3PolicyWrapper(policy, ema=ema, device="cpu")


# ============================================================
# Basic interface tests
# ============================================================

class TestWrapperInterface:
    def test_predict_returns_tensor(self):
        """predict() returns a torch.Tensor."""
        wrapper = _make_wrapper()
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert isinstance(actions, torch.Tensor)

    def test_predict_output_shape(self):
        """predict() returns (T_p, ac_dim) — no batch dim."""
        wrapper = _make_wrapper()
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)

    def test_predict_output_finite(self):
        """Predicted actions are finite."""
        wrapper = _make_wrapper()
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert torch.isfinite(actions).all()


# ============================================================
# Image preprocessing tests
# ============================================================

class TestImagePreprocessing:
    def test_accepts_uint8_images(self):
        """Works with uint8 images (as provided by envs)."""
        wrapper = _make_wrapper()
        images = torch.randint(0, 256, (1, T_O, K, 3, H, W), dtype=torch.uint8)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)

    def test_accepts_float_images(self):
        """Works with float32 images already in [0,1]."""
        wrapper = _make_wrapper()
        images = torch.rand(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)


# ============================================================
# EMA tests
# ============================================================

class TestEMAIntegration:
    def test_works_with_ema(self):
        """predict() works with EMA enabled."""
        wrapper = _make_wrapper(with_ema=True)
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)
        assert torch.isfinite(actions).all()

    def test_works_without_ema(self):
        """predict() works without EMA."""
        wrapper = _make_wrapper(with_ema=False)
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)

    def test_ema_weights_restored_after_predict(self):
        """After predict(), model weights are restored (not stuck on EMA)."""
        policy = _make_policy()
        ema = EMA(policy, decay=0.999, warmup_steps=0)
        # Diverge model from EMA
        with torch.no_grad():
            for p in policy.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.1)
        original_state = {k: v.clone() for k, v in policy.state_dict().items()}

        wrapper = Stage3PolicyWrapper(policy, ema=ema, device="cpu")
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        wrapper.predict(images, proprio, vp)

        # Weights should be restored
        for k, v in policy.state_dict().items():
            assert torch.equal(v, original_state[k]), f"Weight {k} not restored after EMA"


# ============================================================
# Partial views
# ============================================================

class TestPartialViews:
    def test_partial_views(self):
        """Works when some cameras are absent."""
        wrapper = _make_wrapper()
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.tensor([[True, True, False, False]])
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)
        assert torch.isfinite(actions).all()

    def test_single_view(self):
        """Works with only one camera."""
        wrapper = _make_wrapper()
        images = torch.randn(1, T_O, K, 3, H, W)
        proprio = torch.randn(1, T_O, D_PROP)
        vp = torch.tensor([[True, False, False, False]])
        actions = wrapper.predict(images, proprio, vp)
        assert actions.shape == (T_P, AC_DIM)
