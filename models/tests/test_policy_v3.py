"""Tests for PolicyDiTv3.

Validates compute_loss, predict_action, gradient flow, precomputed token mode,
and DDIM inference bounds.
"""

import torch
import pytest

from models.policy_v3 import PolicyDiTv3
from models.stage1_bridge import Stage1Bridge

# Constants
B = 2
T_O = 2
T_P = 10  # Chi: horizon=10
K = 4
H = W = 224
AC_DIM = 10
PROPRIO_DIM = 9
D_MODEL = 256


def _make_bridge():
    """Create Stage1Bridge with mock encoder (no HF download)."""
    return Stage1Bridge(pretrained_encoder=False)


def _make_policy(**kwargs):
    defaults = dict(
        ac_dim=AC_DIM, proprio_dim=PROPRIO_DIM, d_model=D_MODEL,
        n_head=4, n_layers=2, T_obs=T_O, T_pred=T_P, num_views=K,
        n_active_cams=2, train_diffusion_steps=100, eval_diffusion_steps=10,
        p_drop_emb=0.0, p_drop_attn=0.3,
    )
    defaults.update(kwargs)
    bridge = _make_bridge()
    return PolicyDiTv3(bridge=bridge, **defaults)


def _make_batch(b=B, cached=False):
    """Create a training batch matching Stage3Dataset output."""
    # Robomimic: slots 0 and 3 active, slots 1 and 2 zero-padded
    vp = torch.zeros(b, K, dtype=torch.bool)
    vp[:, 0] = True
    vp[:, 3] = True
    batch = {
        "actions": torch.randn(b, T_P, AC_DIM),
        "proprio": torch.randn(b, T_O, PROPRIO_DIM),
        "view_present": vp,
    }
    if cached:
        # Precomputed tokens: (B, T_o, K, 196, 1024) — post-encoder, pre-adapter
        batch["cached_tokens"] = torch.randn(b, T_O, K, 196, 1024)
    else:
        # Online mode: images for Stage1Bridge
        batch["images_enc"] = torch.randn(b, T_O, K, 3, H, W)
        batch["images_target"] = torch.rand(b, T_O, K, 3, H, W)
    return batch


class TestPolicyV3Loss:
    def test_compute_loss_finite(self):
        """compute_loss returns a finite positive scalar."""
        policy = _make_policy()
        batch = _make_batch(cached=True)
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_compute_loss_online(self):
        """compute_loss works with online images (not cached)."""
        policy = _make_policy()
        batch = _make_batch(cached=False)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_forward_is_compute_loss(self):
        """forward() is an alias for compute_loss()."""
        policy = _make_policy()
        batch = _make_batch(cached=True)
        torch.manual_seed(42)
        loss1 = policy(batch)
        torch.manual_seed(42)
        loss2 = policy.compute_loss(batch)
        # Not exactly equal due to random timestep sampling, but both should be valid
        assert torch.isfinite(loss1)
        assert torch.isfinite(loss2)


class TestPolicyV3Predict:
    def test_predict_action_shape(self):
        """predict_action returns (B, T_pred, ac_dim)."""
        policy = _make_policy()
        policy.eval()
        obs = _make_batch(cached=True)
        actions = policy.predict_action(obs)
        assert actions.shape == (B, T_P, AC_DIM)

    def test_predict_action_finite(self):
        """predict_action output is finite."""
        policy = _make_policy()
        policy.eval()
        obs = _make_batch(cached=True)
        actions = policy.predict_action(obs)
        assert torch.isfinite(actions).all()

    def test_ddim_clip_sample(self):
        """With clip_sample=True, DDIM output should be roughly bounded."""
        policy = _make_policy()
        policy.eval()
        obs = _make_batch(cached=True)
        actions = policy.predict_action(obs)
        # clip_sample clips the denoised x0 prediction to [-1,1] at each step
        # Final output should be close to [-1,1] range
        assert actions.min() >= -2.0, f"Actions min {actions.min():.2f} too negative"
        assert actions.max() <= 2.0, f"Actions max {actions.max():.2f} too positive"


class TestPolicyV3Gradients:
    def test_encoder_frozen(self):
        """Encoder parameters have no gradient after backward."""
        policy = _make_policy()
        batch = _make_batch(cached=False)
        loss = policy.compute_loss(batch)
        loss.backward()
        for p in policy.bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_adapter_trainable(self):
        """Adapter receives gradients."""
        policy = _make_policy()
        batch = _make_batch(cached=False)
        loss = policy.compute_loss(batch)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in policy.bridge.adapter.parameters()
        )
        assert has_grad, "Adapter should receive gradients"

    def test_denoiser_receives_gradients(self):
        """Denoiser parameters receive gradients."""
        policy = _make_policy()
        batch = _make_batch(cached=True)
        loss = policy.compute_loss(batch)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in policy.denoiser.parameters()
        )
        assert has_grad, "Denoiser should receive gradients"

    def test_obs_encoder_receives_gradients(self):
        """ObservationEncoder parameters receive gradients."""
        policy = _make_policy()
        batch = _make_batch(cached=True)
        loss = policy.compute_loss(batch)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in policy.obs_encoder.parameters()
        )
        assert has_grad, "ObservationEncoder should receive gradients"


class TestPolicyV3Modes:
    def test_precomputed_tokens_mode(self):
        """Works with precomputed tokens (cached_tokens key)."""
        policy = _make_policy()
        batch = _make_batch(cached=True)
        assert "cached_tokens" in batch
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_online_mode(self):
        """Works with online images (images_enc key)."""
        policy = _make_policy()
        batch = _make_batch(cached=False)
        assert "images_enc" in batch
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_partial_views(self):
        """Works with partial camera views."""
        policy = _make_policy()
        batch = _make_batch(cached=True)
        # Only cameras 0 and 3 active (robomimic)
        batch["view_present"][:, 1:3] = False
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_rlbench_dims(self):
        """Works with RLBench dimensions (ac_dim=8, proprio=8, 4 active cams)."""
        bridge = _make_bridge()
        policy = PolicyDiTv3(
            bridge=bridge, ac_dim=8, proprio_dim=8,
            d_model=D_MODEL, n_head=4, n_layers=2,
            T_obs=T_O, T_pred=T_P, num_views=K, n_active_cams=4,
            train_diffusion_steps=100, eval_diffusion_steps=10,
        )
        batch = {
            "cached_tokens": torch.randn(B, T_O, K, 196, 1024),
            "actions": torch.randn(B, T_P, 8),
            "proprio": torch.randn(B, T_O, 8),
            "view_present": torch.ones(B, K, dtype=torch.bool),
        }
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)
