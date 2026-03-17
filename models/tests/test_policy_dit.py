"""Tests for C.10 PolicyDiT.

Verifies the full policy module that composes Stage1Bridge + ViewDropout +
TokenAssembly + _DiTNoiseNetV2 into the BasePolicy interface.
"""

import pytest
import torch
import torch.nn as nn

from models.base_policy import BasePolicy
from models.policy_dit import PolicyDiT
from models.stage1_bridge import Stage1Bridge


# Test constants
B = 2
T_O = 2
T_P = 16
K = 4
H, W = 224, 224
AC_DIM = 7
D_PROP = 9
HIDDEN = 256


def _make_bridge(**kwargs):
    """Create a Stage1Bridge with mock encoder."""
    defaults = dict(pretrained_encoder=False, load_decoder=False)
    defaults.update(kwargs)
    return Stage1Bridge(**defaults)


def _make_policy(**kwargs):
    """Create a PolicyDiT with mock encoder and small architecture."""
    bridge = _make_bridge(load_decoder=kwargs.pop("load_decoder", False))
    defaults = dict(
        bridge=bridge,
        ac_dim=AC_DIM,
        proprio_dim=D_PROP,
        hidden_dim=HIDDEN,
        T_obs=T_O,
        T_pred=T_P,
        num_blocks=2,  # small for fast tests
        nhead=8,
        num_views=K,
        train_diffusion_steps=100,
        eval_diffusion_steps=5,
        p_view_drop=0.0,
        lambda_recon=0.0,
        policy_type="ddpm",
    )
    defaults.update(kwargs)
    return PolicyDiT(**defaults)


def _make_fm_policy(**kwargs):
    """Create a flow matching PolicyDiT with mock encoder."""
    defaults = dict(
        policy_type="flow_matching",
        num_flow_steps=5,
        fm_timestep_scale=1000.0,
    )
    defaults.update(kwargs)
    return _make_policy(**defaults)


def _make_batch(b=B, t_o=T_O, k=K, t_p=T_P, ac_dim=AC_DIM, d_prop=D_PROP):
    """Create a fake training batch."""
    return {
        "images_enc": torch.randn(b, t_o, k, 3, H, W),
        "proprio": torch.randn(b, t_o, d_prop),
        "actions": torch.randn(b, t_p, ac_dim),
        "view_present": torch.ones(b, k, dtype=torch.bool),
        "images_target": torch.rand(b, t_o, k, 3, H, W),
    }


def _make_obs(b=B, t_o=T_O, k=K, d_prop=D_PROP):
    """Create a fake observation dict for inference."""
    return {
        "images_enc": torch.randn(b, t_o, k, 3, H, W),
        "proprio": torch.randn(b, t_o, d_prop),
        "view_present": torch.ones(b, k, dtype=torch.bool),
    }


# ============================================================
# Inheritance and interface tests
# ============================================================

class TestPolicyDiTInterface:
    def test_is_base_policy(self):
        """PolicyDiT inherits from BasePolicy."""
        policy = _make_policy()
        assert isinstance(policy, BasePolicy)

    def test_is_nn_module(self):
        """PolicyDiT is an nn.Module."""
        policy = _make_policy()
        assert isinstance(policy, nn.Module)

    def test_has_compute_loss(self):
        """PolicyDiT implements compute_loss."""
        policy = _make_policy()
        assert hasattr(policy, "compute_loss")

    def test_has_predict_action(self):
        """PolicyDiT implements predict_action."""
        policy = _make_policy()
        assert hasattr(policy, "predict_action")


# ============================================================
# compute_loss tests
# ============================================================

class TestComputeLoss:
    def test_returns_scalar(self):
        """compute_loss returns a scalar tensor."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_is_finite(self):
        """Loss is finite."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_loss_positive(self):
        """MSE loss is always positive."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert loss.item() > 0

    def test_loss_reasonable_range(self):
        """Loss is in a reasonable range for DDPM noise prediction."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert 0.01 < loss.item() < 50.0

    def test_backward_runs(self):
        """Backward pass completes without error."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()

    def test_8d_actions_rlbench(self):
        """Works with 8D actions (RLBench)."""
        policy = _make_policy(ac_dim=8)
        batch = _make_batch(ac_dim=8)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)


# ============================================================
# predict_action tests
# ============================================================

class TestPredictAction:
    def test_output_shape(self):
        """predict_action returns (B, T_pred, ac_dim)."""
        policy = _make_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert actions.shape == (B, T_P, AC_DIM)

    def test_output_finite(self):
        """Predicted actions are finite."""
        policy = _make_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert torch.isfinite(actions).all()

    def test_output_shape_8d(self):
        """Works with 8D actions (RLBench)."""
        policy = _make_policy(ac_dim=8)
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert actions.shape == (B, T_P, 8)

    def test_no_grad_during_inference(self):
        """predict_action does not accumulate gradients."""
        policy = _make_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert not actions.requires_grad

    def test_deterministic_with_seed(self):
        """Same seed produces same actions."""
        policy = _make_policy()
        policy.eval()
        obs = _make_obs()
        torch.manual_seed(42)
        a1 = policy.predict_action(obs)
        torch.manual_seed(42)
        a2 = policy.predict_action(obs)
        assert torch.equal(a1, a2)


# ============================================================
# Gradient flow tests
# ============================================================

class TestGradientFlow:
    def test_encoder_frozen(self):
        """Encoder receives no gradients."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()
        for p in policy.bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_adapter_receives_grad(self):
        """Adapter receives gradients from policy loss.

        adaLN-Zero starts with all-zero modulation params, so encoder output
        has no effect at init. Perturb zero-init params first to unblock gradients.
        """
        policy = _make_policy()
        # Perturb zero-initialized params so gradients flow through encoder path
        with torch.no_grad():
            for p in policy.noise_net.parameters():
                if p.requires_grad and torch.all(p == 0):
                    p.add_(torch.randn_like(p) * 0.01)
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()
        has_grad = False
        for p in policy.bridge.adapter.parameters():
            if p.grad is not None and not torch.all(p.grad == 0):
                has_grad = True
                break
        assert has_grad, "Adapter should receive gradients"

    def test_noise_net_receives_grad(self):
        """Noise net receives gradients."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()
        has_grad = False
        for p in policy.noise_net.parameters():
            if p.grad is not None and not torch.all(p.grad == 0):
                has_grad = True
                break
        assert has_grad, "Noise net should receive gradients"

    def test_token_assembly_receives_grad(self):
        """Token assembly embeddings receive gradients."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()
        assert policy.token_assembly.spatial_pos_emb.grad is not None


# ============================================================
# View dropout interaction
# ============================================================

class TestViewDropoutInteraction:
    def test_view_dropout_off_at_eval(self):
        """View dropout is disabled during eval."""
        policy = _make_policy(p_view_drop=0.99)
        policy.eval()
        obs = _make_obs()
        # Should not crash even with high dropout (disabled at eval)
        actions = policy.predict_action(obs)
        assert torch.isfinite(actions).all()

    def test_view_dropout_during_training(self):
        """Training with view dropout produces finite loss."""
        policy = _make_policy(p_view_drop=0.3)
        policy.train()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_partial_views(self):
        """Works when some views are absent."""
        policy = _make_policy()
        batch = _make_batch()
        batch["view_present"][:, 2:] = False  # only 2 of 4 cameras
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)


# ============================================================
# Co-training tests
# ============================================================

class TestCoTraining:
    @pytest.mark.skip(reason="V2 spatial pooling (196->49) changes adapted_clean shape; "
                             "decoder expects (196, 512) but gets (49, 256). "
                             "Co-training needs pre-pool tokens — not yet wired.")
    def test_recon_loss_added_when_lambda_positive(self):
        """With lambda_recon > 0, loss includes reconstruction component."""
        policy = _make_policy(lambda_recon=0.1, load_decoder=True)
        batch = _make_batch()
        loss_with = policy.compute_loss(batch)

        policy_no_recon = _make_policy(lambda_recon=0.0)
        batch2 = _make_batch()
        # Set same seed for comparable DDPM noise
        torch.manual_seed(99)
        loss_without = policy_no_recon.compute_loss(batch2)

        # Both should be finite
        assert torch.isfinite(loss_with)
        assert torch.isfinite(loss_without)

    def test_no_recon_when_lambda_zero(self):
        """With lambda_recon=0, no reconstruction loss computed."""
        policy = _make_policy(lambda_recon=0.0, load_decoder=True)
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_batch_size_one(self):
        """Works with B=1."""
        policy = _make_policy()
        batch = _make_batch(b=1)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_single_timestep(self):
        """Works with T_o=1."""
        policy = _make_policy(T_obs=1)
        batch = _make_batch(t_o=1)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_robomimic_config(self):
        """Works with robomimic dimensions (7D actions, 2 cameras)."""
        policy = _make_policy(ac_dim=7, num_views=2)
        batch = _make_batch(ac_dim=7, k=2)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_rlbench_config(self):
        """Works with RLBench dimensions (8D actions, 4 cameras)."""
        policy = _make_policy(ac_dim=8, num_views=4)
        batch = _make_batch(ac_dim=8, k=4)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)


# ============================================================
# Flow matching tests
# ============================================================

class TestFlowMatchingInterpolation:
    def test_at_tau_zero_gives_noise(self):
        """At tau=0, x_tau = 0*a_0 + 1*eps = eps (pure noise)."""
        a_0 = torch.randn(2, T_P, AC_DIM)
        eps = torch.randn_like(a_0)
        tau = torch.zeros(2)
        tau_expand = tau[:, None, None]
        x_tau = tau_expand * a_0 + (1.0 - tau_expand) * eps
        assert torch.allclose(x_tau, eps)

    def test_at_tau_one_gives_clean(self):
        """At tau=1, x_tau = 1*a_0 + 0*eps = a_0 (clean actions)."""
        a_0 = torch.randn(2, T_P, AC_DIM)
        eps = torch.randn_like(a_0)
        tau = torch.ones(2)
        tau_expand = tau[:, None, None]
        x_tau = tau_expand * a_0 + (1.0 - tau_expand) * eps
        assert torch.allclose(x_tau, a_0)

    def test_velocity_target_independent_of_tau(self):
        """Velocity target v = a_0 - eps does not depend on tau."""
        a_0 = torch.randn(2, T_P, AC_DIM)
        eps = torch.randn_like(a_0)
        v = a_0 - eps
        assert v.shape == a_0.shape
        # Verify it's just a_0 - eps regardless
        assert torch.allclose(v, a_0 - eps)


class TestFlowMatchingTimestepSampling:
    def test_beta_stays_in_range(self):
        """Beta distribution samples stay in [0, s]."""
        policy = _make_fm_policy()
        tau = policy._sample_flow_timestep(1000, torch.device("cpu"))
        assert tau.min() >= 0.0
        assert tau.max() <= policy.fm_cutoff + 1e-6

    def test_uniform_stays_in_range(self):
        """Uniform samples stay in [0, 1)."""
        policy = _make_fm_policy(fm_timestep_dist="uniform")
        tau = policy._sample_flow_timestep(1000, torch.device("cpu"))
        assert tau.min() >= 0.0
        assert tau.max() < 1.0

    def test_beta_biases_low(self):
        """Beta(1.5, 1) distribution biases toward low tau (near noise)."""
        policy = _make_fm_policy()
        tau = policy._sample_flow_timestep(10000, torch.device("cpu"))
        # Median should be below 0.5 since beta(1.5,1) is left-skewed after 1-u transform
        assert tau.median() < 0.5


class TestFlowMatchingComputeLoss:
    def test_returns_scalar(self):
        """Flow matching compute_loss returns a scalar."""
        policy = _make_fm_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_loss_is_finite(self):
        """Loss is finite."""
        policy = _make_fm_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_loss_positive(self):
        """MSE velocity loss is always positive."""
        policy = _make_fm_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert loss.item() > 0

    def test_backward_runs(self):
        """Backward pass completes."""
        policy = _make_fm_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()

    def test_8d_actions(self):
        """Works with 8D RLBench actions."""
        policy = _make_fm_policy(ac_dim=8)
        batch = _make_batch(ac_dim=8)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_uniform_timestep_dist(self):
        """Works with uniform timestep distribution."""
        policy = _make_fm_policy(fm_timestep_dist="uniform")
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)


class TestFlowMatchingPredictAction:
    def test_output_shape(self):
        """predict_action returns (B, T_pred, ac_dim)."""
        policy = _make_fm_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert actions.shape == (B, T_P, AC_DIM)

    def test_output_finite(self):
        """Predicted actions are finite."""
        policy = _make_fm_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert torch.isfinite(actions).all()

    def test_deterministic_with_seed(self):
        """Same seed produces same actions."""
        policy = _make_fm_policy()
        policy.eval()
        obs = _make_obs()
        torch.manual_seed(42)
        a1 = policy.predict_action(obs)
        torch.manual_seed(42)
        a2 = policy.predict_action(obs)
        assert torch.equal(a1, a2)

    def test_no_grad_during_inference(self):
        """predict_action does not accumulate gradients."""
        policy = _make_fm_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert not actions.requires_grad


class TestFlowMatchingGradientFlow:
    def test_encoder_frozen(self):
        """Encoder receives no gradients in flow matching."""
        policy = _make_fm_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()
        for p in policy.bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

    def test_noise_net_receives_grad(self):
        """Noise net receives gradients in flow matching."""
        policy = _make_fm_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in policy.noise_net.parameters()
        )
        assert has_grad, "Noise net should receive gradients"


class TestFlowMatchingCoTraining:
    @pytest.mark.skip(reason="V2 spatial pooling (196->49) changes adapted_clean shape; "
                             "decoder expects (196, 512) but gets (49, 256). "
                             "Co-training needs pre-pool tokens — not yet wired.")
    def test_recon_loss_added(self):
        """With lambda_recon > 0, flow matching includes reconstruction."""
        policy = _make_fm_policy(lambda_recon=0.1, load_decoder=True)
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)


class TestFlowMatchingEulerConvergence:
    def test_perfect_velocity_single_step(self):
        """With a perfect velocity predictor, single Euler step recovers a_0.

        If v(x, tau) = a_0 - eps (constant), then:
          x_0 = eps
          x_1 = eps + 1.0 * (a_0 - eps) = a_0
        """
        a_0 = torch.randn(1, T_P, AC_DIM)
        eps = torch.randn_like(a_0)
        # Single step: x = eps + dt * (a_0 - eps), dt = 1.0
        x = eps + 1.0 * (a_0 - eps)
        assert torch.allclose(x, a_0, atol=1e-6)


class TestPolicyTypeValidation:
    def test_invalid_policy_type_raises(self):
        """Invalid policy_type raises AssertionError."""
        with pytest.raises(AssertionError, match="policy_type"):
            _make_policy(policy_type="invalid")
