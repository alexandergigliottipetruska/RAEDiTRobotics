"""Tests for C.1 BasePolicy.

Verifies the abstract base class contract and its integration
with DiffusionTransformerAgent (the existing concrete subclass).
"""

import pytest
import torch
import torch.nn as nn

from models.base_policy import BasePolicy


# ============================================================
# Abstract contract tests
# ============================================================

class TestBasePolicyContract:
    def test_is_nn_module(self):
        """BasePolicy inherits from nn.Module."""
        assert issubclass(BasePolicy, nn.Module)

    def test_compute_loss_raises(self):
        """compute_loss raises NotImplementedError on the base class."""
        policy = BasePolicy()
        with pytest.raises(NotImplementedError):
            policy.compute_loss({})

    def test_predict_action_raises(self):
        """predict_action raises NotImplementedError on the base class."""
        policy = BasePolicy()
        with pytest.raises(NotImplementedError):
            policy.predict_action({})

    def test_instantiable(self):
        """BasePolicy can be instantiated (it's not truly abstract via ABC)."""
        policy = BasePolicy()
        assert isinstance(policy, nn.Module)

    def test_has_compute_loss_method(self):
        """BasePolicy defines compute_loss."""
        assert hasattr(BasePolicy, "compute_loss")
        assert callable(BasePolicy.compute_loss)

    def test_has_predict_action_method(self):
        """BasePolicy defines predict_action."""
        assert hasattr(BasePolicy, "predict_action")
        assert callable(BasePolicy.predict_action)


# ============================================================
# Subclass contract tests
# ============================================================

class _DummyPolicy(BasePolicy):
    """Minimal concrete subclass for testing the contract."""
    def __init__(self, action_dim=7, T_pred=16):
        super().__init__()
        self.net = nn.Linear(64, action_dim * T_pred)
        self.action_dim = action_dim
        self.T_pred = T_pred

    def compute_loss(self, batch):
        x = batch["obs"]
        pred = self.net(x).reshape(x.shape[0], self.T_pred, self.action_dim)
        return (pred - batch["actions"]).pow(2).mean()

    def predict_action(self, obs):
        x = obs["obs"]
        return self.net(x).reshape(x.shape[0], self.T_pred, self.action_dim)


class TestSubclassContract:
    def test_subclass_is_base_policy(self):
        """Concrete subclass is an instance of BasePolicy."""
        policy = _DummyPolicy()
        assert isinstance(policy, BasePolicy)

    def test_subclass_is_nn_module(self):
        """Concrete subclass is an instance of nn.Module."""
        policy = _DummyPolicy()
        assert isinstance(policy, nn.Module)

    def test_compute_loss_returns_scalar(self):
        """compute_loss returns a scalar tensor."""
        policy = _DummyPolicy()
        batch = {
            "obs": torch.randn(4, 64),
            "actions": torch.randn(4, 16, 7),
        }
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.requires_grad

    def test_predict_action_returns_correct_shape(self):
        """predict_action returns (B, T_pred, action_dim)."""
        policy = _DummyPolicy(action_dim=8, T_pred=16)
        obs = {"obs": torch.randn(2, 64)}
        actions = policy.predict_action(obs)
        assert actions.shape == (2, 16, 8)

    def test_subclass_parameters_trainable(self):
        """Subclass parameters are registered with nn.Module."""
        policy = _DummyPolicy()
        params = list(policy.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_gradient_flows_through_compute_loss(self):
        """Gradients flow from compute_loss back through the network."""
        policy = _DummyPolicy()
        batch = {
            "obs": torch.randn(4, 64),
            "actions": torch.randn(4, 16, 7),
        }
        loss = policy.compute_loss(batch)
        loss.backward()
        for p in policy.parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)


# ============================================================
# Integration with DiffusionTransformerAgent
# ============================================================

class TestDiffusionTransformerAgentInheritance:
    def test_agent_inherits_base_policy(self):
        """DiffusionTransformerAgent is a subclass of BasePolicy."""
        from models.diffusion import DiffusionTransformerAgent
        assert issubclass(DiffusionTransformerAgent, BasePolicy)

    def test_agent_super_init_accepts_kwargs(self):
        """BasePolicy.__init__ accepts the kwargs that DiffusionTransformerAgent passes."""
        policy = BasePolicy(
            features="test",
            feat_dim=1024,
            token_dim=512,
            n_cams=4,
            odim=9,
            use_obs=False,
            imgs_per_cam=1,
            share_cam_features=False,
            view_dropout_p=0.0,
            dropout=0,
            feat_norm=None,
            max_patches_per_view=None,
        )
        assert isinstance(policy, nn.Module)
        assert policy._base_config["feat_dim"] == 1024
        assert policy._base_config["n_cams"] == 4

    def test_base_config_stored(self):
        """kwargs passed to BasePolicy are stored in _base_config."""
        policy = BasePolicy(hidden_dim=512, ac_dim=7)
        assert policy._base_config == {"hidden_dim": 512, "ac_dim": 7}

    def test_no_args_gives_empty_config(self):
        """BasePolicy() with no args gives empty _base_config."""
        policy = BasePolicy()
        assert policy._base_config == {}
