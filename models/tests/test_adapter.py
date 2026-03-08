"""Tests for TrainableAdapter (A.2)."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import torch.nn as nn

from models.adapter import TrainableAdapter


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def adapter(device):
    return TrainableAdapter().to(device)


# ── Shape tests ──────────────────────────────────────────────────────

class TestAdapterShape:
    def test_output_shape_basic(self, adapter, device):
        z = torch.randn(2, 196, 1024, device=device)
        out = adapter(z)
        assert out.shape == (2, 196, 512)

    def test_output_shape_single_sample(self, adapter, device):
        z = torch.randn(1, 196, 1024, device=device)
        out = adapter(z)
        assert out.shape == (1, 196, 512)

    def test_output_shape_large_batch(self, adapter, device):
        z = torch.randn(8, 196, 1024, device=device)
        out = adapter(z)
        assert out.shape == (8, 196, 512)


# ── Forward pass tests ──────────────────────────────────────────────

class TestAdapterForward:
    def test_output_dtype(self, adapter, device):
        z = torch.randn(2, 196, 1024, device=device)
        out = adapter(z)
        assert out.dtype == torch.float32

    def test_output_device(self, adapter, device):
        z = torch.randn(2, 196, 1024, device=device)
        out = adapter(z)
        assert out.device.type == device.type

    def test_deterministic_forward(self, adapter, device):
        z = torch.randn(2, 196, 1024, device=device)
        out1 = adapter(z)
        out2 = adapter(z)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self, adapter, device):
        z1 = torch.randn(2, 196, 1024, device=device)
        z2 = torch.randn(2, 196, 1024, device=device)
        assert not torch.allclose(adapter(z1), adapter(z2))


# ── Noise augmentation tests ────────────────────────────────────────

class TestNoiseAugment:
    def test_noise_augment_preserves_shape(self, adapter, device):
        z_bar = torch.randn(2, 196, 512, device=device)
        z_tilde = adapter.noise_augment(z_bar, tau=0.8, training=True)
        assert z_tilde.shape == z_bar.shape

    def test_noise_augment_adds_noise_training(self, adapter, device):
        z_bar = torch.randn(2, 196, 512, device=device)
        z_tilde = adapter.noise_augment(z_bar, tau=0.8, training=True)
        assert not torch.allclose(z_bar, z_tilde)

    def test_noise_augment_identity_eval(self, adapter, device):
        z_bar = torch.randn(2, 196, 512, device=device)
        z_tilde = adapter.noise_augment(z_bar, tau=0.8, training=False)
        assert torch.allclose(z_bar, z_tilde)

    def test_noise_augment_default_training_true(self, adapter, device):
        """Calling without training= should default to True (adds noise)."""
        z_bar = torch.randn(2, 196, 512, device=device)
        z_tilde = adapter.noise_augment(z_bar)
        assert not torch.allclose(z_bar, z_tilde)

    def test_noise_augment_tau_zero_no_noise(self, adapter, device):
        z_bar = torch.randn(2, 196, 512, device=device)
        z_tilde = adapter.noise_augment(z_bar, tau=0.0, training=True)
        assert torch.allclose(z_bar, z_tilde)

    def test_noise_augment_different_each_call(self, adapter, device):
        z_bar = torch.randn(2, 196, 512, device=device)
        t1 = adapter.noise_augment(z_bar, tau=0.8, training=True)
        t2 = adapter.noise_augment(z_bar, tau=0.8, training=True)
        assert not torch.allclose(t1, t2)


# ── Parameter tests ─────────────────────────────────────────────────

class TestAdapterParameters:
    def test_all_params_trainable(self, adapter):
        for name, p in adapter.named_parameters():
            assert p.requires_grad, f"{name} is not trainable"

    def test_gradient_flows(self, adapter, device):
        z = torch.randn(2, 196, 1024, device=device)
        out = adapter(z)
        loss = out.sum()
        loss.backward()
        for name, p in adapter.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_architecture_layers(self, adapter):
        """Verify 2-layer MLP: Linear(1024,1024) -> GELU -> Linear(1024,512)."""
        layers = list(adapter.adapter)
        assert len(layers) == 3
        assert isinstance(layers[0], nn.Linear)
        assert layers[0].in_features == 1024 and layers[0].out_features == 1024
        assert isinstance(layers[1], nn.GELU)
        assert isinstance(layers[2], nn.Linear)
        assert layers[2].in_features == 1024 and layers[2].out_features == 512
