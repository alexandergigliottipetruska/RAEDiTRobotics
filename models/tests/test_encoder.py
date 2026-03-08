"""Tests for FrozenMultiViewEncoder (A.1) using mock backbone."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import torch.nn as nn

from models.encoder import FrozenMultiViewEncoder


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def encoder(device):
    return FrozenMultiViewEncoder(pretrained=False).to(device)


# ── Shape tests ──────────────────────────────────────────────────────

class TestEncoderShape:
    def test_output_shape_basic(self, encoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        out = encoder(x)
        assert out.shape == (2, 196, 1024)

    def test_output_shape_single_sample(self, encoder, device):
        x = torch.randn(1, 3, 224, 224, device=device)
        out = encoder(x)
        assert out.shape == (1, 196, 1024)

    def test_output_shape_large_batch(self, encoder, device):
        x = torch.randn(8, 3, 224, 224, device=device)
        out = encoder(x)
        assert out.shape == (8, 196, 1024)


# ── Frozen tests ─────────────────────────────────────────────────────

class TestEncoderFrozen:
    def test_all_params_frozen(self, encoder):
        for name, p in encoder.named_parameters():
            assert not p.requires_grad, f"{name} has requires_grad=True"

    def test_weights_unchanged_after_forward(self, encoder, device):
        checksum_before = sum(p.sum().item() for p in encoder.parameters())
        x = torch.randn(2, 3, 224, 224, device=device)
        _ = encoder(x)
        checksum_after = sum(p.sum().item() for p in encoder.parameters())
        assert checksum_before == checksum_after

    def test_backbone_in_eval_mode(self, encoder):
        assert not encoder.backbone.training

    def test_backbone_stays_eval_after_train_call(self, encoder):
        encoder.train()
        assert not encoder.backbone.training

    def test_no_gradients_on_backbone(self, encoder):
        for p in encoder.backbone.parameters():
            assert not p.requires_grad


# ── Cancel-Affine LayerNorm tests ───────────────────────────────────

class TestCancelAffineLayerNorm:
    def test_output_mean_near_zero(self, encoder, device):
        x = torch.randn(4, 3, 224, 224, device=device)
        out = encoder(x)
        mean_val = out.mean().item()
        assert abs(mean_val) < 0.1, f"Mean is {mean_val}, expected near 0"

    def test_output_std_near_one(self, encoder, device):
        x = torch.randn(4, 3, 224, 224, device=device)
        out = encoder(x)
        std_val = out.std().item()
        assert abs(std_val - 1.0) < 0.1, f"Std is {std_val}, expected near 1"

    def test_layernorm_no_affine(self, encoder):
        """Cancel-affine means elementwise_affine=False (no learned gamma/beta)."""
        assert not encoder.cancel_affine_ln.elementwise_affine


# ── Forward pass tests ──────────────────────────────────────────────

class TestEncoderForward:
    def test_output_dtype(self, encoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        out = encoder(x)
        assert out.dtype == torch.float32

    def test_output_device(self, encoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        out = encoder(x)
        assert out.device.type == device.type

    def test_deterministic_forward(self, encoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self, encoder, device):
        x1 = torch.randn(2, 3, 224, 224, device=device)
        x2 = torch.randn(2, 3, 224, 224, device=device)
        assert not torch.allclose(encoder(x1), encoder(x2))


# ── Mock mode tests ─────────────────────────────────────────────────

class TestMockMode:
    def test_mock_creates_without_secrets(self):
        """pretrained=False should not require secrets.yaml or HF login."""
        enc = FrozenMultiViewEncoder(pretrained=False)
        assert enc is not None

    def test_mock_backbone_output_shape(self, encoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        out = encoder(x)
        assert out.shape == (2, 196, 1024)
