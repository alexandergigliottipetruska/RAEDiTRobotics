"""Tests for PatchDiscriminator (Phase A.4).

Tests cover:
  - Output shape correctness
  - Only head parameters are trainable (backbone frozen)
  - Gradient flow: head receives gradients, backbone does not
  - Mock mode works without pretrained weights
  - Parameter count sanity check for head
"""

import torch
import pytest

from models.discriminator import PatchDiscriminator


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_discriminator_output_shape():
    """Output should be (B, 1) logits for any batch size."""
    disc = PatchDiscriminator(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = disc(x)
    assert logits.shape == (4, 1)


def test_discriminator_single_image():
    """Should handle batch size 1."""
    disc = PatchDiscriminator(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    logits = disc(x)
    assert logits.shape == (1, 1)


def test_discriminator_large_batch():
    """Should handle larger batches."""
    disc = PatchDiscriminator(pretrained=False)
    x = torch.randn(16, 3, 224, 224)
    logits = disc(x)
    assert logits.shape == (16, 1)


# ---------------------------------------------------------------------------
# Trainability / frozen backbone tests
# ---------------------------------------------------------------------------

def test_discriminator_head_trainable():
    """Only head parameters should have requires_grad=True."""
    disc = PatchDiscriminator(pretrained=False)
    trainable = [n for n, p in disc.named_parameters() if p.requires_grad]
    frozen = [n for n, p in disc.named_parameters() if not p.requires_grad]

    # All trainable params must be in the head
    assert len(trainable) > 0, "Must have at least some trainable params"
    assert all("head" in n for n in trainable), (
        f"Non-head params are trainable: {[n for n in trainable if 'head' not in n]}"
    )


def test_discriminator_backbone_frozen():
    """Backbone parameters should all be frozen."""
    disc = PatchDiscriminator(pretrained=False)
    backbone_params = [
        (n, p) for n, p in disc.named_parameters()
        if "head" not in n
    ]
    assert len(backbone_params) > 0, "Mock should have backbone params to test"
    for name, param in backbone_params:
        assert not param.requires_grad, f"Backbone param {name} should be frozen"


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------

def test_discriminator_gradient_flows_to_head():
    """Backward pass should produce gradients for head parameters."""
    disc = PatchDiscriminator(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = disc(x)
    loss = logits.mean()
    loss.backward()

    for name, p in disc.named_parameters():
        if "head" in name:
            assert p.grad is not None, f"Head param {name} should have gradient"
            assert p.grad.abs().sum() > 0, f"Head param {name} gradient should be non-zero"


def test_discriminator_no_gradient_to_backbone():
    """Backward pass should NOT produce gradients for backbone."""
    disc = PatchDiscriminator(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = disc(x)
    loss = logits.mean()
    loss.backward()

    backbone_params = [
        (n, p) for n, p in disc.named_parameters()
        if "head" not in n
    ]
    assert len(backbone_params) > 0, "Mock should have backbone params to test"
    for name, p in backbone_params:
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"Backbone param {name} should not receive gradient"
        )


# ---------------------------------------------------------------------------
# Parameter count tests
# ---------------------------------------------------------------------------

def test_discriminator_head_param_count():
    """Head should have: Linear(384,256) + Linear(256,1) = ~99K params."""
    disc = PatchDiscriminator(pretrained=False)
    head_params = sum(
        p.numel() for n, p in disc.named_parameters()
        if "head" in n and p.requires_grad
    )
    # Linear(384, 256): 384*256 + 256 = 98,560
    # Linear(256, 1):   256*1 + 1     = 257
    # Total: ~98,817
    assert 95_000 < head_params < 105_000, f"Head param count {head_params} outside expected range"


# ---------------------------------------------------------------------------
# Mock mode test
# ---------------------------------------------------------------------------

def test_discriminator_mock_produces_finite_values():
    """Mock discriminator should produce finite, non-NaN logits."""
    disc = PatchDiscriminator(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = disc(x)
    assert torch.isfinite(logits).all(), "Logits contain inf or NaN"


def test_discriminator_eval_mode():
    """Should work in eval mode (inference)."""
    disc = PatchDiscriminator(pretrained=False)
    disc.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 224, 224)
        logits = disc(x)
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()
