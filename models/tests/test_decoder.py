"""Tests for ViTDecoder (A.3)."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import torch.nn as nn

from models.decoder import ViTDecoder


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def decoder(device):
    return ViTDecoder().to(device)


# ── Shape tests ──────────────────────────────────────────────────────

class TestDecoderShape:
    def test_output_shape_basic(self, decoder, device):
        z = torch.randn(2, 196, 512, device=device)
        out = decoder(z)
        assert out.shape == (2, 3, 224, 224)

    def test_output_shape_single_sample(self, decoder, device):
        z = torch.randn(1, 196, 512, device=device)
        out = decoder(z)
        assert out.shape == (1, 3, 224, 224)

    def test_output_shape_large_batch(self, decoder, device):
        z = torch.randn(8, 196, 512, device=device)
        out = decoder(z)
        assert out.shape == (8, 3, 224, 224)


# ── Output range tests ──────────────────────────────────────────────

class TestDecoderOutputRange:
    def test_output_in_zero_one(self, decoder, device):
        z = torch.randn(4, 196, 512, device=device)
        out = decoder(z)
        assert out.min() >= 0.0, f"Min is {out.min().item()}"
        assert out.max() <= 1.0, f"Max is {out.max().item()}"

    def test_output_not_all_same(self, decoder, device):
        z = torch.randn(2, 196, 512, device=device)
        out = decoder(z)
        assert out.std() > 0.0, "All output pixels are identical"

    def test_different_inputs_different_outputs(self, decoder, device):
        z1 = torch.randn(2, 196, 512, device=device)
        z2 = torch.randn(2, 196, 512, device=device)
        assert not torch.allclose(decoder(z1), decoder(z2))


# ── last_layer_weight tests ─────────────────────────────────────────

class TestLastLayerWeight:
    def test_last_layer_weight_exists(self, decoder):
        assert hasattr(decoder, 'last_layer_weight')

    def test_last_layer_weight_is_parameter(self, decoder):
        assert isinstance(decoder.last_layer_weight, (nn.Parameter, torch.Tensor))

    def test_last_layer_weight_is_head_weight(self, decoder):
        """last_layer_weight should be the final linear layer's weight."""
        assert decoder.last_layer_weight is decoder.head.weight

    def test_last_layer_weight_requires_grad(self, decoder):
        assert decoder.last_layer_weight.requires_grad

    def test_adaptive_lambda_gradient_flows(self, decoder, device):
        """Simulate compute_adaptive_lambda: grad of output w.r.t. last_layer_weight must exist."""
        z = torch.randn(2, 196, 512, device=device)
        out = decoder(z)
        fake_loss = out.mean()
        grads = torch.autograd.grad(fake_loss, decoder.last_layer_weight, retain_graph=True)
        assert grads[0] is not None
        assert grads[0].shape == decoder.last_layer_weight.shape


# ── Unpatchify tests ────────────────────────────────────────────────

class TestUnpatchify:
    def test_unpatchify_shape(self, decoder, device):
        x = torch.randn(2, 196, 768, device=device)  # 768 = 16*16*3
        imgs = decoder.unpatchify(x)
        assert imgs.shape == (2, 3, 224, 224)

    def test_unpatchify_single_sample(self, decoder, device):
        x = torch.randn(1, 196, 768, device=device)
        imgs = decoder.unpatchify(x)
        assert imgs.shape == (1, 3, 224, 224)


# ── Architecture tests ──────────────────────────────────────────────

class TestDecoderArchitecture:
    def test_hidden_dim(self, decoder):
        assert decoder.hidden_dim == 512

    def test_num_patches(self, decoder):
        assert decoder.num_patches == 196

    def test_grid_size(self, decoder):
        assert decoder.grid_size == 14

    def test_head_projects_to_pixel_space(self, decoder):
        """head should map 512 -> 768 (16*16*3)."""
        assert decoder.head.in_features == 512
        assert decoder.head.out_features == 768

    def test_pos_embed_shape(self, decoder):
        assert decoder.pos_embed.shape == (1, 196, 512)


# ── Parameter tests ─────────────────────────────────────────────────

class TestDecoderParameters:
    def test_all_params_trainable(self, decoder):
        for name, p in decoder.named_parameters():
            assert p.requires_grad, f"{name} is not trainable"

    def test_gradient_flows_through_full_forward(self, decoder, device):
        z = torch.randn(2, 196, 512, device=device)
        out = decoder(z)
        loss = out.sum()
        loss.backward()
        for name, p in decoder.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
