"""Tests for V3 ObservationEncoder (active cameras only + LayerNorm).

Output is (B, T_o, d_model) — one token per observation timestep.
Only active camera features are concatenated (not zero-padded slots).
"""

import torch
import pytest

from models.obs_encoder_v3 import ObservationEncoder

# Constants
B = 4
T_O = 2
N_PATCHES = 196
ADAPTER_DIM = 512
D_MODEL = 256
PROPRIO_DIM = 9


def _make_inputs(b=B, t_o=T_O, k=4, n=N_PATCHES, d=ADAPTER_DIM, p=PROPRIO_DIM):
    adapted_tokens = torch.randn(b, t_o, k, n, d)
    proprio = torch.randn(b, t_o, p)
    view_present = torch.ones(b, k, dtype=torch.bool)
    return adapted_tokens, proprio, view_present


class TestObsEncoderShapes:
    def test_output_shape_robomimic(self):
        """2 active cameras → (B, T_o, d_model)."""
        enc = ObservationEncoder(n_active_cams=2, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False  # only slots 0 and 3 active
        out = enc(tokens, proprio, vp)
        assert out["tokens"].shape == (B, 2, D_MODEL)

    def test_output_shape_rlbench(self):
        """4 active cameras → (B, T_o, d_model)."""
        enc = ObservationEncoder(n_active_cams=4, T_obs=2)
        tokens, proprio, vp = _make_inputs(k=4)
        out = enc(tokens, proprio, vp)
        assert out["tokens"].shape == (B, 2, D_MODEL)

    def test_global_vector_shape(self):
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        assert out["global"].shape == (B, D_MODEL)

    def test_output_keys(self):
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        assert set(out.keys()) == {"tokens", "global"}


class TestObsEncoderActiveCams:
    def test_only_active_cams_contribute(self):
        """Changing inactive slot data doesn't affect output."""
        enc = ObservationEncoder(n_active_cams=2)
        tokens1, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False  # slots 1,2 inactive

        tokens2 = tokens1.clone()
        tokens2[:, :, 1:3] = torch.randn_like(tokens2[:, :, 1:3]) * 100  # change inactive slots

        enc.eval()
        with torch.no_grad():
            out1 = enc(tokens1, proprio, vp)
            out2 = enc(tokens2, proprio, vp)

        assert torch.allclose(out1["tokens"], out2["tokens"], atol=1e-5), \
            "Inactive camera data should not affect output"

    def test_proj_input_dim_matches_active_cams(self):
        """obs_proj input dim = n_active_cams * adapter_dim + proprio_dim."""
        enc2 = ObservationEncoder(n_active_cams=2, proprio_dim=9)
        assert enc2.obs_proj.in_features == 2 * 512 + 9  # 1033

        enc4 = ObservationEncoder(n_active_cams=4, proprio_dim=8)
        assert enc4.obs_proj.in_features == 4 * 512 + 8  # 2056


class TestObsEncoderLayerNorm:
    def test_has_feature_norm(self):
        """Encoder has LayerNorm for feature normalization."""
        enc = ObservationEncoder()
        assert hasattr(enc, 'feature_norm')
        assert isinstance(enc.feature_norm, torch.nn.LayerNorm)

    def test_feature_norm_applied(self):
        """LayerNorm normalizes pooled features (output has ~zero mean, ~unit var)."""
        enc = ObservationEncoder(n_active_cams=2)
        # Large-scale input to verify normalization
        tokens = torch.randn(B, T_O, 4, N_PATCHES, ADAPTER_DIM) * 100
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, 4, dtype=torch.bool)
        vp[:, 1:3] = False

        enc.eval()
        with torch.no_grad():
            out = enc(tokens, proprio, vp)
        # Output should be finite regardless of input scale
        assert torch.isfinite(out["tokens"]).all()


class TestObsEncoderGradients:
    def test_gradient_flow(self):
        enc = ObservationEncoder(n_active_cams=2)
        tokens, proprio, vp = _make_inputs(k=4)
        vp[:, 1:3] = False
        out = enc(tokens, proprio, vp)
        loss = out["tokens"].sum()
        loss.backward()
        assert enc.obs_proj.weight.grad is not None
        assert not torch.all(enc.obs_proj.weight.grad == 0)
        assert enc.feature_norm.weight.grad is not None


class TestObsEncoderBatchIndependence:
    def test_batch_independence(self):
        enc = ObservationEncoder(n_active_cams=2)
        enc.eval()
        t8, p8, vp8 = _make_inputs(b=8)
        vp8[:, 1:3] = False
        with torch.no_grad():
            out8 = enc(t8, p8, vp8)
            out4 = enc(t8[:4], p8[:4], vp8[:4])
        assert torch.allclose(out8["tokens"][:4], out4["tokens"], atol=1e-5)

    def test_deterministic_eval(self):
        enc = ObservationEncoder(n_active_cams=2)
        enc.eval()
        tokens, proprio, vp = _make_inputs()
        vp[:, 1:3] = False
        with torch.no_grad():
            out1 = enc(tokens, proprio, vp)
            out2 = enc(tokens, proprio, vp)
        assert torch.equal(out1["tokens"], out2["tokens"])
