"""C.0 Round-Trip Tests: Batch-first conversion of diffusion.py

Verifies that all components in diffusion.py operate correctly in
batch-first (B, S, d) format after the C.0 conversion.

Test categories:
  1. Shape — every class outputs (B, S, d) or (B, T, out)
  2. Batch independence — output[i] depends only on input[i]
     (catches transposed batch/seq dims)
  3. Gradient flow — all trainable parameters receive gradients
  4. Zero-init — adaLN-Zero blocks start with zero gates
  5. Consistency — forward_enc + forward_dec == forward
  6. Edge cases — batch=1, single token, B==S, 8D actions, NaN check
"""

import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.diffusion import (
    _PositionalEncoding,
    _TimeNetwork,
    _SelfAttnEncoder,
    _TransformerEncoder,
    _ShiftScaleMod,
    _ZeroScaleMod,
    _FinalLayer,
    _DiTCrossAttnBlock,
    _LightningDiTBlock,
    _DiTNoiseNet,
    _DDTHead,
    _DiTDHNoiseNet,
)

# Small dims for fast tests
D = 64
NHEAD = 4
FFN = 128
B = 4
S_OBS = 16
T_P = 8
AC_DIM = 7
NUM_BLOCKS = 2


def _make_noise_net(**overrides):
    """Helper to create a small _DiTNoiseNet with test defaults."""
    defaults = dict(
        ac_dim=AC_DIM, ac_chunk=T_P, hidden_dim=D,
        num_blocks=NUM_BLOCKS, nhead=NHEAD, dim_feedforward=FFN,
    )
    defaults.update(overrides)
    return _DiTNoiseNet(**defaults)


def _make_ddt_head(**overrides):
    """Helper to create a small _DDTHead with test defaults."""
    defaults = dict(
        ac_dim=AC_DIM, ac_chunk=T_P, backbone_dim=D,
        head_dim=128, num_blocks=2, nhead=NHEAD, dim_feedforward=FFN,
    )
    defaults.update(overrides)
    return _DDTHead(**defaults)


# =====================================================================
# 1. Shape Tests
# =====================================================================

class TestPositionalEncodingShape:
    def test_output_shape(self):
        pe = _PositionalEncoding(D)
        x = torch.randn(B, S_OBS, D)
        out = pe(x)
        assert out.shape == (B, S_OBS, D)

    def test_shared_across_batch(self):
        """All batch elements get identical positional encodings."""
        pe = _PositionalEncoding(D)
        x = torch.randn(B, S_OBS, D)
        out = pe(x)
        for i in range(1, B):
            assert torch.equal(out[0], out[i])

    def test_prefix_consistency(self):
        """Shorter sequence gets prefix of longer sequence's PE."""
        pe = _PositionalEncoding(D)
        short = torch.randn(B, 10, D)
        long_ = torch.randn(B, 20, D)
        out_short = pe(short)
        out_long = pe(long_)
        assert torch.equal(out_short, out_long[:, :10])

    def test_deterministic(self):
        pe = _PositionalEncoding(D)
        x = torch.randn(B, S_OBS, D)
        assert torch.equal(pe(x), pe(x))


class TestSelfAttnEncoderShape:
    def test_output_shape(self):
        enc = _SelfAttnEncoder(D, nhead=NHEAD, dim_feedforward=FFN)
        src = torch.randn(B, S_OBS, D)
        pos = torch.randn(B, S_OBS, D)
        out = enc(src, pos)
        assert out.shape == (B, S_OBS, D)


class TestTransformerEncoderShape:
    def test_per_depth_output_shapes(self):
        base = _SelfAttnEncoder(D, nhead=NHEAD, dim_feedforward=FFN)
        enc = _TransformerEncoder(base, NUM_BLOCKS)
        src = torch.randn(B, S_OBS, D)
        pos = torch.randn(B, S_OBS, D)
        outputs = enc(src, pos)
        assert len(outputs) == NUM_BLOCKS
        for o in outputs:
            assert o.shape == (B, S_OBS, D)


class TestShiftScaleModShape:
    def test_output_shape(self):
        mod = _ShiftScaleMod(D)
        x = torch.randn(B, T_P, D)
        c = torch.randn(B, D)
        out = mod(x, c)
        assert out.shape == (B, T_P, D)


class TestZeroScaleModShape:
    def test_output_shape(self):
        mod = _ZeroScaleMod(D)
        x = torch.randn(B, T_P, D)
        c = torch.randn(B, D)
        out = mod(x, c)
        assert out.shape == (B, T_P, D)

    def test_zero_init_produces_zeros(self):
        """With zero-initialized weights, output should be all zeros."""
        mod = _ZeroScaleMod(D)
        mod.reset_parameters()
        x = torch.randn(B, T_P, D)
        c = torch.randn(B, D)
        out = mod(x, c)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


class TestFinalLayerShape:
    def test_output_shape(self):
        fl = _FinalLayer(D, AC_DIM)
        x = torch.randn(B, T_P, D)
        t = torch.randn(B, D)
        cond = torch.randn(B, S_OBS, D)
        out = fl(x, t, cond)
        assert out.shape == (B, T_P, AC_DIM)

    def test_zero_init_all_params(self):
        """FinalLayer zero-initializes all parameters."""
        fl = _FinalLayer(D, AC_DIM)
        for name, p in fl.named_parameters():
            assert torch.allclose(p, torch.zeros_like(p)), \
                f"Parameter {name} not zero-initialized"


class TestTimeNetworkShape:
    def test_output_shape(self):
        tn = _TimeNetwork(256, D)
        t = torch.randint(0, 100, (B,)).float()
        out = tn(t)
        assert out.shape == (B, D)


class TestDiTCrossAttnBlockShape:
    def test_output_shape(self):
        block = _DiTCrossAttnBlock(D, nhead=NHEAD, dim_feedforward=FFN)
        x = torch.randn(B, T_P, D)
        t = torch.randn(B, D)
        cond = torch.randn(B, S_OBS, D)
        out = block(x, t, cond)
        assert out.shape == (B, T_P, D)


class TestLightningDiTBlockShape:
    def test_output_shape(self):
        block = _LightningDiTBlock(D, nhead=NHEAD, dim_feedforward=FFN)
        x = torch.randn(B, T_P, D)
        t = torch.randn(B, D)
        cond = torch.randn(B, S_OBS, D)
        out = block(x, t, cond)
        assert out.shape == (B, T_P, D)

    def test_adaln_zero_init(self):
        """adaLN output layer is zero-initialized."""
        block = _LightningDiTBlock(D, nhead=NHEAD, dim_feedforward=FFN)
        # Zero-timestep should produce zero modulation
        t = torch.zeros(B, D)
        ada_out = block.adaLN(t)
        assert torch.allclose(ada_out, torch.zeros_like(ada_out), atol=1e-7)


class TestDiTNoiseNetShape:
    def test_forward_shapes(self):
        net = _make_noise_net()
        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        enc_cache, eps = net(noise_ac, time, obs)
        assert eps.shape == (B, T_P, AC_DIM)
        assert isinstance(enc_cache, list)
        assert len(enc_cache) == NUM_BLOCKS
        for e in enc_cache:
            assert e.shape == (B, S_OBS, D)

    def test_forward_enc_shape(self):
        net = _make_noise_net()
        obs = torch.randn(B, S_OBS, D)
        enc_cache = net.forward_enc(obs)
        assert isinstance(enc_cache, list)
        assert len(enc_cache) == NUM_BLOCKS
        for e in enc_cache:
            assert e.shape == (B, S_OBS, D)

    def test_forward_dec_shape(self):
        net = _make_noise_net()
        obs = torch.randn(B, S_OBS, D)
        enc_cache = net.forward_enc(obs)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        eps = net.forward_dec(noise_ac, time, enc_cache)
        assert eps.shape == (B, T_P, AC_DIM)

    def test_external_conditioning_single_tensor(self):
        """Single tensor (not list) as enc_cache — RAE-DiT path."""
        net = _make_noise_net()
        ext_cond = torch.randn(B, 32, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        enc_cache, eps = net(noise_ac, time, None, enc_cache=ext_cond)
        assert eps.shape == (B, T_P, AC_DIM)
        assert enc_cache is ext_cond

    def test_use_lightning_false(self):
        """Standard block variant works in batch-first."""
        net = _make_noise_net(use_lightning=False)
        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        _, eps = net(noise_ac, time, obs)
        assert eps.shape == (B, T_P, AC_DIM)


class TestDDTHeadShape:
    def test_output_shape(self):
        head = _make_ddt_head()
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        zt = torch.randn(B, T_P, D)
        out = head(noise_ac, time, zt)
        assert out.shape == (B, T_P, AC_DIM)


class TestDiTDHNoiseNetShape:
    def test_forward_shapes(self):
        backbone = _make_noise_net()
        head = _make_ddt_head()
        ditdh = _DiTDHNoiseNet(backbone, head)
        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        enc_cache, eps = ditdh(noise_ac, time, obs)
        assert eps.shape == (B, T_P, AC_DIM)
        assert isinstance(enc_cache, list)


# =====================================================================
# 2. Batch Independence Tests (CRITICAL for detecting transposed dims)
# =====================================================================

class TestBatchIndependence:
    """If batch and sequence dims are swapped, output[i] would depend on
    input[j]. These tests catch that by comparing batch-8 with batch-4
    runs (first 4 samples should match exactly in eval mode)."""

    def test_encoder_batch_independence(self):
        torch.manual_seed(42)
        base = _SelfAttnEncoder(D, nhead=NHEAD, dim_feedforward=FFN)
        enc = _TransformerEncoder(base, NUM_BLOCKS)
        enc.eval()

        src = torch.randn(8, S_OBS, D)
        pos = torch.randn(8, S_OBS, D)

        out_8 = enc(src, pos)
        out_4 = enc(src[:4], pos[:4])

        for o8, o4 in zip(out_8, out_4):
            assert torch.allclose(o8[:4], o4, atol=1e-5), \
                "Encoder output depends on other batch elements!"

    def test_noise_net_batch_independence(self):
        torch.manual_seed(42)
        net = _make_noise_net()
        net.eval()

        obs = torch.randn(8, S_OBS, D)
        noise_ac = torch.randn(8, T_P, AC_DIM)
        time = torch.randint(0, 100, (8,))

        _, eps_8 = net(noise_ac, time, obs)
        _, eps_4 = net(noise_ac[:4], time[:4], obs[:4])

        assert torch.allclose(eps_8[:4], eps_4, atol=1e-5), \
            "NoiseNet output depends on other batch elements!"

    def test_noise_net_standard_block_batch_independence(self):
        """Same test with use_lightning=False."""
        torch.manual_seed(42)
        net = _make_noise_net(use_lightning=False)
        net.eval()

        obs = torch.randn(8, S_OBS, D)
        noise_ac = torch.randn(8, T_P, AC_DIM)
        time = torch.randint(0, 100, (8,))

        _, eps_8 = net(noise_ac, time, obs)
        _, eps_4 = net(noise_ac[:4], time[:4], obs[:4])

        assert torch.allclose(eps_8[:4], eps_4, atol=1e-5), \
            "Standard-block NoiseNet output depends on other batch elements!"

    def test_ddt_head_batch_independence(self):
        torch.manual_seed(42)
        head = _make_ddt_head()
        head.eval()

        noise_ac = torch.randn(8, T_P, AC_DIM)
        time = torch.randint(0, 100, (8,))
        zt = torch.randn(8, T_P, D)

        out_8 = head(noise_ac, time, zt)
        out_4 = head(noise_ac[:4], time[:4], zt[:4])

        assert torch.allclose(out_8[:4], out_4, atol=1e-5), \
            "DDTHead output depends on other batch elements!"

    def test_ditdh_batch_independence(self):
        torch.manual_seed(42)
        backbone = _make_noise_net()
        head = _make_ddt_head()
        ditdh = _DiTDHNoiseNet(backbone, head)
        ditdh.eval()

        obs = torch.randn(8, S_OBS, D)
        noise_ac = torch.randn(8, T_P, AC_DIM)
        time = torch.randint(0, 100, (8,))

        _, eps_8 = ditdh(noise_ac, time, obs)
        _, eps_4 = ditdh(noise_ac[:4], time[:4], obs[:4])

        assert torch.allclose(eps_8[:4], eps_4, atol=1e-5), \
            "DiTDH output depends on other batch elements!"

    def test_equal_batch_and_seq_dims(self):
        """B == S is the MOST DANGEROUS case: shapes are compatible either
        way, so a transposed format silently computes the wrong thing."""
        torch.manual_seed(42)
        net = _make_noise_net()
        net.eval()

        # B=16, S_obs=16 — if dims are swapped, shapes still match
        obs_16 = torch.randn(16, 16, D)
        noise_ac = torch.randn(16, T_P, AC_DIM)
        time = torch.randint(0, 100, (16,))

        _, eps_16 = net(noise_ac, time, obs_16)
        _, eps_4 = net(noise_ac[:4], time[:4], obs_16[:4])

        assert torch.allclose(eps_16[:4], eps_4, atol=1e-5), \
            "Batch independence fails when B == S_obs!"


# =====================================================================
# 3. Consistency Tests
# =====================================================================

class TestConsistency:
    def test_enc_dec_equals_forward(self):
        """forward_enc + forward_dec should match full forward."""
        torch.manual_seed(42)
        net = _make_noise_net()
        net.eval()

        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))

        enc_cache_1, eps_1 = net(noise_ac, time, obs)
        enc_cache_2 = net.forward_enc(obs)
        eps_2 = net.forward_dec(noise_ac, time, enc_cache_2)

        assert torch.allclose(eps_1, eps_2, atol=1e-5)
        for e1, e2 in zip(enc_cache_1, enc_cache_2):
            assert torch.allclose(e1, e2, atol=1e-5)

    def test_ditdh_enc_dec_equals_forward(self):
        torch.manual_seed(42)
        backbone = _make_noise_net()
        head = _make_ddt_head()
        ditdh = _DiTDHNoiseNet(backbone, head)
        ditdh.eval()

        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))

        enc_cache_1, eps_1 = ditdh(noise_ac, time, obs)
        enc_cache_2 = ditdh.forward_enc(obs)
        eps_2 = ditdh.forward_dec(noise_ac, time, enc_cache_2)

        assert torch.allclose(eps_1, eps_2, atol=1e-5)


# =====================================================================
# 4. Gradient Flow Tests
# =====================================================================

class TestGradientFlow:
    def test_noise_net_all_params_receive_gradient(self):
        """All params get nonzero gradients after perturbing zero-init layers."""
        net = _make_noise_net()
        # DiT identity init: _FinalLayer and adaLN-Zero modulation start at zero,
        # blocking gradients to most of the decoder path (ac_proj, dec_pos,
        # time_net, decoder blocks). Perturb all zero-init params so that
        # gradients flow through the entire network — mirrors state after 1 step.
        with torch.no_grad():
            for p in net.parameters():
                if p.requires_grad and torch.all(p == 0):
                    p.add_(torch.randn_like(p) * 0.01)

        obs = torch.randn(2, 8, D)
        noise_ac = torch.randn(2, T_P, AC_DIM)
        time = torch.randint(0, 100, (2,))
        _, eps = net(noise_ac, time, obs)
        eps.sum().backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_dec_pos_gradient_after_warmup(self):
        """dec_pos receives nonzero gradient once FinalLayer params are nonzero."""
        net = _make_noise_net()
        # Perturb FinalLayer params away from zero so gradient flows
        with torch.no_grad():
            for p in net.eps_out.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        obs = torch.randn(2, 8, D)
        noise_ac = torch.randn(2, T_P, AC_DIM)
        time = torch.randint(0, 100, (2,))
        _, eps = net(noise_ac, time, obs)
        eps.sum().backward()
        assert net.dec_pos.grad is not None
        assert not torch.all(net.dec_pos.grad == 0)

    def test_obs_input_receives_gradient(self):
        """Gradients flow back through encoder to observation input."""
        net = _make_noise_net()
        obs = torch.randn(2, 8, D, requires_grad=True)
        noise_ac = torch.randn(2, T_P, AC_DIM)
        time = torch.randint(0, 100, (2,))
        _, eps = net(noise_ac, time, obs)
        eps.sum().backward()
        assert obs.grad is not None
        assert torch.isfinite(obs.grad).all()

    def test_noise_actions_receive_gradient(self):
        """Gradients flow back to noised action input."""
        net = _make_noise_net()
        obs = torch.randn(2, 8, D)
        noise_ac = torch.randn(2, T_P, AC_DIM, requires_grad=True)
        time = torch.randint(0, 100, (2,))
        _, eps = net(noise_ac, time, obs)
        eps.sum().backward()
        assert noise_ac.grad is not None

    def test_ditdh_gradient_flow(self):
        backbone = _make_noise_net()
        head = _make_ddt_head()
        ditdh = _DiTDHNoiseNet(backbone, head)

        obs = torch.randn(2, 8, D, requires_grad=True)
        noise_ac = torch.randn(2, T_P, AC_DIM)
        time = torch.randint(0, 100, (2,))
        _, eps = ditdh(noise_ac, time, obs)
        eps.sum().backward()
        assert obs.grad is not None

    def test_standard_block_gradient_flow(self):
        """use_lightning=False: all params receive gradients."""
        net = _make_noise_net(use_lightning=False)
        obs = torch.randn(2, 8, D)
        noise_ac = torch.randn(2, T_P, AC_DIM)
        time = torch.randint(0, 100, (2,))
        _, eps = net(noise_ac, time, obs)
        eps.sum().backward()

        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# =====================================================================
# 5. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_batch_size_one(self):
        net = _make_noise_net()
        obs = torch.randn(1, S_OBS, D)
        noise_ac = torch.randn(1, T_P, AC_DIM)
        time = torch.randint(0, 100, (1,))
        _, eps = net(noise_ac, time, obs)
        assert eps.shape == (1, T_P, AC_DIM)

    def test_single_obs_token(self):
        net = _make_noise_net()
        obs = torch.randn(B, 1, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        _, eps = net(noise_ac, time, obs)
        assert eps.shape == (B, T_P, AC_DIM)

    def test_8d_actions_rlbench(self):
        """RLBench uses 8D actions."""
        net = _make_noise_net(ac_dim=8)
        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, 8)
        time = torch.randint(0, 100, (B,))
        _, eps = net(noise_ac, time, obs)
        assert eps.shape == (B, T_P, 8)

    def test_outputs_finite(self):
        """No NaN or Inf in any output."""
        net = _make_noise_net()
        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        enc_cache, eps = net(noise_ac, time, obs)
        assert torch.isfinite(eps).all(), "NaN or Inf in eps output"
        for i, e in enumerate(enc_cache):
            assert torch.isfinite(e).all(), f"NaN or Inf in enc_cache[{i}]"

    def test_large_obs_sequence(self):
        """Realistic obs token count: T_o=2, K=4, N=196 = 1568 tokens."""
        net = _make_noise_net()
        obs = torch.randn(2, 1568, D)
        noise_ac = torch.randn(2, T_P, AC_DIM)
        time = torch.randint(0, 100, (2,))
        _, eps = net(noise_ac, time, obs)
        assert eps.shape == (2, T_P, AC_DIM)

    def test_ddt_head_with_different_backbone_dim(self):
        """DDTHead with backbone_dim != head_dim."""
        head = _DDTHead(
            ac_dim=AC_DIM, ac_chunk=T_P, backbone_dim=32,
            head_dim=256, num_blocks=2, nhead=4, dim_feedforward=FFN,
        )
        noise_ac = torch.randn(B, T_P, AC_DIM)
        time = torch.randint(0, 100, (B,))
        zt = torch.randn(B, T_P, 32)
        out = head(noise_ac, time, zt)
        assert out.shape == (B, T_P, AC_DIM)

    def test_prediction_horizon_50(self):
        """T_p=50 (RLBench default)."""
        net = _make_noise_net(ac_chunk=50)
        obs = torch.randn(B, S_OBS, D)
        noise_ac = torch.randn(B, 50, AC_DIM)
        time = torch.randint(0, 100, (B,))
        _, eps = net(noise_ac, time, obs)
        assert eps.shape == (B, 50, AC_DIM)
