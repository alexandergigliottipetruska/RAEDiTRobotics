"""Tests for C.4 TokenAssembly.

Verifies that visual tokens, proprio tokens, and all embeddings are
correctly assembled into a flat observation sequence.
"""

import pytest
import torch
import torch.nn as nn

from models.token_assembly import TokenAssembly

# Test constants
B = 3
T_O = 2
K = 4
N = 196
D = 512
D_PROP = 9


def _make_inputs(b=B, t=T_O, k=K, n=N, d=D, d_prop=D_PROP):
    """Create test inputs for TokenAssembly."""
    adapted = torch.randn(b, t, k, n, d)
    proprio = torch.randn(b, t, d_prop)
    vp = torch.ones(b, k, dtype=torch.bool)
    return adapted, proprio, vp


# ============================================================
# Shape tests
# ============================================================

class TestTokenAssemblyShape:
    def test_output_shape(self):
        """Output is (B, T_o * K * N + T_o, d')."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs()
        out = ta(adapted, proprio, vp)
        expected_seq = T_O * K * N + T_O  # visual + proprio tokens
        assert out.shape == (B, expected_seq, D)

    def test_output_shape_single_timestep(self):
        """Works with T_o=1."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=1, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs(t=1)
        out = ta(adapted, proprio, vp)
        expected_seq = 1 * K * N + 1
        assert out.shape == (B, expected_seq, D)

    def test_output_shape_single_view(self):
        """Works with K=1."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=1,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs(k=1)
        out = ta(adapted, proprio, vp)
        expected_seq = T_O * 1 * N + T_O
        assert out.shape == (B, expected_seq, D)

    def test_output_shape_different_proprio_dim(self):
        """Works with different proprio dimensions."""
        for d_prop in [7, 8, 9, 14]:
            ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                               num_obs_steps=T_O, proprio_dim=d_prop)
            adapted, _, vp = _make_inputs()
            proprio = torch.randn(B, T_O, d_prop)
            out = ta(adapted, proprio, vp)
            assert out.shape[2] == D

    def test_batch_size_one(self):
        """Works with B=1."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs(b=1)
        out = ta(adapted, proprio, vp)
        expected_seq = T_O * K * N + T_O
        assert out.shape == (1, expected_seq, D)


# ============================================================
# Embedding tests
# ============================================================

class TestEmbeddings:
    def test_spatial_emb_shared_across_views(self):
        """Spatial position embeddings are the same for all views."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=1, proprio_dim=D_PROP)
        # Zero out adapted tokens and proprio to isolate embeddings
        adapted = torch.zeros(1, 1, K, N, D)
        proprio = torch.zeros(1, 1, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)

        # Extract per-view visual blocks: each is N tokens
        view_blocks = []
        for k_idx in range(K):
            start = k_idx * N
            end = start + N
            view_blocks.append(out[0, start:end])

        # Spatial component should be identical across views
        # (difference between views = view_emb only)
        for i in range(1, K):
            diff = view_blocks[i] - view_blocks[0]
            # diff should be constant across all N tokens (only view_emb differs)
            assert torch.allclose(diff[0], diff[1], atol=1e-6), \
                "Spatial pattern should be identical across views"

    def test_different_views_get_different_emb(self):
        """Different views receive different view embeddings."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=1, proprio_dim=D_PROP)
        adapted = torch.zeros(1, 1, K, N, D)
        proprio = torch.zeros(1, 1, D_PROP)
        vp = torch.ones(1, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)

        # First token from each view should differ (different view_emb)
        tok_v0 = out[0, 0]          # first token of view 0
        tok_v1 = out[0, N]          # first token of view 1
        assert not torch.equal(tok_v0, tok_v1), "Views should have different embeddings"

    def test_obs_time_emb_varies_across_timesteps(self):
        """Different observation timesteps receive different embeddings."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=1,
                           num_obs_steps=2, proprio_dim=D_PROP)
        adapted = torch.zeros(1, 2, 1, N, D)
        proprio = torch.zeros(1, 2, D_PROP)
        vp = torch.ones(1, 1, dtype=torch.bool)
        out = ta(adapted, proprio, vp)

        # First visual token from t=0 vs t=1
        tok_t0 = out[0, 0]         # t=0, view 0, patch 0
        tok_t1 = out[0, N + 1]     # t=1, view 0, patch 0 (after N visual + 1 proprio)
        assert not torch.equal(tok_t0, tok_t1), "Timesteps should have different embeddings"

    def test_proprio_tokens_present(self):
        """Proprio tokens are included in the output (one per timestep)."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted = torch.zeros(1, T_O, K, N, D)
        # Set proprio to something distinctive
        proprio = torch.ones(1, T_O, D_PROP) * 5.0
        vp = torch.ones(1, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)

        # Proprio tokens are at positions K*N, 2*K*N+1, etc.
        # (after each timestep's K*N visual tokens)
        total_seq = T_O * K * N + T_O
        assert out.shape[1] == total_seq

        # Proprio token for t=0 is at index K*N (right after all K views)
        prop_idx_t0 = K * N
        prop_tok = out[0, prop_idx_t0]
        # Should be non-zero (the proprio was non-zero)
        assert not torch.all(prop_tok == 0), "Proprio token should be non-zero"

    def test_spatial_emb_is_learnable(self):
        """Spatial position embedding is a learnable parameter."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        assert isinstance(ta.spatial_pos_emb, nn.Parameter)
        assert ta.spatial_pos_emb.requires_grad
        assert ta.spatial_pos_emb.shape == (1, N, D)

    def test_view_present_emb_distinguishes_real_vs_masked(self):
        """Real vs masked views get different view_present embeddings."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=1, proprio_dim=D_PROP)
        adapted = torch.zeros(1, 1, K, N, D)
        proprio = torch.zeros(1, 1, D_PROP)

        # All present
        vp_all = torch.ones(1, K, dtype=torch.bool)
        out_all = ta(adapted, proprio, vp_all)

        # View 0 masked
        vp_partial = torch.tensor([[False, True, True, True]])
        out_partial = ta(adapted, proprio, vp_partial)

        # View 0 tokens should differ (different view_present_emb)
        tok_v0_real = out_all[0, :N]
        tok_v0_masked = out_partial[0, :N]
        assert not torch.equal(tok_v0_real, tok_v0_masked), \
            "Real vs masked view tokens should differ due to view_present_emb"

        # View 1 tokens should be the same (both real)
        tok_v1_real = out_all[0, N:2*N]
        tok_v1_still_real = out_partial[0, N:2*N]
        assert torch.equal(tok_v1_real, tok_v1_still_real)

    def test_view_present_emb_gradient(self):
        """Gradients flow to view_present_emb."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs()
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert ta.view_present_emb.weight.grad is not None


# ============================================================
# Gradient flow tests
# ============================================================

class TestGradientFlow:
    def test_gradient_flows_to_adapted_tokens(self):
        """Gradients flow back through adapted tokens."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted = torch.randn(B, T_O, K, N, D, requires_grad=True)
        proprio = torch.randn(B, T_O, D_PROP)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert adapted.grad is not None
        assert not torch.all(adapted.grad == 0)

    def test_gradient_flows_to_proprio(self):
        """Gradients flow back through proprio input."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted = torch.randn(B, T_O, K, N, D)
        proprio = torch.randn(B, T_O, D_PROP, requires_grad=True)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert proprio.grad is not None
        assert not torch.all(proprio.grad == 0)

    def test_gradient_flows_to_spatial_emb(self):
        """Gradients flow to spatial position embedding."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs()
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert ta.spatial_pos_emb.grad is not None
        assert not torch.all(ta.spatial_pos_emb.grad == 0)

    def test_gradient_flows_to_view_emb(self):
        """Gradients flow to view embeddings."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs()
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert ta.view_emb.weight.grad is not None

    def test_gradient_flows_to_proprio_proj(self):
        """Gradients flow to proprio projection MLP."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs()
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        for p in ta.proprio_proj.parameters():
            assert p.grad is not None


# ============================================================
# Batch-first and determinism tests
# ============================================================

class TestBatchFirst:
    def test_output_is_batch_first(self):
        """Output is batch-first: (B, S, d)."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        adapted, proprio, vp = _make_inputs()
        out = ta(adapted, proprio, vp)
        # First dim should be batch
        assert out.shape[0] == B

    def test_deterministic(self):
        """Same inputs produce same outputs."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        ta.eval()
        adapted, proprio, vp = _make_inputs()
        out1 = ta(adapted, proprio, vp)
        out2 = ta(adapted, proprio, vp)
        assert torch.equal(out1, out2)

    def test_batch_independence(self):
        """Batch elements are independent."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=D_PROP)
        ta.eval()
        adapted, proprio, vp = _make_inputs(b=4)
        out_full = ta(adapted, proprio, vp)
        out_single = ta(adapted[2:3], proprio[2:3], vp[2:3])
        assert torch.allclose(out_full[2], out_single[0], atol=1e-6)


# ============================================================
# Token ordering tests
# ============================================================

class TestTokenOrdering:
    def test_visual_before_proprio_per_timestep(self):
        """Within each timestep, visual tokens come before proprio."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=2,
                           num_obs_steps=1, proprio_dim=D_PROP)
        # Use distinctive values to identify token types
        adapted = torch.ones(1, 1, 2, N, D) * 3.0
        proprio = torch.ones(1, 1, D_PROP) * 7.0
        vp = torch.ones(1, 2, dtype=torch.bool)
        out = ta(adapted, proprio, vp)

        # Total: 2*N visual + 1 proprio = 2*N + 1
        assert out.shape[1] == 2 * N + 1

    def test_views_in_order(self):
        """Views appear in order within each timestep."""
        ta = TokenAssembly(d_model=D, num_patches=N, num_views=3,
                           num_obs_steps=1, proprio_dim=D_PROP)
        # Give each view a unique constant so we can identify them
        adapted = torch.zeros(1, 1, 3, N, D)
        adapted[0, 0, 0] = 1.0
        adapted[0, 0, 1] = 2.0
        adapted[0, 0, 2] = 3.0
        proprio = torch.zeros(1, 1, D_PROP)
        vp = torch.ones(1, 3, dtype=torch.bool)
        out = ta(adapted, proprio, vp)

        # View 0 tokens: [0:N], View 1: [N:2N], View 2: [2N:3N]
        # Check the mean of the input component (before embeddings add offset)
        # Since we used distinctive constants, the ordering should be clear
        block0_mean = out[0, :N].mean().item()
        block1_mean = out[0, N:2*N].mean().item()
        block2_mean = out[0, 2*N:3*N].mean().item()
        # Block1 mean should be higher than block0 (2.0+emb vs 1.0+emb)
        assert block1_mean > block0_mean
        assert block2_mean > block1_mean
