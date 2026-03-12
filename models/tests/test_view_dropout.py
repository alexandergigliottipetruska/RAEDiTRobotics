"""Tests for C.3 ViewDropout.

ViewDropout replaces entire camera view tokens with a learned mask token
during training. Applied AFTER adapter, BEFORE token assembly.
"""

import pytest
import torch
import torch.nn as nn

from models.view_dropout import ViewDropout

# Test constants
B = 4       # batch size
K = 4       # camera slots
N = 196     # tokens per view (14x14)
D = 512     # model dim


def _make_inputs(b=B, k=K, n=N, d=D):
    """Create test inputs: adapted_tokens [B, K, N, d] and view_present [B, K]."""
    tokens = torch.randn(b, k, n, d)
    view_present = torch.ones(b, k, dtype=torch.bool)
    return tokens, view_present


# ============================================================
# Shape tests
# ============================================================

class TestViewDropoutShape:
    def test_output_shapes(self):
        """Output shapes match input shapes."""
        vd = ViewDropout(d_model=D, p_drop=0.5)
        vd.train()
        tokens, vp = _make_inputs()
        out_tokens, out_vp = vd(tokens, vp)
        assert out_tokens.shape == tokens.shape
        assert out_vp.shape == vp.shape

    def test_mask_token_shape(self):
        """Mask token is a d-dimensional vector."""
        vd = ViewDropout(d_model=D)
        assert vd.mask_token.shape == (D,)

    def test_mask_token_is_parameter(self):
        """Mask token is a learnable nn.Parameter."""
        vd = ViewDropout(d_model=D)
        assert isinstance(vd.mask_token, nn.Parameter)
        assert vd.mask_token.requires_grad


# ============================================================
# Eval / no-op behavior
# ============================================================

class TestViewDropoutEval:
    def test_noop_at_eval(self):
        """No views are dropped during evaluation."""
        vd = ViewDropout(d_model=D, p_drop=0.99)
        vd.eval()
        tokens, vp = _make_inputs()
        out_tokens, out_vp = vd(tokens, vp)
        assert torch.equal(out_tokens, tokens)
        assert torch.equal(out_vp, vp)

    def test_noop_when_p_drop_zero(self):
        """No views are dropped when p_drop=0."""
        vd = ViewDropout(d_model=D, p_drop=0.0)
        vd.train()
        tokens, vp = _make_inputs()
        out_tokens, out_vp = vd(tokens, vp)
        assert torch.equal(out_tokens, tokens)
        assert torch.equal(out_vp, vp)


# ============================================================
# Drop behavior
# ============================================================

class TestViewDropoutBehavior:
    def test_approximate_drop_rate(self):
        """Empirical drop rate is close to p_drop over many trials."""
        p = 0.3
        vd = ViewDropout(d_model=D, p_drop=p)
        vd.train()
        total_views = 0
        dropped_views = 0
        for _ in range(200):
            tokens, vp = _make_inputs(b=8, k=4)
            _, out_vp = vd(tokens, vp)
            total_views += vp.sum().item()
            dropped_views += (vp & ~out_vp).sum().item()
        empirical_rate = dropped_views / total_views
        assert abs(empirical_rate - p) < 0.05, (
            f"Empirical drop rate {empirical_rate:.3f} too far from {p}"
        )

    def test_dropped_views_filled_with_mask_token(self):
        """Dropped view tokens are replaced with the mask token, not zeros."""
        vd = ViewDropout(d_model=D, p_drop=1.0)
        vd.train()
        tokens, vp = _make_inputs()
        out_tokens, out_vp = vd(tokens, vp)
        # Views marked as dropped should be filled with mask_token
        # (one view per sample is kept by the safety guarantee)
        expected = vd.mask_token.detach().unsqueeze(0).expand(N, -1)
        for b in range(B):
            for k in range(K):
                if not out_vp[b, k]:  # this view was dropped
                    assert torch.allclose(out_tokens[b, k], expected), (
                        f"Dropped view [{b},{k}] not filled with mask token"
                    )

    def test_present_views_unchanged(self):
        """Views that are NOT dropped retain their original tokens."""
        vd = ViewDropout(d_model=D, p_drop=0.0)
        vd.train()
        tokens, vp = _make_inputs()
        out_tokens, _ = vd(tokens, vp)
        assert torch.equal(out_tokens, tokens)

    def test_absent_views_not_dropped(self):
        """Already-absent views (view_present=False) are not re-dropped."""
        vd = ViewDropout(d_model=D, p_drop=1.0)
        vd.train()
        tokens, vp = _make_inputs()
        # Mark view 0 as absent for all batches
        vp[:, 0] = False
        original_view0 = tokens[:, 0].clone()
        out_tokens, out_vp = vd(tokens, vp)
        # View 0 was already absent — should NOT be modified
        assert torch.equal(out_tokens[:, 0], original_view0)
        assert not out_vp[:, 0].any()

    def test_view_present_updated_for_drops(self):
        """Dropped views have their view_present set to False."""
        vd = ViewDropout(d_model=D, p_drop=1.0)
        vd.train()
        tokens, vp = _make_inputs()
        _, out_vp = vd(tokens, vp)
        # With p=1.0, all but one view per sample should be dropped
        # (safety guarantee keeps exactly one)
        assert (out_vp.sum(dim=1) == 1).all(), (
            "Each sample should have exactly 1 view remaining with p=1.0"
        )

    def test_never_drops_all_views(self):
        """At least one view must remain present per sample (safety guarantee)."""
        vd = ViewDropout(d_model=D, p_drop=0.99)
        vd.train()
        # Run many times — should never produce all-False row
        for _ in range(500):
            tokens, vp = _make_inputs(b=8, k=4)
            _, out_vp = vd(tokens, vp)
            per_sample = out_vp.sum(dim=1)  # [B]
            assert (per_sample >= 1).all(), "Some sample has zero present views!"


# ============================================================
# Gradient flow
# ============================================================

class TestViewDropoutGradient:
    def test_gradient_flows_to_mask_token(self):
        """Gradient flows through the mask token when views are dropped."""
        vd = ViewDropout(d_model=D, p_drop=1.0)
        vd.train()
        tokens, vp = _make_inputs()
        out_tokens, _ = vd(tokens, vp)
        loss = out_tokens.sum()
        loss.backward()
        assert vd.mask_token.grad is not None
        assert not torch.all(vd.mask_token.grad == 0)

    def test_gradient_flows_through_kept_views(self):
        """Gradient flows through views that were NOT dropped."""
        vd = ViewDropout(d_model=D, p_drop=0.0)
        vd.train()
        tokens = torch.randn(B, K, N, D, requires_grad=True)
        vp = torch.ones(B, K, dtype=torch.bool)
        out_tokens, _ = vd(tokens, vp)
        loss = out_tokens.sum()
        loss.backward()
        assert tokens.grad is not None
        assert not torch.all(tokens.grad == 0)


# ============================================================
# Edge cases
# ============================================================

class TestViewDropoutEdgeCases:
    def test_single_view(self):
        """Works with K=1 (single camera). Never drops it."""
        vd = ViewDropout(d_model=D, p_drop=0.99)
        vd.train()
        tokens, vp = _make_inputs(k=1)
        out_tokens, out_vp = vd(tokens, vp)
        # With K=1, must keep the only view
        assert out_vp.all()
        assert torch.equal(out_tokens, tokens)

    def test_batch_size_one(self):
        """Works with B=1."""
        vd = ViewDropout(d_model=D, p_drop=0.5)
        vd.train()
        tokens, vp = _make_inputs(b=1)
        out_tokens, out_vp = vd(tokens, vp)
        assert out_tokens.shape == tokens.shape
