"""Tests for loss functions (Phase A.5).

Wave 1 (22 tests) — shape, positivity, basic properties:
  - L1 loss: shape, positivity, zero-for-identical, differentiability
  - LPIPS loss: shape, positivity, near-zero identical, no gradient to VGG
  - GAN generator loss: shape, sign behavior, differentiable
  - GAN discriminator loss: shape, non-negative, perfect/poor discrimination
  - Adaptive lambda: shape, clamping, detached, tiny-gradient handling
  - Reconstruction loss: recon-only, with-GAN, GAN-differs-from-recon

Wave 2 (25+ tests) — numerical correctness, edge cases, stress tests:
  - L1 loss: hand-computed values, symmetry, batch-size independence
  - LPIPS loss: range conversion correctness, gradient flows to pred,
    different spatial sizes, batch size 1
  - GAN generator: exact numerical value, gradient magnitude
  - GAN discriminator: exact hinge math, margin boundary, symmetry
  - Adaptive lambda: known gradient ratio, equal gradients → λ≈1,
    large rec small gan → large λ, multi-dim weight
  - Reconstruction loss: omega_L scaling, omega_G scaling,
    differentiable end-to-end, assert guards

References:
  - Zheng et al. 2025, Appendix C.2, Table 12
  - Esser et al. 2021 (VQGAN) for adaptive lambda
"""

import torch
import torch.nn as nn
import pytest


from models.losses import (
    l1_loss,
    lpips_loss_fn,
    gan_generator_loss,
    gan_discriminator_loss,
    compute_adaptive_lambda,
    reconstruction_loss,
    create_lpips_net,
)


# ---------------------------------------------------------------------------
# L1 loss
# ---------------------------------------------------------------------------

def test_l1_loss_shape():
    """L1 loss should be a scalar."""
    pred = torch.randn(2, 3, 224, 224)
    target = torch.randn(2, 3, 224, 224)
    loss = l1_loss(pred, target)
    assert loss.shape == ()


def test_l1_loss_positive():
    """L1 loss should be positive for non-identical inputs."""
    pred = torch.randn(2, 3, 224, 224)
    target = torch.randn(2, 3, 224, 224)
    loss = l1_loss(pred, target)
    assert loss.item() > 0


def test_l1_loss_zero_for_identical():
    """L1 loss should be zero when pred == target."""
    x = torch.randn(2, 3, 224, 224)
    loss = l1_loss(x, x)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_l1_loss_differentiable():
    """L1 loss should produce gradients for pred."""
    pred = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randn(2, 3, 32, 32)
    loss = l1_loss(pred, target)
    loss.backward()
    assert pred.grad is not None


# ---------------------------------------------------------------------------
# LPIPS loss
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lpips_net():
    """Create LPIPS network once for all tests (downloads VGG weights)."""
    return create_lpips_net()


def test_lpips_loss_shape(lpips_net):
    """LPIPS loss should be a scalar."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    loss = lpips_loss_fn(pred, target, lpips_net)
    assert loss.shape == ()


def test_lpips_loss_positive(lpips_net):
    """LPIPS loss should be positive for different images."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    loss = lpips_loss_fn(pred, target, lpips_net)
    assert loss.item() > 0


def test_lpips_loss_near_zero_for_identical(lpips_net):
    """LPIPS loss should be near zero for identical images."""
    x = torch.randn(2, 3, 64, 64)
    loss = lpips_loss_fn(x, x, lpips_net)
    assert loss.item() < 0.01


def test_lpips_no_gradient_to_network(lpips_net):
    """LPIPS VGG backbone should not receive gradients."""
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    loss = lpips_loss_fn(pred, target, lpips_net)
    loss.backward()
    for p in lpips_net.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


# ---------------------------------------------------------------------------
# GAN generator loss
# ---------------------------------------------------------------------------

def test_gan_generator_loss_shape():
    """Generator loss should be a scalar."""
    logits_fake = torch.randn(8, 1)
    loss = gan_generator_loss(logits_fake)
    assert loss.shape == ()


def test_gan_generator_loss_sign():
    """Generator wants fake logits to be positive (high).
    Loss = -mean(logits_fake), so positive logits -> negative loss."""
    positive_logits = torch.ones(8, 1) * 5.0
    negative_logits = torch.ones(8, 1) * -5.0
    loss_pos = gan_generator_loss(positive_logits)
    loss_neg = gan_generator_loss(negative_logits)
    # Generator loss should be lower when fake logits are positive
    assert loss_pos.item() < loss_neg.item()


def test_gan_generator_loss_differentiable():
    """Generator loss should be differentiable w.r.t. logits."""
    logits = torch.randn(4, 1, requires_grad=True)
    loss = gan_generator_loss(logits)
    loss.backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# GAN discriminator loss
# ---------------------------------------------------------------------------

def test_gan_disc_loss_shape():
    """Discriminator loss should be a scalar."""
    real = torch.randn(8, 1)
    fake = torch.randn(8, 1)
    loss = gan_discriminator_loss(real, fake)
    assert loss.shape == ()


def test_gan_disc_loss_non_negative():
    """Hinge loss is always non-negative."""
    real = torch.randn(8, 1)
    fake = torch.randn(8, 1)
    loss = gan_discriminator_loss(real, fake)
    assert loss.item() >= 0


def test_gan_disc_loss_perfect_discrimination():
    """When disc perfectly separates real (high) and fake (low),
    hinge loss should be near zero."""
    real = torch.ones(8, 1) * 5.0   # well above margin 1
    fake = torch.ones(8, 1) * -5.0  # well below margin -1
    loss = gan_discriminator_loss(real, fake)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_gan_disc_loss_poor_discrimination():
    """When disc can't separate, hinge loss should be positive."""
    real = torch.ones(8, 1) * -1.0  # wrong: real scored low
    fake = torch.ones(8, 1) * 1.0   # wrong: fake scored high
    loss = gan_discriminator_loss(real, fake)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Adaptive lambda
# ---------------------------------------------------------------------------

def test_adaptive_lambda_shape():
    """Lambda should be a scalar."""
    w = torch.randn(10, requires_grad=True)
    L_rec = (w ** 2).sum()
    L_gan = (w * 3).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert lam.shape == ()


def test_adaptive_lambda_clamped():
    """Lambda should be in [0, 1e4]."""
    w = torch.randn(10, requires_grad=True)
    L_rec = (w ** 2).sum()
    L_gan = (w * 3).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert 0 <= lam.item() <= 1e4


def test_adaptive_lambda_detached():
    """Lambda should be detached (no gradient through it)."""
    w = torch.randn(10, requires_grad=True)
    L_rec = (w ** 2).sum()
    L_gan = (w * 3).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert not lam.requires_grad


def test_adaptive_lambda_tiny_gan_grad():
    """When GAN gradient is near-zero, lambda should be large but clamped."""
    w = torch.randn(10, requires_grad=True)
    L_rec = (w ** 2).sum()
    # L_gan has tiny gradient w.r.t. w (1e-10 scale)
    L_gan = (w * 1e-10).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert torch.isfinite(lam)
    assert lam.item() <= 1e4


# ---------------------------------------------------------------------------
# Combined reconstruction loss
# ---------------------------------------------------------------------------

def test_reconstruction_loss_recon_only(lpips_net):
    """Without GAN phase, loss should be L1 + omega_L * LPIPS."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    loss = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
        omega_L=1.0, omega_G=0.75,
    )
    assert loss.shape == ()
    assert loss.item() > 0


def test_reconstruction_loss_with_gan(lpips_net):
    """With GAN phase, loss should include adaptive lambda term.

    We need a real computation graph where both L_rec and L_gan
    depend on last_layer_weight, otherwise torch.autograd.grad fails.
    Simulate this by making pred depend on a parameter (like
    decoder.head.weight would in real training).
    """
    # Simulate decoder's last layer: a scalar weight that pred depends on
    last_w = nn.Parameter(torch.ones(1))
    raw = torch.randn(2, 3, 64, 64)
    pred = raw * last_w  # pred depends on last_w

    target = torch.randn(2, 3, 64, 64)

    # logits_fake also depends on pred (as it would through discriminator)
    logits_fake = pred.mean(dim=[1, 2, 3], keepdim=True)  # (2, 1)

    loss = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=True, logits_fake=logits_fake,
        last_layer_weight=last_w,
        omega_L=1.0, omega_G=0.75,
    )
    assert loss.shape == ()
    assert loss.item() > 0


def test_reconstruction_loss_gan_larger_than_recon_only(lpips_net):
    """Loss with GAN should differ from loss without GAN."""
    last_w = nn.Parameter(torch.ones(1))
    raw = torch.randn(2, 3, 64, 64)
    pred = raw * last_w
    target = torch.randn(2, 3, 64, 64)
    logits_fake = pred.mean(dim=[1, 2, 3], keepdim=True)

    loss_no_gan = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
        omega_L=1.0, omega_G=0.75,
    )
    loss_with_gan = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=True, logits_fake=logits_fake,
        last_layer_weight=last_w,
        omega_L=1.0, omega_G=0.75,
    )
    # They should differ (GAN term adds something)
    assert loss_no_gan.item() != pytest.approx(loss_with_gan.item(), abs=1e-6)


# ===========================================================================
# WAVE 2: Numerical correctness, edge cases, stress tests
# ===========================================================================

# ---------------------------------------------------------------------------
# L1 loss — numerical correctness
# ---------------------------------------------------------------------------

def test_l1_loss_hand_computed():
    """L1 loss should match hand-computed value.
    pred = [[1, 2], [3, 4]], target = [[0, 0], [0, 0]]
    L1 = mean(|1|+|2|+|3|+|4|) = (1+2+3+4)/4 = 2.5
    """
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros(2, 2)
    loss = l1_loss(pred, target)
    assert loss.item() == pytest.approx(2.5, abs=1e-6)


def test_l1_loss_symmetric():
    """L1(pred, target) should equal L1(target, pred)."""
    a = torch.randn(4, 3, 32, 32)
    b = torch.randn(4, 3, 32, 32)
    assert l1_loss(a, b).item() == pytest.approx(l1_loss(b, a).item(), abs=1e-6)


def test_l1_loss_scales_with_magnitude():
    """Doubling the difference should double the loss."""
    target = torch.zeros(2, 3, 16, 16)
    pred_1x = torch.ones(2, 3, 16, 16)
    pred_2x = torch.ones(2, 3, 16, 16) * 2
    loss_1x = l1_loss(pred_1x, target)
    loss_2x = l1_loss(pred_2x, target)
    assert loss_2x.item() == pytest.approx(2.0 * loss_1x.item(), abs=1e-5)


def test_l1_loss_batch_size_independent():
    """L1 (mean reduction) should give same value regardless of batch size.
    With identical images repeated, the mean shouldn't change."""
    single = torch.randn(1, 3, 16, 16)
    target = torch.randn(1, 3, 16, 16)
    # Repeat to batch of 4
    pred_4 = single.repeat(4, 1, 1, 1)
    target_4 = target.repeat(4, 1, 1, 1)
    loss_1 = l1_loss(single, target)
    loss_4 = l1_loss(pred_4, target_4)
    assert loss_1.item() == pytest.approx(loss_4.item(), abs=1e-5)


def test_l1_loss_gradient_magnitude():
    """L1 gradient should be ±1/N (sign of diff, averaged)."""
    pred = torch.tensor([3.0, -2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    loss = l1_loss(pred, target)
    loss.backward()
    # grad = sign(pred - target) / N = [1, -1] / 2 = [0.5, -0.5]
    expected = torch.tensor([0.5, -0.5])
    assert torch.allclose(pred.grad, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# LPIPS loss — range conversion, spatial flexibility
# ---------------------------------------------------------------------------

def test_lpips_gradient_flows_to_pred(lpips_net):
    """LPIPS loss should produce gradients for pred (needed for decoder training)."""
    pred = torch.randn(2, 3, 64, 64, requires_grad=True)
    target = torch.randn(2, 3, 64, 64)
    loss = lpips_loss_fn(pred, target, lpips_net)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_lpips_handles_different_spatial_sizes(lpips_net):
    """LPIPS should work with various spatial dimensions (VGG is flexible)."""
    for size in [32, 48, 64, 128]:
        pred = torch.randn(1, 3, size, size)
        target = torch.randn(1, 3, size, size)
        loss = lpips_loss_fn(pred, target, lpips_net)
        assert loss.shape == (), f"Failed for size {size}"
        assert torch.isfinite(loss), f"Non-finite for size {size}"


def test_lpips_batch_size_one(lpips_net):
    """LPIPS should work with single-image batches."""
    pred = torch.randn(1, 3, 64, 64)
    target = torch.randn(1, 3, 64, 64)
    loss = lpips_loss_fn(pred, target, lpips_net)
    assert loss.shape == ()
    assert loss.item() > 0


def test_lpips_range_conversion_matters(lpips_net):
    """Verify our [0,1]→[-1,1] conversion is important.
    LPIPS expects [-1,1]; feeding [0,1] directly would give different results.
    We test that our function and a manual [-1,1] feed give the same result."""
    pred_01 = torch.rand(2, 3, 64, 64)  # in [0, 1]
    target_01 = torch.rand(2, 3, 64, 64)  # in [0, 1]

    # Our function (converts internally)
    loss_ours = lpips_loss_fn(pred_01, target_01, lpips_net)

    # Manual conversion and direct LPIPS call
    pred_11 = pred_01 * 2 - 1
    target_11 = target_01 * 2 - 1
    loss_manual = lpips_net(pred_11, target_11).mean()

    assert loss_ours.item() == pytest.approx(loss_manual.item(), abs=1e-6)


def test_lpips_different_images_higher_than_similar(lpips_net):
    """LPIPS should be higher for very different images vs slightly different."""
    base = torch.rand(2, 3, 64, 64)
    similar = base + torch.randn_like(base) * 0.01  # tiny perturbation
    different = torch.rand(2, 3, 64, 64)  # completely different

    loss_similar = lpips_loss_fn(base, similar, lpips_net)
    loss_different = lpips_loss_fn(base, different, lpips_net)
    assert loss_similar.item() < loss_different.item()


# ---------------------------------------------------------------------------
# GAN generator loss — numerical
# ---------------------------------------------------------------------------

def test_gan_generator_loss_exact_value():
    """Generator loss = -mean(logits). Verify exact value."""
    logits = torch.tensor([[2.0], [4.0], [6.0]])
    loss = gan_generator_loss(logits)
    # -mean([2, 4, 6]) = -4.0
    assert loss.item() == pytest.approx(-4.0, abs=1e-6)


def test_gan_generator_loss_zero_logits():
    """Zero logits should give zero loss."""
    logits = torch.zeros(4, 1)
    loss = gan_generator_loss(logits)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_gan_generator_loss_gradient_is_uniform():
    """Gradient of -mean(x) w.r.t. x should be -1/N for all elements."""
    logits = torch.randn(4, 1, requires_grad=True)
    loss = gan_generator_loss(logits)
    loss.backward()
    expected_grad = -1.0 / 4.0
    assert torch.allclose(logits.grad, torch.full_like(logits, expected_grad), atol=1e-6)


# ---------------------------------------------------------------------------
# GAN discriminator loss — hinge math
# ---------------------------------------------------------------------------

def test_gan_disc_loss_exact_value():
    """Hand-compute hinge loss for specific values.
    real = [0.5], fake = [0.5]
    relu(1 - 0.5) = 0.5, relu(1 + 0.5) = 1.5
    loss = mean(0.5) + mean(1.5) = 2.0
    """
    real = torch.tensor([[0.5]])
    fake = torch.tensor([[0.5]])
    loss = gan_discriminator_loss(real, fake)
    assert loss.item() == pytest.approx(2.0, abs=1e-6)


def test_gan_disc_loss_at_margin_boundary():
    """At exactly the hinge margins (real=1, fake=-1), loss should be zero."""
    real = torch.ones(4, 1) * 1.0   # relu(1-1) = 0
    fake = torch.ones(4, 1) * -1.0  # relu(1+(-1)) = 0
    loss = gan_discriminator_loss(real, fake)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_gan_disc_loss_real_term_only():
    """When fake is well-classified (< -1), only real term contributes."""
    real = torch.zeros(4, 1)         # relu(1-0) = 1
    fake = torch.ones(4, 1) * -5.0  # relu(1+(-5)) = relu(-4) = 0
    loss = gan_discriminator_loss(real, fake)
    # Only real term: mean(relu(1 - 0)) = 1.0
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_gan_disc_loss_fake_term_only():
    """When real is well-classified (> 1), only fake term contributes."""
    real = torch.ones(4, 1) * 5.0   # relu(1-5) = relu(-4) = 0
    fake = torch.zeros(4, 1)        # relu(1+0) = 1
    loss = gan_discriminator_loss(real, fake)
    # Only fake term: mean(relu(1 + 0)) = 1.0
    assert loss.item() == pytest.approx(1.0, abs=1e-6)


def test_gan_disc_loss_symmetric_in_mistake():
    """Swapping real and fake (both wrong) should still be non-negative."""
    logits = torch.randn(8, 1)
    loss_normal = gan_discriminator_loss(logits, -logits)
    loss_swapped = gan_discriminator_loss(-logits, logits)
    assert loss_normal.item() >= 0
    assert loss_swapped.item() >= 0


# ---------------------------------------------------------------------------
# Adaptive lambda — numerical verification
# ---------------------------------------------------------------------------

def test_adaptive_lambda_equal_gradients():
    """When rec and GAN have equal gradient norms, λ should ≈ 1."""
    w = torch.randn(10, requires_grad=True)
    # Both losses are same function of w → equal gradients
    L_rec = (w * 2).sum()
    L_gan = (w * 2).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert lam.item() == pytest.approx(1.0, abs=0.01)


def test_adaptive_lambda_known_ratio():
    """When rec_grad_norm / gan_grad_norm = 3, lambda should be ≈ 3.
    L_rec = (3*w).sum() → grad = 3*ones(10), norm = 3*sqrt(10)
    L_gan = (1*w).sum() → grad = 1*ones(10), norm = sqrt(10)
    ratio = 3.0
    """
    w = torch.randn(10, requires_grad=True)
    L_rec = (w * 3).sum()
    L_gan = (w * 1).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert lam.item() == pytest.approx(3.0, abs=0.1)


def test_adaptive_lambda_large_rec_small_gan():
    """When rec gradient is much larger, λ should be large (but clamped)."""
    w = torch.randn(10, requires_grad=True)
    L_rec = (w * 1000).sum()
    L_gan = (w * 0.001).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    # ratio = 1000/0.001 = 1e6, should be clamped to 1e4
    assert lam.item() == pytest.approx(1e4, abs=1.0)


def test_adaptive_lambda_multidim_weight():
    """Lambda should work with multi-dimensional weight tensors (like real conv weights)."""
    w = torch.randn(64, 32, 3, 3, requires_grad=True)
    L_rec = (w ** 2).sum()
    L_gan = (w * 0.5).sum()
    lam = compute_adaptive_lambda(L_rec, L_gan, w)
    assert lam.shape == ()
    assert torch.isfinite(lam)
    assert 0 <= lam.item() <= 1e4


def test_adaptive_lambda_preserves_computation_graph():
    """Lambda should be detached but the losses should still have their graphs.
    This tests that retain_graph=True works correctly."""
    w = torch.randn(10, requires_grad=True)
    L_rec = (w ** 2).sum()
    L_gan = (w * 3).sum()

    lam = compute_adaptive_lambda(L_rec, L_gan, w)

    # Lambda itself is detached
    assert not lam.requires_grad

    # But the original losses should still be backprop-able
    total = L_rec + L_gan
    total.backward()
    assert w.grad is not None


# ---------------------------------------------------------------------------
# Combined reconstruction loss — weighting, differentiability, guards
# ---------------------------------------------------------------------------

def test_reconstruction_loss_omega_L_scales_lpips(lpips_net):
    """Doubling omega_L should increase loss (LPIPS component grows)."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    loss_1x = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
        omega_L=1.0, omega_G=0.75,
    )
    loss_2x = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
        omega_L=2.0, omega_G=0.75,
    )
    # With omega_L=2, LPIPS term is doubled, so total should be larger
    assert loss_2x.item() > loss_1x.item()


def test_reconstruction_loss_omega_L_zero_is_l1_only(lpips_net):
    """With omega_L=0, reconstruction loss should equal pure L1."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)

    loss_combined = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
        omega_L=0.0, omega_G=0.75,
    )
    loss_l1_only = l1_loss(pred, target)
    assert loss_combined.item() == pytest.approx(loss_l1_only.item(), abs=1e-5)


def test_reconstruction_loss_differentiable_end_to_end(lpips_net):
    """Full reconstruction loss (with GAN) should be differentiable w.r.t.
    the simulated decoder parameter."""
    last_w = nn.Parameter(torch.ones(1))
    raw = torch.randn(2, 3, 64, 64)
    pred = raw * last_w
    target = torch.randn(2, 3, 64, 64)
    logits_fake = pred.mean(dim=[1, 2, 3], keepdim=True)

    loss = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=True, logits_fake=logits_fake,
        last_layer_weight=last_w,
        omega_L=1.0, omega_G=0.75,
    )
    loss.backward()
    assert last_w.grad is not None
    assert last_w.grad.abs().sum() > 0


def test_reconstruction_loss_asserts_on_missing_logits(lpips_net):
    """Should raise AssertionError if use_gan=True but logits_fake is None."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    with pytest.raises(AssertionError, match="logits_fake"):
        reconstruction_loss(
            pred, target, lpips_net,
            use_gan=True, logits_fake=None,
            last_layer_weight=torch.randn(10, requires_grad=True),
            omega_L=1.0, omega_G=0.75,
        )


def test_reconstruction_loss_asserts_on_missing_weight(lpips_net):
    """Should raise AssertionError if use_gan=True but last_layer_weight is None."""
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    with pytest.raises(AssertionError, match="last_layer_weight"):
        reconstruction_loss(
            pred, target, lpips_net,
            use_gan=True, logits_fake=torch.randn(2, 1),
            last_layer_weight=None,
            omega_L=1.0, omega_G=0.75,
        )


def test_reconstruction_loss_finite_output(lpips_net):
    """Combined loss should always be finite (no NaN or Inf)."""
    last_w = nn.Parameter(torch.ones(1))
    raw = torch.randn(2, 3, 64, 64)
    pred = raw * last_w
    target = torch.randn(2, 3, 64, 64)
    logits_fake = pred.mean(dim=[1, 2, 3], keepdim=True)

    loss = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=True, logits_fake=logits_fake,
        last_layer_weight=last_w,
        omega_L=1.0, omega_G=0.75,
    )
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# LPIPS network creation
# ---------------------------------------------------------------------------

def test_create_lpips_net_frozen():
    """create_lpips_net should return a fully frozen network."""
    net = create_lpips_net()
    for name, p in net.named_parameters():
        assert not p.requires_grad, f"LPIPS param {name} should be frozen"


def test_create_lpips_net_eval_mode():
    """create_lpips_net should be in eval mode."""
    net = create_lpips_net()
    assert not net.training


# ===========================================================================
# WAVE 3: Heavy integration, statistical, and stress tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Full pipeline integration: losses + discriminator together
# ---------------------------------------------------------------------------

from models.discriminator import PatchDiscriminator


class TestFullPipelineIntegration:
    """Test that loss functions work correctly with the real PatchDiscriminator.
    These tests simulate what actually happens during training."""

    @pytest.fixture(autouse=True)
    def setup(self, lpips_net):
        self.lpips_net = lpips_net
        self.disc = PatchDiscriminator(pretrained=False)

    def test_discriminator_loss_on_real_and_fake(self):
        """Discriminator hinge loss with actual PatchDiscriminator outputs."""
        real_images = torch.randn(4, 3, 224, 224)
        fake_images = torch.randn(4, 3, 224, 224)

        logits_real = self.disc(real_images)
        logits_fake = self.disc(fake_images)

        loss = gan_discriminator_loss(logits_real, logits_fake)
        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.item() >= 0  # hinge is always non-negative

    def test_discriminator_loss_backward_updates_head(self):
        """Discriminator loss backward should produce gradients for disc head only."""
        real_images = torch.randn(4, 3, 224, 224)
        fake_images = torch.randn(4, 3, 224, 224)

        logits_real = self.disc(real_images)
        logits_fake = self.disc(fake_images.detach())

        loss = gan_discriminator_loss(logits_real, logits_fake)
        loss.backward()

        # Head should get gradients
        for n, p in self.disc.named_parameters():
            if "head" in n and p.requires_grad:
                assert p.grad is not None, f"Head param {n} missing gradient"

        # Backbone should NOT get gradients
        for n, p in self.disc.named_parameters():
            if "head" not in n:
                assert p.grad is None or p.grad.abs().sum() == 0

    def test_generator_loss_with_disc_forward(self):
        """Generator loss using actual discriminator logits."""
        fake_images = torch.randn(4, 3, 224, 224, requires_grad=True)
        logits_fake = self.disc(fake_images)

        loss = gan_generator_loss(logits_fake)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def _disc_forward_with_grad(self, x):
        """Discriminator forward allowing gradient flow through backbone.

        During the generator step, we need gradients to flow from
        logits_fake back through the disc to the input images (and
        ultimately to the decoder's last_layer_weight for adaptive λ).

        The disc's forward() uses torch.no_grad() on the backbone as a
        memory optimization, but that severs the computation graph.
        The backbone's requires_grad=False already prevents parameter
        updates — torch.no_grad() is redundant for that purpose.

        In the real training loop (A.6), we'll call backbone + head
        separately for the generator step, same as here.
        """
        feat = self.disc.backbone(x)   # no torch.no_grad() wrapper
        return self.disc.head(feat)

    def test_full_recon_loss_with_disc(self):
        """Full reconstruction_loss with actual discriminator in the loop.
        Simulates the real generator-step forward pass."""
        last_w = nn.Parameter(torch.ones(1))
        raw = torch.randn(2, 3, 64, 64)
        pred = raw * last_w  # pred depends on last_w
        target = torch.randn(2, 3, 64, 64)

        # Generator step: need gradient flow through disc
        pred_224 = torch.nn.functional.interpolate(
            pred, size=224, mode="bilinear", align_corners=False
        )
        logits_fake = self._disc_forward_with_grad(pred_224)

        loss = reconstruction_loss(
            pred, target, self.lpips_net,
            use_gan=True, logits_fake=logits_fake,
            last_layer_weight=last_w,
            omega_L=1.0, omega_G=0.75,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)
        # Note: total can be negative because hinge GAN generator loss
        # is -logits_fake.mean(), which can dominate with random weights.

    def test_full_pipeline_gradient_flow(self):
        """Gradients should flow to decoder param through disc head,
        but NOT update disc backbone params."""
        last_w = nn.Parameter(torch.ones(1))
        raw = torch.randn(2, 3, 64, 64)
        pred = raw * last_w
        target = torch.randn(2, 3, 64, 64)

        pred_224 = torch.nn.functional.interpolate(
            pred, size=224, mode="bilinear", align_corners=False
        )
        logits_fake = self._disc_forward_with_grad(pred_224)

        loss = reconstruction_loss(
            pred, target, self.lpips_net,
            use_gan=True, logits_fake=logits_fake,
            last_layer_weight=last_w,
            omega_L=1.0, omega_G=0.75,
        )
        loss.backward()

        # Decoder param should receive gradient
        assert last_w.grad is not None
        assert last_w.grad.abs().sum() > 0

        # Backbone params should NOT receive gradient (requires_grad=False)
        for n, p in self.disc.named_parameters():
            if "head" not in n:
                assert p.grad is None or p.grad.abs().sum() == 0

    def test_separate_disc_and_gen_updates(self):
        """Simulate a real training step: separate disc and gen optimizer steps.
        This is the actual training pattern from Zheng et al.

        Disc step: uses disc.forward() (with torch.no_grad on backbone — fine,
        we don't need input gradients here).
        Gen step: uses backbone+head separately (need input gradients for
        adaptive λ to connect L_gan back to decoder's last_layer_weight).
        """
        last_w = nn.Parameter(torch.ones(1))
        raw = torch.randn(2, 3, 64, 64)
        pred = (raw * last_w).detach()  # stop grad for disc step
        target = torch.randn(2, 3, 64, 64)

        # --- Discriminator step (no input gradients needed) ---
        pred_224 = torch.nn.functional.interpolate(
            pred, size=224, mode="bilinear", align_corners=False
        )
        target_224 = torch.nn.functional.interpolate(
            target, size=224, mode="bilinear", align_corners=False
        )
        logits_real = self.disc(target_224)    # disc.forward() is fine here
        logits_fake = self.disc(pred_224)
        d_loss = gan_discriminator_loss(logits_real, logits_fake)

        assert d_loss.shape == ()
        assert d_loss.item() >= 0

        d_loss.backward()
        disc_head_grads = {
            n: p.grad.clone()
            for n, p in self.disc.named_parameters()
            if "head" in n and p.grad is not None
        }
        assert len(disc_head_grads) > 0

        # --- Generator step (need gradient flow through disc) ---
        self.disc.zero_grad()
        pred_gen = raw * last_w  # now with graph through last_w
        pred_gen_224 = torch.nn.functional.interpolate(
            pred_gen, size=224, mode="bilinear", align_corners=False
        )
        logits_fake_gen = self._disc_forward_with_grad(pred_gen_224)

        g_loss = reconstruction_loss(
            pred_gen, target, self.lpips_net,
            use_gan=True, logits_fake=logits_fake_gen,
            last_layer_weight=last_w,
            omega_L=1.0, omega_G=0.75,
        )
        g_loss.backward()

        # Decoder param gets gradient from gen step
        assert last_w.grad is not None


# ---------------------------------------------------------------------------
# Statistical robustness: many random seeds
# ---------------------------------------------------------------------------

class TestStatisticalRobustness:
    """Run loss computations across many random seeds to ensure
    no occasional NaN/Inf or sign errors."""

    @pytest.mark.parametrize("seed", range(50))
    def test_l1_loss_never_negative(self, seed):
        """L1 loss should never be negative across 50 random seeds."""
        torch.manual_seed(seed)
        pred = torch.randn(4, 3, 16, 16)
        target = torch.randn(4, 3, 16, 16)
        loss = l1_loss(pred, target)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("seed", range(50))
    def test_gan_disc_hinge_never_negative(self, seed):
        """Hinge loss should be non-negative across 50 random seeds."""
        torch.manual_seed(seed)
        real = torch.randn(8, 1)
        fake = torch.randn(8, 1)
        loss = gan_discriminator_loss(real, fake)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("seed", range(20))
    def test_adaptive_lambda_always_finite_and_bounded(self, seed):
        """Lambda should always be finite and in [0, 1e4] across 20 seeds."""
        torch.manual_seed(seed)
        w = torch.randn(32, requires_grad=True)
        # Random coefficient magnitudes
        a = torch.randn(1).abs().item() * 10 + 0.01
        b = torch.randn(1).abs().item() * 10 + 0.01
        L_rec = (w * a).sum()
        L_gan = (w * b).sum()
        lam = compute_adaptive_lambda(L_rec, L_gan, w)
        assert torch.isfinite(lam), f"Non-finite lambda at seed {seed}"
        assert 0 <= lam.item() <= 1e4, f"Lambda out of bounds at seed {seed}"

    @pytest.mark.parametrize("seed", range(10))
    def test_full_reconstruction_loss_always_finite(self, seed, lpips_net):
        """Full reconstruction loss should be finite across 10 seeds."""
        torch.manual_seed(seed)
        last_w = nn.Parameter(torch.randn(1))
        raw = torch.randn(2, 3, 64, 64)
        pred = raw * last_w
        target = torch.randn(2, 3, 64, 64)
        logits_fake = pred.mean(dim=[1, 2, 3], keepdim=True)

        loss = reconstruction_loss(
            pred, target, lpips_net,
            use_gan=True, logits_fake=logits_fake,
            last_layer_weight=last_w,
            omega_L=1.0, omega_G=0.75,
        )
        assert torch.isfinite(loss), f"Non-finite loss at seed {seed}"


# ---------------------------------------------------------------------------
# Numerical stability edge cases
# ---------------------------------------------------------------------------

class TestNumericalStability:
    """Test edge cases that could cause numerical issues in training."""

    def test_l1_loss_very_large_values(self):
        """L1 should handle large values without overflow."""
        pred = torch.ones(2, 3, 8, 8) * 1e6
        target = torch.zeros(2, 3, 8, 8)
        loss = l1_loss(pred, target)
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(1e6, rel=1e-5)

    def test_l1_loss_very_small_values(self):
        """L1 should handle tiny values without underflow to zero."""
        pred = torch.ones(2, 3, 8, 8) * 1e-7
        target = torch.zeros(2, 3, 8, 8)
        loss = l1_loss(pred, target)
        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_gan_gen_loss_extreme_logits(self):
        """Generator loss should be finite even with extreme logits."""
        large_pos = torch.ones(4, 1) * 1e4
        large_neg = torch.ones(4, 1) * -1e4
        assert torch.isfinite(gan_generator_loss(large_pos))
        assert torch.isfinite(gan_generator_loss(large_neg))

    def test_gan_disc_loss_extreme_logits(self):
        """Discriminator hinge loss should handle extreme logits (relu saturates)."""
        real = torch.ones(4, 1) * 1000
        fake = torch.ones(4, 1) * -1000
        loss = gan_discriminator_loss(real, fake)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)  # both saturated

    def test_adaptive_lambda_with_large_weight_tensor(self):
        """Lambda should work with weight tensors shaped like real decoder layers."""
        # Simulate a real decoder last layer: e.g. Linear(512, 768)
        w = torch.randn(768, 512, requires_grad=True)
        L_rec = (w ** 2).sum() * 0.001  # scaled to realistic range
        L_gan = (w.mean() * 2)
        lam = compute_adaptive_lambda(L_rec, L_gan, w)
        assert torch.isfinite(lam)
        assert 0 <= lam.item() <= 1e4

    def test_lpips_with_constant_images(self, lpips_net):
        """LPIPS on constant (single-color) images should not crash."""
        pred = torch.ones(1, 3, 64, 64) * 0.5
        target = torch.zeros(1, 3, 64, 64)
        loss = lpips_loss_fn(pred, target, lpips_net)
        assert torch.isfinite(loss)

    def test_lpips_with_clamped_images(self, lpips_net):
        """LPIPS should work when images are exactly at [0,1] boundaries."""
        pred = torch.ones(2, 3, 64, 64)   # all 1.0
        target = torch.zeros(2, 3, 64, 64)  # all 0.0
        loss = lpips_loss_fn(pred, target, lpips_net)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_reconstruction_loss_with_zero_pred(self, lpips_net):
        """Recon loss should handle all-zero predictions."""
        pred = torch.zeros(2, 3, 64, 64)
        target = torch.randn(2, 3, 64, 64)
        loss = reconstruction_loss(
            pred, target, lpips_net,
            use_gan=False, logits_fake=None, last_layer_weight=None,
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Gradient consistency checks
# ---------------------------------------------------------------------------

class TestGradientConsistency:
    """Verify gradient behavior matches mathematical expectations."""

    def test_l1_grad_direction_matches_sign(self):
        """L1 gradient at each element should match sign(pred - target)."""
        pred = torch.tensor([5.0, -3.0, 0.1, -0.1], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 0.0, 0.0])
        loss = l1_loss(pred, target)
        loss.backward()
        # sign(pred - target) / N
        expected_signs = torch.tensor([1.0, -1.0, 1.0, -1.0])
        actual_signs = pred.grad.sign()
        assert torch.equal(actual_signs, expected_signs)

    def test_gan_gen_grad_is_constant(self):
        """Generator loss gradient should be -1/N regardless of logit values."""
        for val in [-100, -1, 0, 1, 100]:
            logits = torch.full((8, 1), float(val), requires_grad=True)
            loss = gan_generator_loss(logits)
            loss.backward()
            expected = -1.0 / 8.0
            assert torch.allclose(
                logits.grad, torch.full_like(logits, expected), atol=1e-6
            ), f"Failed for logit value {val}"

    def test_disc_loss_grad_zero_inside_margin(self):
        """Disc hinge: when real > 1 and fake < -1, gradient should be zero
        (both relu terms are saturated at 0)."""
        real = torch.tensor([[5.0]], requires_grad=True)   # > 1: relu(1-5)=0
        fake = torch.tensor([[-5.0]], requires_grad=True)  # < -1: relu(1+(-5))=0
        loss = gan_discriminator_loss(real, fake)
        loss.backward()
        assert real.grad.item() == pytest.approx(0.0, abs=1e-6)
        assert fake.grad.item() == pytest.approx(0.0, abs=1e-6)

    def test_disc_loss_grad_nonzero_outside_margin(self):
        """Disc hinge: when real < 1 or fake > -1, gradient should be non-zero."""
        real = torch.tensor([[0.0]], requires_grad=True)   # < 1: relu(1-0)=1
        fake = torch.tensor([[0.0]], requires_grad=True)   # > -1: relu(1+0)=1
        loss = gan_discriminator_loss(real, fake)
        loss.backward()
        # d/d(real) of relu(1-real).mean() = -1/N when real < 1
        assert real.grad.item() == pytest.approx(-1.0, abs=1e-6)
        # d/d(fake) of relu(1+fake).mean() = 1/N when fake > -1
        assert fake.grad.item() == pytest.approx(1.0, abs=1e-6)

    def test_adaptive_lambda_gradient_ratio_is_correct(self):
        """Verify λ actually equals ‖∇_w L_rec‖ / ‖∇_w L_gan‖ by hand."""
        w = torch.tensor([3.0, 4.0], requires_grad=True)
        L_rec = (w * 2).sum()   # grad = [2, 2], norm = 2*sqrt(2)
        L_gan = (w * 1).sum()   # grad = [1, 1], norm = sqrt(2)
        lam = compute_adaptive_lambda(L_rec, L_gan, w)
        # ratio = 2*sqrt(2) / sqrt(2) = 2.0
        assert lam.item() == pytest.approx(2.0, abs=1e-5)
