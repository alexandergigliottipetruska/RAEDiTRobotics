"""Reconstruction + GAN losses with adaptive lambda (Phase A.5).

Loss components (Zheng et al. 2025, Appendix C.2, Table 12):
  L_recon = L_ℓ1 + ω_L · L_LPIPS + ω_G · λ · L_GAN

  - L_ℓ1:   pixel-wise L1 between reconstruction and target
  - L_LPIPS: perceptual loss (frozen VGG, images in [-1,1])
  - L_GAN:   non-saturating generator loss = -mean(logits_fake)
  - L_disc:  hinge discriminator loss (separate optimizer)
  - λ:       adaptive balancing (VQGAN, Esser et al. 2021)
             = ‖∇_w L_rec‖ / (‖∇_w L_GAN‖ + ε), clamped [0, 1e4]
             computed w.r.t. decoder's last layer weight

Constants: ω_L = 1.0, ω_G = 0.75 (Zheng et al. 2025, Appendix C.2, Table 12).
"""

import torch
import torch.nn.functional as F
import lpips


# ---------------------------------------------------------------------------
# LPIPS network creation
# ---------------------------------------------------------------------------

def create_lpips_net() -> lpips.LPIPS:
    """Create frozen LPIPS perceptual loss network (VGG backbone).

    Returns an lpips.LPIPS module with all parameters frozen.
    The network expects inputs in [-1, 1] range.
    """
    net = lpips.LPIPS(net="vgg")
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------

def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise L1 loss.

    Args:
        pred:   (B, 3, H, W) reconstructed images
        target: (B, 3, H, W) ground-truth images

    Returns:
        Scalar L1 loss.
    """
    return F.l1_loss(pred, target)


def lpips_loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_net: lpips.LPIPS,
) -> torch.Tensor:
    """Perceptual loss via LPIPS (frozen VGG).

    Converts images from [0, 1] to [-1, 1] as LPIPS expects.

    Args:
        pred:      (B, 3, H, W) reconstructed images in [0, 1]
        target:    (B, 3, H, W) ground-truth images in [0, 1]
        lpips_net: frozen LPIPS network from create_lpips_net()

    Returns:
        Scalar LPIPS loss (mean over batch).
    """
    return lpips_net(pred * 2 - 1, target * 2 - 1).mean()


def gan_generator_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss.

    Generator wants discriminator to score fakes highly.

    Args:
        logits_fake: (B, 1) discriminator logits on generated images

    Returns:
        Scalar generator loss = -mean(logits_fake).
    """
    return -logits_fake.mean()


def gan_discriminator_loss(
    logits_real: torch.Tensor,
    logits_fake: torch.Tensor,
) -> torch.Tensor:
    """Hinge discriminator loss.

    Args:
        logits_real: (B, 1) discriminator logits on real images
        logits_fake: (B, 1) discriminator logits on generated images

    Returns:
        Scalar hinge loss (always >= 0).
    """
    return (F.relu(1.0 - logits_real).mean()
            + F.relu(1.0 + logits_fake).mean())


# ---------------------------------------------------------------------------
# Adaptive lambda (VQGAN, Esser et al. 2021)
# ---------------------------------------------------------------------------

def compute_adaptive_lambda(
    L_rec: torch.Tensor,
    L_gan: torch.Tensor,
    last_layer_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Adaptive λ balancing reconstruction and GAN gradients.

    λ = ‖∇_w L_rec‖ / (‖∇_w L_GAN‖ + ε)

    Computed w.r.t. the decoder's last layer weight to balance
    how much the GAN loss influences the decoder relative to
    reconstruction.

    Reference: taming-transformers/taming/modules/losses/vqperceptual.py

    Args:
        L_rec:              scalar reconstruction loss (L1 + LPIPS)
        L_gan:              scalar generator loss
        last_layer_weight:  decoder.head.weight (parameter tensor)
        eps:                numerical stability

    Returns:
        Detached scalar λ, clamped to [0, 1e4].
    """
    rec_grads = torch.autograd.grad(
        L_rec, last_layer_weight, retain_graph=True
    )[0]
    gan_grads = torch.autograd.grad(
        L_gan, last_layer_weight, retain_graph=True
    )[0]
    lam = torch.norm(rec_grads) / (torch.norm(gan_grads) + eps)
    return torch.clamp(lam, 0.0, 1e4).detach()


# ---------------------------------------------------------------------------
# Combined reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_net: lpips.LPIPS,
    *,
    use_gan: bool = False,
    logits_fake: torch.Tensor | None = None,
    last_layer_weight: torch.Tensor | None = None,
    omega_L: float = 1.0,
    omega_G: float = 0.75,
) -> torch.Tensor:
    """Combined Stage 1 reconstruction loss.

    Without GAN (epochs < E_g):
        L = L_ℓ1 + ω_L · L_LPIPS

    With GAN (epochs >= E_g):
        L = L_ℓ1 + ω_L · L_LPIPS + ω_G · λ · L_GAN

    Args:
        pred:               (B, 3, H, W) reconstructed images
        target:             (B, 3, H, W) ground-truth images
        lpips_net:          frozen LPIPS network
        use_gan:            whether to include GAN term
        logits_fake:        (B, 1) discriminator logits on pred
        last_layer_weight:  decoder's last layer weight for adaptive λ
        omega_L:            LPIPS weight (default 1.0)
        omega_G:            GAN weight before adaptive λ (default 0.75)

    Returns:
        Scalar total reconstruction loss.
    """
    L_l1 = l1_loss(pred, target)
    L_lpips = lpips_loss_fn(pred, target, lpips_net)
    L_rec = L_l1 + omega_L * L_lpips

    if not use_gan:
        return L_rec

    assert logits_fake is not None, "logits_fake required when use_gan=True"
    assert last_layer_weight is not None, "last_layer_weight required when use_gan=True"

    L_gan = gan_generator_loss(logits_fake)
    lam = compute_adaptive_lambda(L_rec, L_gan, last_layer_weight)
    return L_rec + omega_G * lam * L_gan
