"""Phased RAE training loop (Phase A.6).

Implements the three-phase training schedule from Zheng et al. 2025:
  Phase 1 (epochs 0-E_d): L1 + LPIPS only
  Phase 2 (epochs E_d-E_g): + discriminator training
  Phase 3 (epochs E_g-end): + GAN loss for decoder

Separate optimizers for decoder/adapter vs discriminator head.
Hyperparameters from Zheng et al. Table 12.

Owner: Swagman (scaffold), other team member (integration)
"""

raise NotImplementedError("A.6: train_stage1.py — to be implemented")
