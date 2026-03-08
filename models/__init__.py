"""Stage 1 RAE model components.

Modules:
    encoder        - Frozen DINOv3-L encoder wrapper (A.1)
    adapter        - Trainable token adapter (A.2)
    decoder        - ViT-based RAE decoder (A.3)
    discriminator  - Frozen DINO-S/8 patch discriminator (A.4)
    losses         - Reconstruction + GAN losses with adaptive lambda (A.5)
"""
