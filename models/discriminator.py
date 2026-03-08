"""Frozen DINO-S/8 patch discriminator (Phase A.4).

Frozen DINO-S/8 backbone + trainable classification head.
Provides adversarial signal for RAE decoder training.

Architecture (from Zheng et al. 2025, Appendix C.2):
  - Backbone: DINO ViT-S/8 (21M params, frozen, eval mode)
  - Head: Linear(384, 256) -> LeakyReLU(0.2) -> Linear(256, 1)
  - Input images interpolated to 224x224 before backbone

Mock mode (pretrained=False): uses a lightweight frozen feature
extractor for local CPU testing without downloading weights.
"""

import torch
import torch.nn as nn


class _MockBackbone(nn.Module):
    """Lightweight frozen substitute for DINO-S/8 in mock mode.

    Flattens + projects image to 384-d feature vector.
    All parameters frozen. Deterministic output (no randomness).
    """

    def __init__(self):
        super().__init__()
        # AdaptiveAvgPool to fixed spatial size, then flatten + project
        self.pool = nn.AdaptiveAvgPool2d(4)       # (B, 3, 4, 4) = 48-d
        self.proj = nn.Linear(3 * 4 * 4, 384)

    def forward(self, x):
        x = self.pool(x)
        x = x.flatten(1)           # (B, 48)
        return self.proj(x)         # (B, 384)


class PatchDiscriminator(nn.Module):
    """Frozen DINO-S/8 backbone with trainable classification head.

    Args:
        pretrained: If True, load real DINO-S/8 via torch.hub.
                    If False, use mock backbone for testing.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.backbone = torch.hub.load(
                "facebookresearch/dino", "dino_vits8"
            )
        else:
            self.backbone = _MockBackbone()

        # Freeze backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Trainable classification head (Zheng et al. Appendix C.2)
        self.head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: images (B, 3, 224, 224)

        Returns:
            logits: (B, 1)
        """
        with torch.no_grad():
            feat = self.backbone(x)     # (B, 384)
        return self.head(feat)

    def train(self, mode: bool = True):
        """Override to keep backbone in eval mode always."""
        super().train(mode)
        self.backbone.eval()
        return self
