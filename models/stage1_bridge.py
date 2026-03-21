"""Stage1Bridge — C.2.

Loads Stage 1 checkpoint (encoder, adapter, optionally decoder) and provides
the token extraction pipeline for Stage 3 policy training.

Usage:
    bridge = Stage1Bridge("checkpoints/stage1.pt", device="cuda")
    adapted = bridge.encode(images_enc, view_present)
    # adapted: (B, K, N, d') tensor of adapted visual tokens

For co-training (lambda_recon > 0):
    loss_recon = bridge.compute_recon_loss(adapted, images_target, view_present)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import FrozenMultiViewEncoder
from models.adapter import TrainableAdapter
from models.decoder import ViTDecoder

log = logging.getLogger(__name__)


class Stage1Bridge(nn.Module):
    """Loads Stage 1 components and provides adapted tokens from images.

    The encoder is always frozen. The adapter is trainable by default (receives
    policy gradients in Stage 3, optionally anchored by reconstruction loss).
    The decoder is loaded only when lambda_recon > 0 (co-training).

    Args:
        checkpoint_path: Path to Stage 1 checkpoint (.pt file).
            If empty string, uses randomly initialized weights (for testing).
        pretrained_encoder: Use real DINOv3 encoder (True) or mock (False).
        load_decoder: Load the decoder for reconstruction co-training.
        trainable_adapter: Whether adapter receives gradients.
    """

    # ImageNet normalization constants
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        checkpoint_path: str = "",
        pretrained_encoder: bool = True,
        load_decoder: bool = False,
        trainable_adapter: bool = True,
    ):
        super().__init__()

        # Encoder: always frozen
        self.encoder = FrozenMultiViewEncoder(pretrained=pretrained_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        # Cancel-Affine LayerNorm (matches Stage 1)
        self.cancel_affine_ln = nn.LayerNorm(1024, elementwise_affine=False)

        # Adapter: trainable by default
        self.adapter = TrainableAdapter()
        self.adapter.requires_grad_(trainable_adapter)

        # Decoder: optional, for co-training
        self.decoder = None
        self._lpips_net = None
        if load_decoder:
            self.decoder = ViTDecoder()

        # Load checkpoint weights
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> None:
        """Load Stage 1 checkpoint into adapter and optionally decoder."""
        ckpt = torch.load(path, weights_only=False, map_location="cpu")

        # Strip torch.compile prefix if present
        def _strip(sd):
            prefix = "_orig_mod."
            if any(k.startswith(prefix) for k in sd):
                return {k.removeprefix(prefix): v for k, v in sd.items()}
            return sd

        self.adapter.load_state_dict(_strip(ckpt["adapter"]))
        log.info("Loaded adapter from %s", path)

        if self.decoder is not None and "decoder" in ckpt:
            self.decoder.load_state_dict(_strip(ckpt["decoder"]))
            log.info("Loaded decoder from %s", path)

    @torch.no_grad()
    def encode_frozen(self, images: torch.Tensor) -> torch.Tensor:
        """Run frozen encoder on images.

        Args:
            images: (B, 3, 224, 224) ImageNet-normalized.

        Returns:
            Raw tokens: (B, 196, 1024).
        """
        return self.encoder(images)

    def encode(
        self,
        images_enc: torch.Tensor,
        view_present: torch.Tensor,
        pre_normalized: bool = False,
    ) -> torch.Tensor:
        """Full pipeline: resize -> ImageNet norm -> frozen encoder -> LN -> adapter.

        Args:
            images_enc: (B, T_o, K, 3, H, W) images.
                If pre_normalized=True: already ImageNet-normalized (from Stage3Dataset).
                If pre_normalized=False: float [0,1] (from eval env wrapper).
            view_present: (B, K) bool mask of real cameras.
            pre_normalized: Whether images are already ImageNet-normalized.
                Stage3Dataset returns pre-normalized images (True).
                Eval wrapper returns float [0,1] images (False).

        Returns:
            adapted: (B, T_o, K, N, d') adapted tokens.
                Absent views (view_present=False) are zero-filled.
        """
        B, T_o, K, C, H, W = images_enc.shape
        N = 196  # 14x14 patches
        d_prime = 512  # adapter output dim
        device = images_enc.device

        adapted = None  # lazily created to match adapter output dtype (handles autocast)

        for t in range(T_o):
            for k in range(K):
                # Only encode views that are present (at least one sample has it)
                mask = view_present[:, k]  # (B,)
                if not mask.any():
                    continue

                imgs_k = images_enc[mask, t, k]  # (B_real, 3, H, W)

                # Resize to 224x224 if needed (env returns 84x84)
                if imgs_k.shape[-1] != 224 or imgs_k.shape[-2] != 224:
                    imgs_k = F.interpolate(
                        imgs_k, size=(224, 224), mode='bilinear', align_corners=False
                    )

                # Chi's normalization: [0,1] → [-1,1] → ImageNet norm
                if not pre_normalized:
                    imgs_k = imgs_k * 2.0 - 1.0  # [0,1] → [-1,1]
                    mean = torch.tensor(self._IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                    std = torch.tensor(self._IMAGENET_STD, device=device).view(1, 3, 1, 1)
                    imgs_k = (imgs_k - mean) / std

                with torch.no_grad():
                    raw = self.encoder(imgs_k)  # (B_real, 196, 1024)
                    normed = self.cancel_affine_ln(raw)

                # Adapter is trainable — gradient flows here
                tokens = self.adapter(normed)  # (B_real, 196, 512)

                # Create output tensor on first adapter call (matches autocast dtype)
                if adapted is None:
                    adapted = torch.zeros(B, T_o, K, N, d_prime,
                                          device=device, dtype=tokens.dtype)

                adapted[mask, t, k] = tokens

        # If no views were present at all, return float32 zeros
        if adapted is None:
            adapted = torch.zeros(B, T_o, K, N, d_prime, device=device)

        return adapted

    def adapt(
        self,
        cached_tokens: torch.Tensor,
        view_present: torch.Tensor,
    ) -> torch.Tensor:
        """Run adapter on precomputed encoder tokens (skips encoder + LN).

        Args:
            cached_tokens: (B, T_o, K, 196, 1024) precomputed post-encoder/LN tokens.
            view_present: (B, K) bool mask of real cameras.

        Returns:
            adapted: (B, T_o, K, N, d') adapted tokens.
        """
        B, T_o, K, N, d_enc = cached_tokens.shape
        d_prime = 512
        device = cached_tokens.device

        adapted = None

        for t in range(T_o):
            for k in range(K):
                mask = view_present[:, k]
                if not mask.any():
                    continue

                tokens_k = cached_tokens[mask, t, k]  # (B_real, 196, 1024)
                tokens = self.adapter(tokens_k)  # (B_real, 196, 512)

                if adapted is None:
                    adapted = torch.zeros(B, T_o, K, N, d_prime,
                                          device=device, dtype=tokens.dtype)
                adapted[mask, t, k] = tokens

        if adapted is None:
            adapted = torch.zeros(B, T_o, K, N, d_prime, device=device)

        return adapted

    def compute_recon_loss(
        self,
        adapted: torch.Tensor,
        images_target: torch.Tensor,
        view_present: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss for co-training.

        Uses L1 + LPIPS (no GAN in Stage 3).

        Args:
            adapted: (B, T_o, K, N, d') adapted tokens (before noise augment).
            images_target: (B, T_o, K, 3, H, W) raw [0,1] target images.
            view_present: (B, K) bool.

        Returns:
            Scalar reconstruction loss.
        """
        if self.decoder is None:
            raise RuntimeError("Decoder not loaded. Set load_decoder=True.")

        # Lazy init LPIPS on first call
        if self._lpips_net is None:
            from models.losses import create_lpips_net
            device = next(self.decoder.parameters()).device
            self._lpips_net = create_lpips_net().to(device)

        from models.losses import l1_loss, lpips_loss_fn

        B, T_o, K, C, H, W = images_target.shape
        total_loss = torch.tensor(0.0, device=adapted.device)
        n_views = 0

        for t in range(T_o):
            for k in range(K):
                mask = view_present[:, k]
                if not mask.any():
                    continue

                tokens_k = adapted[mask, t, k]  # (B_real, N, d')
                target_k = images_target[mask, t, k]  # (B_real, 3, H, W)

                pred_k = self.decoder(tokens_k)  # (B_real, 3, H, W) in [0,1]

                loss_l1 = l1_loss(pred_k, target_k)
                loss_lpips = lpips_loss_fn(pred_k, target_k, self._lpips_net)
                total_loss = total_loss + loss_l1 + loss_lpips
                n_views += 1

        if n_views > 0:
            total_loss = total_loss / n_views

        return total_loss

    @property
    def last_layer_weight(self) -> torch.Tensor | None:
        """Expose decoder's last layer weight for adaptive lambda (if needed)."""
        if self.decoder is not None:
            return self.decoder.last_layer_weight
        return None
