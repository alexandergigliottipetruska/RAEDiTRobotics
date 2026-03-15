"""Stage3PolicyWrapper — C.8.

Wraps PolicyDiT for the evaluation harness interface (rollout.py).
Handles:
  - EMA weight swapping (via context manager)
  - Image preprocessing (uint8 -> float32 -> ImageNet normalize)
  - DDIM inference via policy.predict_action()
  - Output format: (T_p, ac_dim) numpy, removing batch dim

Usage:
    wrapper = Stage3PolicyWrapper(policy, ema, device="cuda")
    # Used inside evaluate_policy():
    actions = wrapper.predict(images, proprio, view_present)
"""

import torch
import torch.nn as nn
import numpy as np

# ImageNet stats for torch-native normalization (CHW format)
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3, 1, 1)


def _imagenet_normalize_torch(images: torch.Tensor) -> torch.Tensor:
    """ImageNet normalize a torch tensor already in (B, T_o, K, 3, H, W) CHW float [0,1]."""
    mean = _IMAGENET_MEAN.to(images.device)
    std = _IMAGENET_STD.to(images.device)
    return (images - mean) / std


class Stage3PolicyWrapper:
    """Wraps PolicyDiT for the evaluation harness.

    Args:
        policy: PolicyDiT instance.
        ema: EMA instance (optional). If provided, uses EMA weights for inference.
        device: Device to run inference on.
    """

    def __init__(self, policy, ema=None, device="cpu"):
        self.policy = policy
        self.ema = ema
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()

    def predict(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        view_present: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference for the evaluation harness.

        Args:
            images: (1, T_o, K, 3, H, W) uint8 or float images.
            proprio: (1, T_o, D_prop) normalized proprio.
            view_present: (1, K) bool.

        Returns:
            actions: (T_p, ac_dim) normalized actions (no batch dim).
        """
        # Preprocess images: uint8 -> float32 [0,1] -> ImageNet normalize
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        images = _imagenet_normalize_torch(images)

        proprio = proprio.float().to(self.device)
        view_present = view_present.bool().to(self.device)

        obs = {
            "images_enc": images,
            "proprio": proprio,
            "view_present": view_present,
        }

        if self.ema is not None:
            with self.ema.averaged_model():
                actions = self.policy.predict_action(obs)
        else:
            actions = self.policy.predict_action(obs)

        # Remove batch dim: (1, T_p, ac_dim) -> (T_p, ac_dim)
        return actions[0]
