"""BasePolicy — C.1 (Other member).

Minimal abstract base class for all robot manipulation policies.

Concrete subclasses (e.g. PolicyDiT in C.10) must implement:
  - compute_loss(batch) → scalar Tensor for training
  - predict_action(obs)  → action Tensor for inference
"""

import torch
import torch.nn as nn


class BasePolicy(nn.Module):
    """Abstract base class for diffusion-based manipulation policies.

    Accepts **kwargs so subclasses like DiffusionTransformerAgent can pass
    visual encoder config (features, feat_dim, n_cams, etc.) without crashing.
    These are stored in self._base_config for introspection but not used here —
    Stage1Bridge handles encoding in our pipeline.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._base_config = kwargs

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute training loss for a batch.

        Args:
            batch: Dict with keys: images, proprio, actions, mask, view_present

        Returns:
            Scalar loss tensor.
        """
        raise NotImplementedError

    def predict_action(self, obs: dict) -> torch.Tensor:
        """Run inference and return a predicted action chunk.

        Args:
            obs: Dict with observation tensors (images, proprio, view_present).

        Returns:
            Action tensor of shape (B, T_pred, action_dim).
        """
        raise NotImplementedError
