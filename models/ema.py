"""C.5 EMAModel — Exponential Moving Average of model parameters.

Standard for diffusion models. Keeps a shadow copy of weights updated as:
    shadow = decay * shadow + (1 - decay) * param

Use apply_to() to swap EMA weights into the model for evaluation,
then restore() to put the original weights back for training.
"""

import copy

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate. 0.9999 is standard for diffusion models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow_params = [p.data.clone() for p in model.parameters()]
        self._backup = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow params: shadow = decay * shadow + (1 - decay) * param."""
        for s, p in zip(self.shadow_params, model.parameters()):
            s.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into model for evaluation. Backs up originals."""
        self._backup = [p.data.clone() for p in model.parameters()]
        for s, p in zip(self.shadow_params, model.parameters()):
            p.data.copy_(s)

    def restore(self, model: nn.Module) -> None:
        """Restore original weights after evaluation."""
        if self._backup is None:
            raise RuntimeError("restore() called without prior apply_to()")
        for bk, p in zip(self._backup, model.parameters()):
            p.data.copy_(bk)
        self._backup = None

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_params": [p.clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        self.shadow_params = [p.clone() for p in state_dict["shadow_params"]]
