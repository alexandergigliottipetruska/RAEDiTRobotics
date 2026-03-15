"""Exponential Moving Average wrapper for model parameters (C.5).

Usage:
    ema = EMA(model, decay=0.9999)

    # After each optimizer step:
    ema.update()

    # For evaluation with EMA weights:
    with ema.averaged_model():
        preds = model(batch)

    # Save / restore:
    ckpt = {'ema': ema.state_dict(), 'model': model.state_dict()}
    ema.load_state_dict(ckpt['ema'])
"""

import contextlib
import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average of a model's parameters.

    Shadow parameters are kept on CPU to avoid GPU memory overhead.
    They are moved to the model's device only when `apply_shadow()` is called.

    Args:
        model: The nn.Module whose parameters to track.
        decay: EMA decay rate. Typical values: 0.999–0.9999.
               Higher = slower update = smoother average.
        warmup_steps: Number of steps before decay reaches its target value.
                      During warmup, effective decay = min(decay, (1+step)/(10+step)).
                      Set to 0 to disable warmup.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 100):
        self.model        = model
        self.decay        = decay
        self.warmup_steps = warmup_steps
        self._step        = 0

        # Deep-copy current parameters as shadow (detached, CPU)
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def _effective_decay(self) -> float:
        if self.warmup_steps > 0:
            warmup_decay = (1 + self._step) / (10 + self._step)
            return min(self.decay, warmup_decay)
        return self.decay

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters from the current model state.

        Call once after each optimizer step.
        """
        self._step += 1
        d = self._effective_decay()
        model_state = self.model.state_dict()
        for k in self.shadow:
            self.shadow[k] = (
                d * self.shadow[k] + (1.0 - d) * model_state[k].detach().cpu()
            )

    def apply_to(self, model: nn.Module) -> None:
        """Load shadow parameters into *model* in-place.

        Args:
            model: The nn.Module to overwrite with EMA weights.
                   Typically ``self.model``, but can be any compatible module.
        """
        device = next(model.parameters()).device
        shadow_on_device = {k: v.to(device) for k, v in self.shadow.items()}
        model.load_state_dict(shadow_on_device)

    def restore(self, model: nn.Module, backup: dict) -> None:
        """Restore *model* from a previously saved state dict.

        Args:
            model:  The nn.Module to restore.
            backup: State dict returned by ``model.state_dict()`` before
                    ``apply_to`` was called.
        """
        model.load_state_dict(backup)

    @contextlib.contextmanager
    def averaged_model(self):
        """Context manager: temporarily swap in EMA weights for evaluation.

        Example::
            with ema.averaged_model():
                loss = evaluate(model, val_loader)
        """
        backup = copy.deepcopy(self.model.state_dict())
        self.apply_to(self.model)
        try:
            yield
        finally:
            self.restore(self.model, backup)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "shadow":       self.shadow,
            "decay":        self.decay,
            "warmup_steps": self.warmup_steps,
            "_step":        self._step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow       = state["shadow"]
        self.decay        = state.get("decay",        self.decay)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self._step        = state.get("_step",        0)
