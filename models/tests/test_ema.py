"""Tests for C.5 EMAModel.

Exponential Moving Average of model parameters for diffusion policy evaluation.
"""

import copy

import pytest
import torch
import torch.nn as nn

from models.ema import EMAModel


def _make_model():
    """Simple 2-layer model for testing."""
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))


class TestEMABasic:
    def test_init_copies_weights(self):
        """EMA weights match the model at initialization."""
        model = _make_model()
        ema = EMAModel(model, decay=0.999)
        for p_ema, p_model in zip(ema.shadow_params, model.parameters()):
            assert torch.equal(p_ema, p_model.data)

    def test_decay_stored(self):
        """Decay value is stored correctly."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9999)
        assert ema.decay == 0.9999

    def test_update_moves_weights(self):
        """After an update step, EMA weights differ from both old and new model."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9)

        old_ema = [p.clone() for p in ema.shadow_params]
        # Simulate optimizer step
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)

        for p_ema, p_old in zip(ema.shadow_params, old_ema):
            assert not torch.equal(p_ema, p_old), "EMA should have changed"

    def test_decay_formula(self):
        """EMA update follows: shadow = decay * shadow + (1-decay) * param."""
        model = _make_model()
        decay = 0.9
        ema = EMAModel(model, decay=decay)
        old_shadow = [p.clone() for p in ema.shadow_params]

        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        ema.update(model)

        for p_ema, p_old, p_model in zip(ema.shadow_params, old_shadow, model.parameters()):
            expected = decay * p_old + (1 - decay) * p_model.data
            assert torch.allclose(p_ema, expected, atol=1e-6)

    def test_weights_diverge_over_steps(self):
        """EMA weights progressively diverge from model after many updates."""
        model = _make_model()
        ema = EMAModel(model, decay=0.99)

        for _ in range(50):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update(model)

        # EMA should trail behind the model
        diffs = []
        for p_ema, p_model in zip(ema.shadow_params, model.parameters()):
            diffs.append((p_ema - p_model.data).abs().mean().item())
        assert max(diffs) > 0.001, "EMA should diverge from fast-moving model"


class TestEMASaveLoad:
    def test_state_dict_roundtrip(self):
        """Save and load preserves EMA weights exactly."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9999)
        # Do a few updates
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)

        state = ema.state_dict()
        # Create fresh EMA and load
        model2 = _make_model()
        ema2 = EMAModel(model2, decay=0.9999)
        ema2.load_state_dict(state)

        for p1, p2 in zip(ema.shadow_params, ema2.shadow_params):
            assert torch.equal(p1, p2)

    def test_state_dict_contains_decay(self):
        """State dict stores the decay value."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9999)
        state = ema.state_dict()
        assert "decay" in state
        assert state["decay"] == 0.9999


class TestEMAApplyRestore:
    def test_apply_copies_ema_to_model(self):
        """apply_to() copies EMA weights into the model."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9)
        # Diverge model from EMA
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        ema.update(model)

        ema.apply_to(model)
        for p_ema, p_model in zip(ema.shadow_params, model.parameters()):
            assert torch.equal(p_ema, p_model.data)

    def test_restore_recovers_original(self):
        """restore() puts back the original model weights after apply_to()."""
        model = _make_model()
        ema = EMAModel(model, decay=0.9)
        # Diverge
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p))
        original_weights = [p.data.clone() for p in model.parameters()]
        ema.update(model)

        ema.apply_to(model)
        ema.restore(model)

        for p_model, p_orig in zip(model.parameters(), original_weights):
            assert torch.equal(p_model.data, p_orig)

    def test_apply_restore_cycle(self):
        """Multiple apply/restore cycles work correctly."""
        model = _make_model()
        ema = EMAModel(model, decay=0.99)
        for _ in range(5):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 0.01)
            ema.update(model)

        original = [p.data.clone() for p in model.parameters()]

        # Cycle 1
        ema.apply_to(model)
        ema.restore(model)
        for p, o in zip(model.parameters(), original):
            assert torch.equal(p.data, o)

        # Cycle 2
        ema.apply_to(model)
        ema.restore(model)
        for p, o in zip(model.parameters(), original):
            assert torch.equal(p.data, o)
