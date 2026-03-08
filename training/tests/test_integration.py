"""End-to-end integration tests for Stage 1 RAE pipeline.

Uses REAL encoder (mock backbone), adapter, decoder, discriminator (mock backbone),
and loss functions wired through train_step and validate.
No mocks except the frozen backbones (no HF download needed).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.encoder import FrozenMultiViewEncoder
from models.adapter import TrainableAdapter
from models.decoder import ViTDecoder
from models.discriminator import PatchDiscriminator
from models.losses import (
    l1_loss, lpips_loss_fn, create_lpips_net,
    gan_generator_loss, gan_discriminator_loss, compute_adaptive_lambda,
)
from training.train_stage1 import (
    Stage1Config, disc_forward_with_grad, train_step, validate,
    save_checkpoint, load_checkpoint,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def encoder(device):
    return FrozenMultiViewEncoder(pretrained=False).to(device)


@pytest.fixture(scope="module")
def adapter(device):
    return TrainableAdapter().to(device)


@pytest.fixture(scope="module")
def decoder(device):
    return ViTDecoder().to(device)


@pytest.fixture(scope="module")
def disc(device):
    return PatchDiscriminator(pretrained=False).to(device)


@pytest.fixture(scope="module")
def lpips_net(device):
    return create_lpips_net().to(device)


@pytest.fixture
def config():
    return Stage1Config(
        batch_size=2,
        num_workers=0,
        epoch_start_disc=1,
        epoch_start_gan=2,
        disc_pretrained=False,
    )


def _make_batch(B=2, K=4, device="cpu"):
    """Create a fake batch matching Stage1Dataset output format."""
    images_enc = torch.randn(B, K, 3, 224, 224, device=device)
    images_target = torch.rand(B, K, 3, 224, 224, device=device)  # [0,1]
    view_present = torch.ones(B, K, dtype=torch.bool, device=device)
    return {
        "images_enc": images_enc,
        "images_target": images_target,
        "view_present": view_present,
    }


# ── Forward pipeline tests ──────────────────────────────────────────

class TestForwardPipeline:
    """Verify shapes connect: encoder -> adapter -> decoder."""

    def test_encoder_to_adapter_shape(self, encoder, adapter, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        assert tokens.shape == (2, 196, 1024)
        adapted = adapter(tokens)
        assert adapted.shape == (2, 196, 512)

    def test_adapter_to_decoder_shape(self, adapter, decoder, device):
        tokens = torch.randn(2, 196, 1024, device=device)
        adapted = adapter(tokens)
        pred = decoder(adapted)
        assert pred.shape == (2, 3, 224, 224)

    def test_full_pipeline_shape(self, encoder, adapter, decoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        adapted = adapter(tokens)
        pred = decoder(adapted)
        assert pred.shape == (2, 3, 224, 224)

    def test_decoder_output_range(self, encoder, adapter, decoder, device):
        x = torch.randn(4, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        adapted = adapter(tokens)
        pred = decoder(adapted)
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_noise_augment_in_pipeline(self, encoder, adapter, decoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        adapted = adapter(tokens)
        noisy = adapter.noise_augment(adapted)
        pred = decoder(noisy)
        assert pred.shape == (2, 3, 224, 224)
        assert not torch.allclose(adapted, noisy)


# ── Loss pipeline tests ─────────────────────────────────────────────

class TestLossPipeline:
    """Verify losses work with real component outputs."""

    def test_l1_loss_on_real_output(self, encoder, adapter, decoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        target = torch.rand(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        loss = l1_loss(pred, target)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_lpips_loss_on_real_output(self, encoder, adapter, decoder, lpips_net, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        target = torch.rand(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        loss = lpips_loss_fn(pred, target, lpips_net)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_gan_loss_with_disc(self, encoder, adapter, decoder, disc, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        logits = disc_forward_with_grad(disc, pred)
        g_loss = gan_generator_loss(logits)
        assert g_loss.ndim == 0

    def test_disc_loss_real_vs_fake(self, encoder, adapter, decoder, disc, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        target = torch.rand(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        logits_real = disc(target)
        logits_fake = disc(pred.detach())
        d_loss = gan_discriminator_loss(logits_real, logits_fake)
        assert d_loss.ndim == 0

    def test_adaptive_lambda_with_real_decoder(self, encoder, adapter, decoder, disc, lpips_net, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        target = torch.rand(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        L_rec = l1_loss(pred, target) + lpips_loss_fn(pred, target, lpips_net)
        logits_fake = disc_forward_with_grad(disc, pred)
        L_gan = gan_generator_loss(logits_fake)
        lam = compute_adaptive_lambda(L_rec, L_gan, decoder.last_layer_weight)
        assert lam.ndim == 0
        assert lam.item() >= 0


# ── Gradient flow tests ─────────────────────────────────────────────

class TestGradientFlow:
    """Verify gradients flow through the full pipeline to trainable params."""

    def test_adapter_gets_gradients(self, encoder, adapter, decoder, device):
        adapter.zero_grad()
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        pred.sum().backward()
        for name, p in adapter.named_parameters():
            assert p.grad is not None, f"No gradient for adapter.{name}"

    def test_decoder_gets_gradients(self, encoder, adapter, decoder, device):
        decoder.zero_grad()
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        pred.sum().backward()
        for name, p in decoder.named_parameters():
            assert p.grad is not None, f"No gradient for decoder.{name}"

    def test_encoder_gets_no_gradients(self, encoder, adapter, decoder, device):
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        pred.sum().backward()
        for name, p in encoder.named_parameters():
            assert p.grad is None or (p.grad == 0).all(), \
                f"Encoder param {name} got gradients"

    def test_gradient_through_disc_forward_with_grad(self, encoder, adapter, decoder, disc, device):
        """disc_forward_with_grad must allow grad flow to decoder.last_layer_weight."""
        adapter.zero_grad()
        decoder.zero_grad()
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            tokens = encoder(x)
        pred = decoder(adapter(tokens))
        logits = disc_forward_with_grad(disc, pred)
        L_gan = gan_generator_loss(logits)
        grads = torch.autograd.grad(L_gan, decoder.last_layer_weight, retain_graph=True)
        assert grads[0] is not None
        assert grads[0].abs().sum() > 0


# ── train_step integration ───────────────────────────────────────────

class TestTrainStep:
    """Run real train_step with real components."""

    @pytest.fixture(autouse=True)
    def _setup_optimizers(self, adapter, decoder, disc, config):
        gen_params = list(adapter.parameters()) + list(decoder.parameters())
        self.opt_gen = torch.optim.AdamW(gen_params, lr=config.lr_gen)
        self.opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=config.lr_disc)

    def test_phase1_runs(self, encoder, adapter, decoder, disc, lpips_net, config, device):
        batch = _make_batch(B=2, K=4, device=device)
        losses = train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            self.opt_gen, self.opt_disc, epoch=0, config=config,
        )
        assert "l1" in losses
        assert "lpips" in losses
        assert "total_gen" in losses
        assert "disc" not in losses

    def test_phase2_runs(self, encoder, adapter, decoder, disc, lpips_net, config, device):
        batch = _make_batch(B=2, K=4, device=device)
        losses = train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            self.opt_gen, self.opt_disc, epoch=1, config=config,
        )
        assert "disc" in losses
        assert "gan_gen" not in losses

    def test_phase3_runs(self, encoder, adapter, decoder, disc, lpips_net, config, device):
        batch = _make_batch(B=2, K=4, device=device)
        losses = train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            self.opt_gen, self.opt_disc, epoch=2, config=config,
        )
        assert "gan_gen" in losses
        assert "lambda" in losses
        assert "disc" in losses

    def test_params_update_after_step(self, encoder, adapter, decoder, disc, lpips_net, config, device):
        adapter_before = {k: v.clone() for k, v in adapter.state_dict().items()}
        decoder_before = {k: v.clone() for k, v in decoder.state_dict().items()}

        batch = _make_batch(B=2, K=4, device=device)
        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            self.opt_gen, self.opt_disc, epoch=0, config=config,
        )

        adapter_changed = any(
            not torch.equal(adapter_before[k], v) for k, v in adapter.state_dict().items()
        )
        decoder_changed = any(
            not torch.equal(decoder_before[k], v) for k, v in decoder.state_dict().items()
        )
        assert adapter_changed, "Adapter weights did not update"
        assert decoder_changed, "Decoder weights did not update"

    def test_partial_views(self, encoder, adapter, decoder, disc, lpips_net, config, device):
        batch = _make_batch(B=2, K=4, device=device)
        batch["view_present"][:, 2:] = False  # only 2 of 4 views
        losses = train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            self.opt_gen, self.opt_disc, epoch=0, config=config,
        )
        assert losses["l1"] > 0


# ── validate integration ─────────────────────────────────────────────

class TestValidateIntegration:

    def test_validate_returns_metrics(self, encoder, adapter, decoder, lpips_net, device):
        batch = _make_batch(B=2, K=4, device=device)
        ds = TensorDataset(batch["images_enc"], batch["images_target"], batch["view_present"])

        class _WrappedLoader:
            """Wraps TensorDataset to return dicts like Stage1Dataset."""
            def __init__(self, ds):
                self._loader = DataLoader(ds, batch_size=2)
            def __iter__(self):
                for enc, tgt, vp in self._loader:
                    yield {"images_enc": enc, "images_target": tgt, "view_present": vp}

        val = validate(_WrappedLoader(ds), encoder, adapter, decoder, lpips_net)
        assert "val_l1" in val
        assert "val_lpips" in val
        assert "val_rec" in val
        assert val["val_l1"] > 0
        assert val["val_lpips"] > 0


# ── Checkpoint integration ───────────────────────────────────────────

class TestCheckpointIntegration:

    def test_save_load_roundtrip(self, adapter, decoder, disc, config, device, tmp_path):
        gen_params = list(adapter.parameters()) + list(decoder.parameters())
        opt_gen = torch.optim.AdamW(gen_params, lr=config.lr_gen)
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=config.lr_disc)

        ckpt_path = str(tmp_path / "test_ckpt.pt")
        save_checkpoint(ckpt_path, 5, adapter, decoder, disc, opt_gen, opt_disc, {"val_rec": 0.5})
        assert os.path.isfile(ckpt_path)

        # Create fresh components to load into
        adapter2 = TrainableAdapter().to(device)
        decoder2 = ViTDecoder().to(device)
        disc2 = PatchDiscriminator(pretrained=False).to(device)
        opt_gen2 = torch.optim.AdamW(
            list(adapter2.parameters()) + list(decoder2.parameters()), lr=config.lr_gen
        )
        opt_disc2 = torch.optim.AdamW(disc2.head.parameters(), lr=config.lr_disc)

        resume_epoch = load_checkpoint(ckpt_path, adapter2, decoder2, disc2, opt_gen2, opt_disc2)
        assert resume_epoch == 6

        # Verify weights match
        for k in adapter.state_dict():
            assert torch.equal(adapter.state_dict()[k], adapter2.state_dict()[k])
        for k in decoder.state_dict():
            assert torch.equal(decoder.state_dict()[k], decoder2.state_dict()[k])
