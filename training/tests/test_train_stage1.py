"""Tests for Stage 1 training scaffold (Phase A.6).

Uses mock encoder/adapter/decoder (A.1-A.3 are stubs) to verify
the training loop logic, phase schedule, optimizer wiring, loss
integration, checkpointing, and validation.

Wave 1: Unit tests for individual functions
  - disc_forward_with_grad gradient flow
  - train_step phase logic (phase 1, 2, 3)
  - validate returns expected keys
  - save/load checkpoint round-trip

Wave 2: Integration tests with synthetic HDF5
  - Full train_stage1 for 1-2 epochs
  - Phase transitions
  - Checkpoint save/load resume

Wave 3: Stress and edge cases
  - Single-sample dataset
  - All views masked except one
  - Multiple epochs accumulate correctly
  - Optimizer state preserved across checkpoint resume
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.train_stage1 import (
    Stage1Config,
    disc_forward_with_grad,
    train_step,
    validate,
    save_checkpoint,
    load_checkpoint,
)
from data_pipeline.datasets.stage1_dataset import Stage1Dataset
from models.discriminator import PatchDiscriminator
from models.losses import (
    l1_loss,
    lpips_loss_fn,
    gan_generator_loss,
    gan_discriminator_loss,
    create_lpips_net,
)


# ---------------------------------------------------------------------------
# Mock models (stand-ins for A.1, A.2, A.3 stubs)
# ---------------------------------------------------------------------------

class MockEncoder(nn.Module):
    """Mock frozen encoder: image -> token sequence."""

    def __init__(self, d_model=512, num_patches=196):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        # Simple projection (frozen — no requires_grad)
        self.proj = nn.Linear(3 * 16 * 16, d_model)
        self.pool = nn.AdaptiveAvgPool2d(16)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        # (B, 3, 224, 224) -> (B, num_patches, d_model)
        B = x.shape[0]
        x = self.pool(x)  # (B, 3, 16, 16)
        x = x.reshape(B, 3 * 16 * 16)
        feat = self.proj(x)  # (B, d_model)
        # Repeat to simulate patch tokens
        return feat.unsqueeze(1).expand(B, self.num_patches, self.d_model)


class MockAdapter(nn.Module):
    """Mock adapter: token -> adapted token."""

    def __init__(self, d_in=512, d_out=512):
        super().__init__()
        self.mlp = nn.Linear(d_in, d_out)

    def forward(self, z):
        return self.mlp(z)

    def noise_augment(self, z):
        sigma = abs(torch.randn(1).item()) * 0.8
        return z + torch.randn_like(z) * sigma


class MockDecoder(nn.Module):
    """Mock decoder: tokens -> image [0, 1].

    Exposes last_layer_weight as required by adaptive lambda.
    """

    def __init__(self, d_model=512):
        super().__init__()
        self.fc = nn.Linear(d_model, 3 * 224 * 224)
        # This is what compute_adaptive_lambda differentiates w.r.t.
        self.last_layer_weight = self.fc.weight

    def forward(self, z):
        # z: (B, num_patches, d_model) -> average pool -> fc -> sigmoid
        B = z.shape[0]
        pooled = z.mean(dim=1)  # (B, d_model)
        out = self.fc(pooled)   # (B, 3*224*224)
        return torch.sigmoid(out.reshape(B, 3, 224, 224))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_encoder():
    return MockEncoder()


@pytest.fixture
def mock_adapter():
    return MockAdapter()


@pytest.fixture
def mock_decoder():
    return MockDecoder()


@pytest.fixture
def mock_disc():
    return PatchDiscriminator(pretrained=False)


@pytest.fixture(scope="module")
def lpips_net():
    return create_lpips_net()


@pytest.fixture
def config():
    return Stage1Config(
        batch_size=2,
        num_workers=0,
        num_epochs=3,
        epoch_start_disc=1,
        epoch_start_gan=2,
        save_every=2,
        disc_pretrained=False,
    )


@pytest.fixture
def synthetic_hdf5(tmp_path):
    """Minimal synthetic HDF5 for training tests."""
    path = str(tmp_path / "train_test.hdf5")
    K, H, W = 4, 224, 224
    T = 4
    n_demos = 2

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "test"
        f.attrs["task"] = "test_task"
        f.attrs["proprio_dim"] = 8
        f.attrs["action_dim"] = 8
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = 4

        f.create_group("data")
        f.create_group("mask")

        rng = np.random.RandomState(42)
        for i in range(n_demos):
            grp = f["data"].create_group(f"demo_{i}")
            grp.create_dataset(
                "images",
                data=rng.randint(0, 256, (T, K, H, W, 3), dtype=np.uint8),
            )
            grp.create_dataset(
                "view_present", data=np.array([True, True, True, True])
            )
            grp.create_dataset(
                "actions", data=rng.randn(T, 8).astype(np.float32)
            )
            grp.create_dataset(
                "proprio", data=rng.randn(T, 8).astype(np.float32)
            )

        ns = f.create_group("norm_stats")
        a = ns.create_group("actions")
        a.create_dataset("mean", data=np.zeros(8, dtype=np.float32))
        a.create_dataset("std", data=np.ones(8, dtype=np.float32))

        dt = h5py.special_dtype(vlen=str)
        train_ds = f["mask"].create_dataset("train", shape=(1,), dtype=dt)
        train_ds[:] = [b"demo_0"]
        valid_ds = f["mask"].create_dataset("valid", shape=(1,), dtype=dt)
        valid_ds[:] = [b"demo_1"]

    return path


@pytest.fixture
def synthetic_hdf5_partial(tmp_path):
    """HDF5 with partial views (2 of 4 real)."""
    path = str(tmp_path / "partial.hdf5")
    K, H, W = 4, 224, 224
    T = 4

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "robomimic"
        f.attrs["task"] = "lift"
        f.attrs["proprio_dim"] = 9
        f.attrs["action_dim"] = 7
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = 4

        f.create_group("data")
        f.create_group("mask")

        rng = np.random.RandomState(77)
        grp = f["data"].create_group("demo_0")
        imgs = np.zeros((T, K, H, W, 3), dtype=np.uint8)
        imgs[:, 0] = rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
        imgs[:, 3] = rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
        grp.create_dataset("images", data=imgs)
        grp.create_dataset(
            "view_present", data=np.array([True, False, False, True])
        )
        grp.create_dataset(
            "actions", data=rng.randn(T, 7).astype(np.float32)
        )
        grp.create_dataset(
            "proprio", data=rng.randn(T, 9).astype(np.float32)
        )

        ns = f.create_group("norm_stats")
        a = ns.create_group("actions")
        a.create_dataset("mean", data=np.zeros(7, dtype=np.float32))
        a.create_dataset("std", data=np.ones(7, dtype=np.float32))

        dt = h5py.special_dtype(vlen=str)
        train_ds = f["mask"].create_dataset("train", shape=(1,), dtype=dt)
        train_ds[:] = [b"demo_0"]
        valid_ds = f["mask"].create_dataset("valid", shape=(1,), dtype=dt)
        valid_ds[:] = [b"demo_0"]

    return path


def _make_batch(hdf5_path, batch_size=2):
    """Create a batch from the dataset."""
    ds = Stage1Dataset(hdf5_path, split="train")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


# ===========================================================================
# WAVE 1: Unit tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Stage1Config
# ---------------------------------------------------------------------------

def test_config_defaults():
    """Config should have sensible defaults."""
    cfg = Stage1Config()
    assert cfg.omega_L == 1.0
    assert cfg.omega_G == 0.75
    assert cfg.epoch_start_disc == 6
    assert cfg.epoch_start_gan == 8
    assert cfg.betas == (0.5, 0.9)


def test_config_override():
    """Config fields should be overridable."""
    cfg = Stage1Config(batch_size=64, lr_gen=3e-4, num_epochs=100)
    assert cfg.batch_size == 64
    assert cfg.lr_gen == 3e-4
    assert cfg.num_epochs == 100


# ---------------------------------------------------------------------------
# disc_forward_with_grad
# ---------------------------------------------------------------------------

def test_disc_forward_with_grad_shape(mock_disc):
    """Should return (B, 1) logits."""
    x = torch.randn(2, 3, 224, 224)
    logits = disc_forward_with_grad(mock_disc, x)
    assert logits.shape == (2, 1)


def test_disc_forward_with_grad_allows_gradients(mock_disc):
    """Gradient should flow through to the input (unlike disc.forward)."""
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = disc_forward_with_grad(mock_disc, x)
    logits.sum().backward()
    assert x.grad is not None


def test_disc_forward_normal_blocks_gradients(mock_disc):
    """disc.forward() should NOT allow gradients to flow to input."""
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    logits = mock_disc(x)
    logits.sum().backward()
    assert x.grad is None


def test_disc_forward_with_grad_same_values(mock_disc):
    """Both forward paths should produce identical logit values."""
    x = torch.randn(2, 3, 224, 224)
    logits_normal = mock_disc(x)
    logits_grad = disc_forward_with_grad(mock_disc, x)
    assert torch.allclose(logits_normal, logits_grad, atol=1e-6)


# ---------------------------------------------------------------------------
# train_step: Phase 1 (rec only)
# ---------------------------------------------------------------------------

def test_train_step_phase1_keys(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Phase 1 should return l1, lpips, rec, total_gen (no disc/gan)."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )

    assert "l1" in losses
    assert "lpips" in losses
    assert "rec" in losses
    assert "total_gen" in losses
    assert "disc" not in losses
    assert "gan_gen" not in losses


def test_train_step_phase1_finite_losses(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """All phase 1 losses should be finite and positive."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )

    for k, v in losses.items():
        assert np.isfinite(v), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# train_step: Phase 2 (+ disc)
# ---------------------------------------------------------------------------

def test_train_step_phase2_has_disc_loss(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Phase 2 (epoch >= E_d) should train discriminator."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=1, config=config,
    )

    assert "disc" in losses
    assert "gan_gen" not in losses  # GAN not yet active for generator


# ---------------------------------------------------------------------------
# train_step: Phase 3 (+ GAN)
# ---------------------------------------------------------------------------

def test_train_step_phase3_has_gan(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Phase 3 (epoch >= E_g) should have GAN + disc + lambda."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=2, config=config,
    )

    assert "disc" in losses
    assert "gan_gen" in losses
    assert "lambda" in losses
    assert losses["lambda"] >= 0


def test_train_step_phase3_all_finite(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Phase 3 losses should all be finite."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=2, config=config,
    )

    for k, v in losses.items():
        assert np.isfinite(v), f"{k} not finite: {v}"


# ---------------------------------------------------------------------------
# train_step: gradient updates
# ---------------------------------------------------------------------------

def test_adapter_params_update_phase1(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Adapter parameters should change after a phase 1 step."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-3,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    w_before = mock_adapter.mlp.weight.data.clone()
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    w_after = mock_adapter.mlp.weight.data

    assert not torch.equal(w_before, w_after), "Adapter weights didn't update"


def test_decoder_params_update_phase1(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Decoder parameters should change after a phase 1 step."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-3,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    w_before = mock_decoder.fc.weight.data.clone()
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    w_after = mock_decoder.fc.weight.data

    assert not torch.equal(w_before, w_after), "Decoder weights didn't update"


def test_encoder_params_frozen(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Encoder parameters should NOT change (frozen)."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-3,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    w_before = mock_encoder.proj.weight.data.clone()
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    w_after = mock_encoder.proj.weight.data

    assert torch.equal(w_before, w_after), "Encoder weights changed!"


def test_disc_head_updates_phase2(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Disc head should update in phase 2."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-3,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-3)

    w_before = mock_disc.head[0].weight.data.clone()
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=1, config=config,
    )
    w_after = mock_disc.head[0].weight.data

    assert not torch.equal(w_before, w_after), "Disc head didn't update"


def test_disc_head_frozen_phase1(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Disc head should NOT update in phase 1 (no disc step)."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-3,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-3)

    w_before = mock_disc.head[0].weight.data.clone()
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    w_after = mock_disc.head[0].weight.data

    assert torch.equal(w_before, w_after), "Disc head updated in phase 1!"


def test_disc_backbone_always_frozen(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Disc backbone should never update, even in phase 3."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-3,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-3)

    w_before = mock_disc.backbone.proj.weight.data.clone()
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=2, config=config,
    )
    w_after = mock_disc.backbone.proj.weight.data

    assert torch.equal(w_before, w_after), "Disc backbone updated!"


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_returns_expected_keys(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, lpips_net,
):
    """validate should return val_l1, val_lpips, val_rec."""
    ds = Stage1Dataset(synthetic_hdf5, split="valid")
    loader = DataLoader(ds, batch_size=2, num_workers=0)

    metrics = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)

    assert "val_l1" in metrics
    assert "val_lpips" in metrics
    assert "val_rec" in metrics


def test_validate_finite_values(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, lpips_net,
):
    """All validation metrics should be finite."""
    ds = Stage1Dataset(synthetic_hdf5, split="valid")
    loader = DataLoader(ds, batch_size=2, num_workers=0)

    metrics = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)

    for k, v in metrics.items():
        assert np.isfinite(v), f"{k} not finite: {v}"


def test_validate_rec_equals_sum(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, lpips_net,
):
    """val_rec should equal val_l1 + val_lpips."""
    ds = Stage1Dataset(synthetic_hdf5, split="valid")
    loader = DataLoader(ds, batch_size=2, num_workers=0)

    m = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)

    assert abs(m["val_rec"] - (m["val_l1"] + m["val_lpips"])) < 1e-6


def test_validate_deterministic(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, lpips_net,
):
    """Two validation runs (no training between) should be identical."""
    ds = Stage1Dataset(synthetic_hdf5, split="valid")
    loader = DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

    m1 = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)
    m2 = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)

    for k in m1:
        assert m1[k] == pytest.approx(m2[k], abs=1e-6)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_save_checkpoint_creates_file(
    tmp_path, mock_adapter, mock_decoder, mock_disc,
):
    """save_checkpoint should create the file."""
    path = str(tmp_path / "ckpt.pt")
    opt_gen = torch.optim.AdamW(mock_adapter.parameters(), lr=1e-4)
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    save_checkpoint(
        path, epoch=5, adapter=mock_adapter, decoder=mock_decoder,
        disc=mock_disc, opt_gen=opt_gen, opt_disc=opt_disc,
        val_metrics={"val_rec": 0.5},
    )

    assert os.path.isfile(path)


def test_checkpoint_round_trip(
    tmp_path, mock_disc,
):
    """load_checkpoint should restore exact state from save_checkpoint."""
    path = str(tmp_path / "ckpt.pt")

    # Save
    adapter1 = MockAdapter()
    decoder1 = MockDecoder()
    opt_gen1 = torch.optim.AdamW(
        list(adapter1.parameters()) + list(decoder1.parameters()), lr=1e-4,
    )
    opt_disc1 = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    save_checkpoint(
        path, epoch=3, adapter=adapter1, decoder=decoder1,
        disc=mock_disc, opt_gen=opt_gen1, opt_disc=opt_disc1,
        val_metrics={"val_rec": 0.42},
    )

    # Load into fresh instances
    adapter2 = MockAdapter()
    decoder2 = MockDecoder()
    disc2 = PatchDiscriminator(pretrained=False)
    opt_gen2 = torch.optim.AdamW(
        list(adapter2.parameters()) + list(decoder2.parameters()), lr=1e-4,
    )
    opt_disc2 = torch.optim.AdamW(disc2.head.parameters(), lr=1e-4)

    resume_epoch = load_checkpoint(
        path, adapter2, decoder2, disc2, opt_gen2, opt_disc2,
    )

    assert resume_epoch == 4  # saved epoch + 1

    # Verify adapter weights match
    for p1, p2 in zip(adapter1.parameters(), adapter2.parameters()):
        assert torch.equal(p1, p2)

    # Verify decoder weights match
    for p1, p2 in zip(decoder1.parameters(), decoder2.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_saves_epoch(tmp_path, mock_adapter, mock_decoder, mock_disc):
    """Checkpoint should contain the correct epoch."""
    path = str(tmp_path / "ckpt.pt")
    opt_gen = torch.optim.AdamW(mock_adapter.parameters(), lr=1e-4)
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    save_checkpoint(
        path, epoch=7, adapter=mock_adapter, decoder=mock_decoder,
        disc=mock_disc, opt_gen=opt_gen, opt_disc=opt_disc,
        val_metrics={"val_rec": 0.3},
    )

    ckpt = torch.load(path, weights_only=False)
    assert ckpt["epoch"] == 7
    assert ckpt["val_metrics"]["val_rec"] == 0.3


# ===========================================================================
# WAVE 2: Integration tests
# ===========================================================================

def test_full_training_loop_1_epoch(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """A complete epoch should run without errors."""
    ds = Stage1Dataset(synthetic_hdf5, split="train")
    loader = DataLoader(ds, batch_size=2, num_workers=0, drop_last=True)

    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    total_losses = {}
    n = 0
    for batch in loader:
        losses = train_step(
            batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
            lpips_net, opt_gen, opt_disc, epoch=0, config=config,
        )
        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0) + v
        n += 1

    assert n > 0
    assert all(np.isfinite(v) for v in total_losses.values())


def test_phase_transition_epoch_0_to_1(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Transitioning from phase 1 to phase 2 should add disc loss."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    # Phase 1
    l0 = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    assert "disc" not in l0

    # Phase 2
    l1 = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=1, config=config,
    )
    assert "disc" in l1
    assert "gan_gen" not in l1


def test_phase_transition_epoch_1_to_2(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Transitioning from phase 2 to phase 3 should add GAN + lambda."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    # Phase 2
    l1 = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=1, config=config,
    )
    assert "disc" in l1
    assert "gan_gen" not in l1

    # Phase 3
    l2 = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=2, config=config,
    )
    assert "disc" in l2
    assert "gan_gen" in l2
    assert "lambda" in l2


def test_multiple_steps_same_epoch(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Multiple steps in one epoch should all succeed."""
    ds = Stage1Dataset(synthetic_hdf5, split="train")
    loader = DataLoader(ds, batch_size=2, num_workers=0)

    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    for batch in loader:
        losses = train_step(
            batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
            lpips_net, opt_gen, opt_disc, epoch=2, config=config,
        )
        assert all(np.isfinite(v) for v in losses.values())


def test_train_then_validate(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Training followed by validation should work."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    # Train step
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )

    # Validate
    ds = Stage1Dataset(synthetic_hdf5, split="valid")
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    val = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)
    assert all(np.isfinite(v) for v in val.values())


def test_validate_after_phase3(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Validation after phase 3 training should still work (no GAN in val)."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    # Phase 3 step
    train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=2, config=config,
    )

    # Validate (no GAN, no noise augment)
    ds = Stage1Dataset(synthetic_hdf5, split="valid")
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    val = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)
    assert "val_rec" in val
    assert np.isfinite(val["val_rec"])


# ===========================================================================
# WAVE 3: Partial views, edge cases, stress
# ===========================================================================

def test_partial_views_phase1(
    synthetic_hdf5_partial, mock_encoder, mock_adapter, mock_decoder,
    mock_disc, lpips_net, config,
):
    """Training with partial views (2 of 4) should work."""
    batch = _make_batch(synthetic_hdf5_partial)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    assert all(np.isfinite(v) for v in losses.values())


def test_partial_views_phase3(
    synthetic_hdf5_partial, mock_encoder, mock_adapter, mock_decoder,
    mock_disc, lpips_net, config,
):
    """Phase 3 with partial views should include all loss components."""
    batch = _make_batch(synthetic_hdf5_partial)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=2, config=config,
    )
    assert "disc" in losses
    assert "gan_gen" in losses
    assert "lambda" in losses


def test_validate_partial_views(
    synthetic_hdf5_partial, mock_encoder, mock_adapter, mock_decoder,
    lpips_net,
):
    """Validation with partial views should work."""
    ds = Stage1Dataset(synthetic_hdf5_partial, split="valid")
    loader = DataLoader(ds, batch_size=2, num_workers=0)
    val = validate(loader, mock_encoder, mock_adapter, mock_decoder, lpips_net)
    assert all(np.isfinite(v) for v in val.values())


@pytest.mark.parametrize("epoch", range(3))
def test_all_phases(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config, epoch,
):
    """Each phase should produce finite losses."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=epoch, config=config,
    )
    assert all(np.isfinite(v) for v in losses.values())


def test_consecutive_steps_losses_change(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """Loss values should change across consecutive steps (optimizer works)."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-2,  # large LR to see changes
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    l1_val = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )["l1"]

    l2_val = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )["l1"]

    assert l1_val != l2_val, "Loss didn't change between steps"


def test_checkpoint_resume_continues_training(
    tmp_path, synthetic_hdf5, mock_disc, lpips_net, config,
):
    """Checkpoint save -> load -> train should work seamlessly."""
    path = str(tmp_path / "resume.pt")

    # Train 1 step, save
    adapter1 = MockAdapter()
    decoder1 = MockDecoder()
    opt_gen1 = torch.optim.AdamW(
        list(adapter1.parameters()) + list(decoder1.parameters()), lr=1e-4,
    )
    opt_disc1 = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    batch = _make_batch(synthetic_hdf5)
    enc = MockEncoder()
    train_step(
        batch, enc, adapter1, decoder1, mock_disc,
        lpips_net, opt_gen1, opt_disc1, epoch=0, config=config,
    )
    save_checkpoint(
        path, epoch=0, adapter=adapter1, decoder=decoder1,
        disc=mock_disc, opt_gen=opt_gen1, opt_disc=opt_disc1,
        val_metrics={"val_rec": 0.5},
    )

    # Load into new models and train another step
    adapter2 = MockAdapter()
    decoder2 = MockDecoder()
    disc2 = PatchDiscriminator(pretrained=False)
    opt_gen2 = torch.optim.AdamW(
        list(adapter2.parameters()) + list(decoder2.parameters()), lr=1e-4,
    )
    opt_disc2 = torch.optim.AdamW(disc2.head.parameters(), lr=1e-4)

    resume_epoch = load_checkpoint(
        path, adapter2, decoder2, disc2, opt_gen2, opt_disc2,
    )
    assert resume_epoch == 1

    # Should run without error
    losses = train_step(
        batch, enc, adapter2, decoder2, disc2,
        lpips_net, opt_gen2, opt_disc2, epoch=resume_epoch, config=config,
    )
    assert all(np.isfinite(v) for v in losses.values())


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_various_batch_sizes(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config, batch_size,
):
    """Training should work with different batch sizes."""
    ds = Stage1Dataset(synthetic_hdf5, split="train")
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0)
    batch = next(iter(loader))

    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )
    assert all(np.isfinite(v) for v in losses.values())


def test_custom_omega_weights(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net,
):
    """Custom omega_L and omega_G should affect total loss."""
    batch = _make_batch(synthetic_hdf5)

    cfg_high = Stage1Config(
        omega_L=2.0, omega_G=1.5,
        epoch_start_disc=0, epoch_start_gan=0,
        disc_pretrained=False,
    )
    cfg_low = Stage1Config(
        omega_L=0.1, omega_G=0.1,
        epoch_start_disc=0, epoch_start_gan=0,
        disc_pretrained=False,
    )

    for cfg in [cfg_high, cfg_low]:
        adapter = MockAdapter()
        decoder = MockDecoder()
        disc = PatchDiscriminator(pretrained=False)
        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-4,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-4)

        losses = train_step(
            batch, mock_encoder, adapter, decoder, disc,
            lpips_net, opt_gen, opt_disc, epoch=0, config=cfg,
        )
        assert all(np.isfinite(v) for v in losses.values())


def test_rec_equals_l1_plus_lpips(
    synthetic_hdf5, mock_encoder, mock_adapter, mock_decoder, mock_disc,
    lpips_net, config,
):
    """rec loss should equal l1 + omega_L * lpips."""
    batch = _make_batch(synthetic_hdf5)
    opt_gen = torch.optim.AdamW(
        list(mock_adapter.parameters()) + list(mock_decoder.parameters()),
        lr=1e-4,
    )
    opt_disc = torch.optim.AdamW(mock_disc.head.parameters(), lr=1e-4)

    losses = train_step(
        batch, mock_encoder, mock_adapter, mock_decoder, mock_disc,
        lpips_net, opt_gen, opt_disc, epoch=0, config=config,
    )

    expected_rec = losses["l1"] + config.omega_L * losses["lpips"]
    assert abs(losses["rec"] - expected_rec) < 1e-5


# ===========================================================================
# WAVE 4: Gradient accumulation tests
# ===========================================================================

class MockAdapterDeterministic(nn.Module):
    """Adapter without noise_augment — for deterministic gradient tests."""

    def __init__(self, d_in=512, d_out=512):
        super().__init__()
        self.mlp = nn.Linear(d_in, d_out)

    def forward(self, z):
        return self.mlp(z)


class TestGradientAccumulation:
    """Verify gradient accumulation produces correct results.

    Key invariant: N micro-batches of size B with accum=N should produce
    the same parameter update as 1 batch of size B (since we scale loss
    by 1/N and accumulate N times).

    Uses MockAdapterDeterministic (no noise_augment) so forward passes
    are deterministic given the same input and weights.
    """

    @pytest.fixture
    def accum_setup(self, synthetic_hdf5, lpips_net):
        """Create matched models and a batch for accumulation tests."""
        torch.manual_seed(42)
        encoder = MockEncoder()
        adapter = MockAdapterDeterministic()
        decoder = MockDecoder()
        disc = PatchDiscriminator(pretrained=False)
        batch = _make_batch(synthetic_hdf5, batch_size=2)
        config = Stage1Config(
            batch_size=2, num_workers=0, num_epochs=1,
            epoch_start_disc=1, epoch_start_gan=2,
            disc_pretrained=False,
        )
        return encoder, adapter, decoder, disc, lpips_net, batch, config

    def _clone_params(self, model):
        """Deep-copy all parameter values."""
        return {n: p.clone().detach() for n, p in model.named_parameters()}

    def test_accum1_matches_default(self, accum_setup):
        """With accum_steps=1, loss_scale=1.0 + step_optimizers=True
        should produce identical results to the default behavior."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        # Clone adapter to run both paths (deterministic — no noise_augment)
        torch.manual_seed(42)
        adapter_a = MockAdapterDeterministic()
        adapter_b = MockAdapterDeterministic()
        adapter_b.load_state_dict(adapter_a.state_dict())

        opt_a = torch.optim.AdamW(
            list(adapter_a.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        decoder_b = MockDecoder()
        decoder_b.load_state_dict(decoder.state_dict())
        opt_b = torch.optim.AdamW(
            list(adapter_b.parameters()) + list(decoder_b.parameters()), lr=1e-3,
        )

        # Path A: default (loss_scale=1.0, step_optimizers=True)
        opt_a.zero_grad()
        train_step(
            batch, encoder, adapter_a, decoder, disc, lpips_net,
            opt_a, torch.optim.AdamW(disc.head.parameters(), lr=1e-3),
            epoch=0, config=config,
        )

        # Path B: explicit accum=1 params
        opt_b.zero_grad()
        train_step(
            batch, encoder, adapter_b, decoder_b, disc, lpips_net,
            opt_b, torch.optim.AdamW(disc.head.parameters(), lr=1e-3),
            epoch=0, config=config,
            loss_scale=1.0, step_optimizers=True,
        )

        for (na, pa), (nb, pb) in zip(
            adapter_a.named_parameters(), adapter_b.named_parameters()
        ):
            assert torch.allclose(pa, pb, atol=1e-6), f"Mismatch in {na}"

    def test_accum_grads_accumulate_across_microbatches(self, accum_setup):
        """Calling train_step twice with step_optimizers=False should
        accumulate gradients (not zero them)."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_gen.zero_grad()

        # First micro-batch
        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=0, config=config,
            loss_scale=0.5, step_optimizers=False,
        )
        grads_after_1 = {
            n: p.grad.clone() for n, p in adapter.named_parameters()
            if p.grad is not None
        }

        # Second micro-batch (same data for simplicity)
        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=0, config=config,
            loss_scale=0.5, step_optimizers=False,
        )
        grads_after_2 = {
            n: p.grad.clone() for n, p in adapter.named_parameters()
            if p.grad is not None
        }

        # Gradients should be larger after 2 accumulations
        for name in grads_after_1:
            norm_1 = grads_after_1[name].norm().item()
            norm_2 = grads_after_2[name].norm().item()
            # With same data + loss_scale=0.5, after 2 steps grads ≈ 2x the single step
            # Not exactly 2x because adapter has noise_augment, but strictly larger
            assert norm_2 > norm_1 * 0.9, (
                f"{name}: grad norm didn't grow ({norm_1:.6f} -> {norm_2:.6f})"
            )

    def test_accum_no_step_doesnt_update_params(self, accum_setup):
        """With step_optimizers=False, parameters should NOT change."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_gen.zero_grad()

        params_before = self._clone_params(adapter)

        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=0, config=config,
            loss_scale=0.5, step_optimizers=False,
        )

        for name, p in adapter.named_parameters():
            assert torch.equal(p, params_before[name]), (
                f"{name} changed despite step_optimizers=False"
            )

    def test_accum_step_true_updates_params(self, accum_setup):
        """With step_optimizers=True, parameters SHOULD change."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_gen.zero_grad()

        params_before = self._clone_params(adapter)

        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=0, config=config,
            loss_scale=1.0, step_optimizers=True,
        )

        any_changed = False
        for name, p in adapter.named_parameters():
            if not torch.equal(p, params_before[name]):
                any_changed = True
                break
        assert any_changed, "No parameters changed despite step_optimizers=True"

    def test_accum_step_zeros_grads_after(self, accum_setup):
        """After step_optimizers=True, gradients should be zeroed."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_gen.zero_grad()

        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=0, config=config,
            loss_scale=1.0, step_optimizers=True,
        )

        for name, p in adapter.named_parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0), (
                    f"{name} has nonzero grad after step_optimizers=True"
                )

    def test_loss_scale_halves_gradients(self, accum_setup):
        """loss_scale=0.5 should produce half the gradient norm vs scale=1.0."""
        encoder, _, _, disc, lpips_net, batch, config = accum_setup

        results = {}
        for scale in [1.0, 0.5]:
            torch.manual_seed(42)
            adapter = MockAdapterDeterministic()
            decoder = MockDecoder()
            opt_gen = torch.optim.AdamW(
                list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
            )
            opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
            opt_gen.zero_grad()

            train_step(
                batch, encoder, adapter, decoder, disc, lpips_net,
                opt_gen, opt_disc, epoch=0, config=config,
                loss_scale=scale, step_optimizers=False,
            )
            results[scale] = {
                n: p.grad.norm().item()
                for n, p in adapter.named_parameters() if p.grad is not None
            }

        for name in results[1.0]:
            ratio = results[0.5][name] / (results[1.0][name] + 1e-12)
            assert abs(ratio - 0.5) < 0.05, (
                f"{name}: expected ratio ~0.5, got {ratio:.4f}"
            )

    def test_accum_returns_unscaled_losses(self, accum_setup):
        """Loss values in the returned dict should be unscaled (original magnitude).

        Both runs use identical fresh weights (no stepping) so the only
        difference is loss_scale. Logged losses should match exactly.
        """
        encoder, _, _, disc, lpips_net, batch, config = accum_setup

        # Run A: scale=1.0, no step
        torch.manual_seed(42)
        adapter_a = MockAdapterDeterministic()
        decoder_a = MockDecoder()
        opt_a = torch.optim.AdamW(
            list(adapter_a.parameters()) + list(decoder_a.parameters()), lr=1e-3,
        )
        opt_disc_a = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_a.zero_grad()
        losses_full = train_step(
            batch, encoder, adapter_a, decoder_a, disc, lpips_net,
            opt_a, opt_disc_a, epoch=0, config=config,
            loss_scale=1.0, step_optimizers=False,
        )

        # Run B: same weights, scale=0.25, no step
        torch.manual_seed(42)
        adapter_b = MockAdapterDeterministic()
        decoder_b = MockDecoder()
        opt_b = torch.optim.AdamW(
            list(adapter_b.parameters()) + list(decoder_b.parameters()), lr=1e-3,
        )
        opt_disc_b = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_b.zero_grad()
        losses_scaled = train_step(
            batch, encoder, adapter_b, decoder_b, disc, lpips_net,
            opt_b, opt_disc_b, epoch=0, config=config,
            loss_scale=0.25, step_optimizers=False,
        )

        # Logged losses should be the same (unscaled)
        for key in ["l1", "lpips", "rec", "total_gen"]:
            assert abs(losses_full[key] - losses_scaled[key]) < 1e-5, (
                f"{key}: {losses_full[key]:.6f} vs {losses_scaled[key]:.6f}"
            )

    def test_accum_disc_phase2(self, accum_setup):
        """Gradient accumulation should work during disc phase (epoch >= epoch_start_disc)."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_gen.zero_grad()
        opt_disc.zero_grad()

        disc_before = self._clone_params(disc.head)

        # epoch=1 = phase2 (disc on, GAN off)
        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=1, config=config,
            loss_scale=0.5, step_optimizers=False,
        )
        # Disc params shouldn't change yet
        for name, p in disc.head.named_parameters():
            assert torch.equal(p, disc_before[name]), (
                f"disc.head.{name} changed during accumulation"
            )
        # But disc grads should exist
        has_disc_grad = any(
            p.grad is not None and p.grad.norm() > 0
            for p in disc.head.parameters()
        )
        assert has_disc_grad, "No disc gradients accumulated"

        # Now step
        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=1, config=config,
            loss_scale=0.5, step_optimizers=True,
        )
        # Disc params should have changed now
        any_changed = any(
            not torch.equal(p, disc_before[n])
            for n, p in disc.head.named_parameters()
        )
        assert any_changed, "Disc params didn't update after step"

    def test_accum_gan_phase3(self, accum_setup):
        """Gradient accumulation should work during GAN phase (epoch >= epoch_start_gan)."""
        encoder, adapter, decoder, disc, lpips_net, batch, config = accum_setup

        opt_gen = torch.optim.AdamW(
            list(adapter.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        opt_disc = torch.optim.AdamW(disc.head.parameters(), lr=1e-3)
        opt_gen.zero_grad()
        opt_disc.zero_grad()

        # epoch=2 = phase3 (GAN on)
        losses = train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=2, config=config,
            loss_scale=0.5, step_optimizers=False,
        )
        assert "gan_gen" in losses, "GAN loss missing in phase 3"
        assert "lambda" in losses, "Adaptive lambda missing in phase 3"
        assert "disc" in losses, "Disc loss missing in phase 3"

        # Step should work without error
        train_step(
            batch, encoder, adapter, decoder, disc, lpips_net,
            opt_gen, opt_disc, epoch=2, config=config,
            loss_scale=0.5, step_optimizers=True,
        )

    def test_config_grad_accum_steps_default(self):
        """Default grad_accum_steps should be 1 (no accumulation)."""
        cfg = Stage1Config()
        assert cfg.grad_accum_steps == 1

    def test_config_grad_accum_steps_override(self):
        """grad_accum_steps should be overridable."""
        cfg = Stage1Config(grad_accum_steps=4)
        assert cfg.grad_accum_steps == 4

    def test_accum2_numerical_equivalence(self, synthetic_hdf5, lpips_net):
        """2 micro-batches of size B with accum=2 should produce the same
        parameter update as 1 batch of size B with accum=1.

        Since we use the SAME data for both micro-batches:
          accum path:  grad = (0.5 * grad_B) + (0.5 * grad_B) = grad_B
          single path: grad = 1.0 * grad_B
        So final weights should match exactly (same optimizer state).
        """
        config = Stage1Config(
            batch_size=2, num_workers=0, epoch_start_disc=1,
            epoch_start_gan=2, disc_pretrained=False,
        )
        batch = _make_batch(synthetic_hdf5, batch_size=2)

        encoder = MockEncoder()

        # ---- Single step path ----
        torch.manual_seed(99)
        adapter_single = MockAdapterDeterministic()
        decoder_single = MockDecoder()
        disc_single = PatchDiscriminator(pretrained=False)
        opt_gen_s = torch.optim.AdamW(
            list(adapter_single.parameters()) + list(decoder_single.parameters()),
            lr=1e-3, betas=(0.9, 0.999),
        )
        opt_disc_s = torch.optim.AdamW(disc_single.head.parameters(), lr=1e-3)
        opt_gen_s.zero_grad()
        opt_disc_s.zero_grad()

        train_step(
            batch, encoder, adapter_single, decoder_single, disc_single,
            lpips_net, opt_gen_s, opt_disc_s, epoch=0, config=config,
            loss_scale=1.0, step_optimizers=True,
        )

        # ---- Accum=2 path (same data twice) ----
        torch.manual_seed(99)
        adapter_accum = MockAdapterDeterministic()
        decoder_accum = MockDecoder()
        disc_accum = PatchDiscriminator(pretrained=False)
        opt_gen_a = torch.optim.AdamW(
            list(adapter_accum.parameters()) + list(decoder_accum.parameters()),
            lr=1e-3, betas=(0.9, 0.999),
        )
        opt_disc_a = torch.optim.AdamW(disc_accum.head.parameters(), lr=1e-3)
        opt_gen_a.zero_grad()
        opt_disc_a.zero_grad()

        # Micro-batch 1: accumulate
        train_step(
            batch, encoder, adapter_accum, decoder_accum, disc_accum,
            lpips_net, opt_gen_a, opt_disc_a, epoch=0, config=config,
            loss_scale=0.5, step_optimizers=False,
        )
        # Micro-batch 2: accumulate + step
        train_step(
            batch, encoder, adapter_accum, decoder_accum, disc_accum,
            lpips_net, opt_gen_a, opt_disc_a, epoch=0, config=config,
            loss_scale=0.5, step_optimizers=True,
        )

        # Weights should match
        for (ns, ps), (na, pa) in zip(
            adapter_single.named_parameters(),
            adapter_accum.named_parameters(),
        ):
            assert torch.allclose(ps, pa, atol=1e-5), (
                f"adapter.{ns}: single vs accum mismatch "
                f"(max diff={torch.max(torch.abs(ps - pa)).item():.2e})"
            )
