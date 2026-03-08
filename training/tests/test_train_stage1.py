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
