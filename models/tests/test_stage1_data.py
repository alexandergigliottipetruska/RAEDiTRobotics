"""Tests for Stage 1 data integration (Phase A.7).

Tests verify that the Stage1Dataset correctly bridges the unified HDF5
data pipeline with the Stage 1 model components (encoder, losses, discriminator).

Wave 1: Unit tests with synthetic HDF5 (no real data needed)
  - Output shapes and dtypes
  - ImageNet normalization correctness
  - Raw images in [0,1] range
  - view_present mask correctness
  - Denormalization round-trip
  - DataLoader compatibility
  - Index coverage (all timesteps)

Wave 2: Integration with loss functions
  - L1 loss accepts dataset output directly
  - LPIPS loss accepts dataset output
  - GAN discriminator accepts dataset output (interpolated to 224)
  - Reconstruction loss end-to-end
  - view_present masking reduces loss to real views only

Wave 3: Integration with real HDF5 (parametrized, requires --hdf5)
  - Real data shapes match expected
  - Real images have valid pixel ranges
  - Multiple benchmarks (robomimic vs RLBench camera configs)

References:
  - Zheng et al. 2025 — RAE training uses ImageNet-normalized encoder input
    and [0,1] reconstruction targets
"""

import os
import tempfile

import h5py
import numpy as np
import torch
import torch.nn as nn
import pytest

from data_pipeline.datasets.stage1_dataset import Stage1Dataset


# ImageNet constants (must match stage1_dataset.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures: synthetic HDF5 files
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_hdf5_uint8(tmp_path):
    """Create a minimal synthetic unified HDF5 with uint8 images (like RLBench)."""
    path = str(tmp_path / "synthetic_uint8.hdf5")
    K, H, W = 4, 224, 224
    T = 10  # timesteps per demo
    n_demos = 3

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "rlbench"
        f.attrs["task"] = "test_task"
        f.attrs["proprio_dim"] = 8
        f.attrs["action_dim"] = 8
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = 4

        f.create_group("data")
        f.create_group("mask")

        rng = np.random.RandomState(42)
        demo_keys = []
        for i in range(n_demos):
            key = f"demo_{i}"
            demo_keys.append(key)
            grp = f["data"].create_group(key)

            # uint8 images with known values
            imgs = rng.randint(0, 256, (T, K, H, W, 3), dtype=np.uint8)
            grp.create_dataset("images", data=imgs)
            grp.create_dataset("view_present", data=np.array([True, True, True, True]))
            grp.create_dataset("actions", data=rng.randn(T, 8).astype(np.float32))
            grp.create_dataset("proprio", data=rng.randn(T, 8).astype(np.float32))

        # Norm stats (required by base_dataset but not by Stage1Dataset)
        ns = f.create_group("norm_stats")
        a = ns.create_group("actions")
        a.create_dataset("mean", data=np.zeros(8, dtype=np.float32))
        a.create_dataset("std", data=np.ones(8, dtype=np.float32))

        # Write masks
        dt = h5py.special_dtype(vlen=str)
        train_ds = f["mask"].create_dataset("train", shape=(2,), dtype=dt)
        train_ds[:] = [b"demo_0", b"demo_1"]
        valid_ds = f["mask"].create_dataset("valid", shape=(1,), dtype=dt)
        valid_ds[:] = [b"demo_2"]

    return path


@pytest.fixture
def synthetic_hdf5_partial_views(tmp_path):
    """Synthetic HDF5 with only 2 real camera views (like robomimic)."""
    path = str(tmp_path / "synthetic_partial.hdf5")
    K, H, W = 4, 224, 224
    T = 5

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "robomimic"
        f.attrs["task"] = "lift"
        f.attrs["proprio_dim"] = 9
        f.attrs["action_dim"] = 7
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = 4

        f.create_group("data")
        f.create_group("mask")

        rng = np.random.RandomState(123)
        grp = f["data"].create_group("demo_0")

        # Only slots 0 and 3 have real images; slots 1, 2 are zero-padded
        imgs = np.zeros((T, K, H, W, 3), dtype=np.uint8)
        imgs[:, 0] = rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
        imgs[:, 3] = rng.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
        grp.create_dataset("images", data=imgs)
        grp.create_dataset("view_present", data=np.array([True, False, False, True]))
        grp.create_dataset("actions", data=rng.randn(T, 7).astype(np.float32))
        grp.create_dataset("proprio", data=rng.randn(T, 9).astype(np.float32))

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


# ===========================================================================
# WAVE 1: Unit tests with synthetic HDF5
# ===========================================================================

# ---------------------------------------------------------------------------
# Output shapes and dtypes
# ---------------------------------------------------------------------------

def test_output_keys(synthetic_hdf5_uint8):
    """Dataset should return exactly images_enc, images_target, view_present."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert set(sample.keys()) == {"images_enc", "images_target", "view_present"}


def test_output_shapes(synthetic_hdf5_uint8):
    """All outputs should have correct shapes."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    K, H, W = 4, 224, 224
    assert sample["images_enc"].shape == (K, 3, H, W)
    assert sample["images_target"].shape == (K, 3, H, W)
    assert sample["view_present"].shape == (K,)


def test_output_dtypes(synthetic_hdf5_uint8):
    """images should be float32, view_present should be bool."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert sample["images_enc"].dtype == torch.float32
    assert sample["images_target"].dtype == torch.float32
    assert sample["view_present"].dtype == torch.bool


def test_no_actions_or_proprio(synthetic_hdf5_uint8):
    """Stage 1 dataset should NOT return actions or proprio."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert "actions" not in sample
    assert "proprio" not in sample


# ---------------------------------------------------------------------------
# Index coverage
# ---------------------------------------------------------------------------

def test_dataset_length_train(synthetic_hdf5_uint8):
    """Train split should cover all timesteps of train demos."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    # 2 demos × 10 timesteps = 20
    assert len(ds) == 20


def test_dataset_length_valid(synthetic_hdf5_uint8):
    """Valid split should cover all timesteps of valid demos."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="valid")
    # 1 demo × 10 timesteps = 10
    assert len(ds) == 10


def test_all_indices_accessible(synthetic_hdf5_uint8):
    """Every index in range should return a valid sample."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["images_enc"].shape == (4, 3, 224, 224)


# ---------------------------------------------------------------------------
# Image value ranges
# ---------------------------------------------------------------------------

def test_target_images_in_01_range(synthetic_hdf5_uint8):
    """Reconstruction targets should be in [0, 1]."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert sample["images_target"].min() >= 0.0
    assert sample["images_target"].max() <= 1.0


def test_target_images_not_all_zero(synthetic_hdf5_uint8):
    """Target images should have actual content (not all zeros)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert sample["images_target"].abs().sum() > 0


def test_encoder_images_imagenet_range(synthetic_hdf5_uint8):
    """ImageNet-normalized images should be roughly in [-2.2, 2.7]."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    # ImageNet norm maps [0,1] to approximately [-2.12, 2.64]
    assert sample["images_enc"].min() >= -3.0
    assert sample["images_enc"].max() <= 3.0


def test_encoder_and_target_differ(synthetic_hdf5_uint8):
    """Encoder images (normalized) should differ from target images (raw)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert not torch.allclose(sample["images_enc"], sample["images_target"])


# ---------------------------------------------------------------------------
# ImageNet normalization correctness
# ---------------------------------------------------------------------------

def test_imagenet_normalization_manual_check(synthetic_hdf5_uint8):
    """Verify ImageNet normalization matches manual computation."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    # Recover raw from target (already [0,1])
    target = sample["images_target"]  # (K, 3, H, W)

    # Manually normalize
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    expected_enc = (target - mean) / std

    assert torch.allclose(sample["images_enc"], expected_enc, atol=1e-5)


def test_denormalization_round_trip(synthetic_hdf5_uint8):
    """Denormalizing encoder images should recover target images."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    recovered = sample["images_enc"] * std + mean

    assert torch.allclose(recovered, sample["images_target"], atol=1e-5)


# ---------------------------------------------------------------------------
# view_present mask
# ---------------------------------------------------------------------------

def test_view_present_all_true(synthetic_hdf5_uint8):
    """RLBench-like HDF5 should have all views present."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    assert sample["view_present"].all()


def test_view_present_partial(synthetic_hdf5_partial_views):
    """Robomimic-like HDF5 should have only slots 0, 3 present."""
    ds = Stage1Dataset(synthetic_hdf5_partial_views, split="train")
    sample = ds[0]
    expected = torch.tensor([True, False, False, True])
    assert torch.equal(sample["view_present"], expected)


def test_view_present_consistent_across_samples(synthetic_hdf5_uint8):
    """view_present should be the same for all samples (constant per benchmark)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    vp_0 = ds[0]["view_present"]
    vp_5 = ds[5]["view_present"]
    vp_last = ds[len(ds) - 1]["view_present"]
    assert torch.equal(vp_0, vp_5)
    assert torch.equal(vp_0, vp_last)


def test_padded_views_are_zero(synthetic_hdf5_partial_views):
    """Camera slots marked as not-present should have zero images."""
    ds = Stage1Dataset(synthetic_hdf5_partial_views, split="train")
    sample = ds[0]
    vp = sample["view_present"]
    for k in range(4):
        if not vp[k]:
            assert sample["images_target"][k].abs().sum() == 0


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------

def test_dataloader_batching(synthetic_hdf5_uint8):
    """Dataset should work with DataLoader and batch correctly."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    assert batch["images_enc"].shape == (4, 4, 3, 224, 224)
    assert batch["images_target"].shape == (4, 4, 3, 224, 224)
    assert batch["view_present"].shape == (4, 4)


def test_dataloader_shuffle(synthetic_hdf5_uint8):
    """Dataset should support shuffled DataLoader."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))
    assert batch["images_enc"].shape[0] == 2


def test_dataloader_full_epoch(synthetic_hdf5_uint8):
    """Should iterate through entire dataset without error."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    total = 0
    for batch in loader:
        total += batch["images_enc"].shape[0]
    assert total == len(ds)


# ===========================================================================
# WAVE 2: Integration with model components
# ===========================================================================

from models.losses import (
    l1_loss,
    lpips_loss_fn,
    gan_discriminator_loss,
    reconstruction_loss,
    create_lpips_net,
)
from models.discriminator import PatchDiscriminator


@pytest.fixture(scope="module")
def lpips_net():
    return create_lpips_net()


# ---------------------------------------------------------------------------
# Loss function integration
# ---------------------------------------------------------------------------

def test_l1_loss_on_dataset_output(synthetic_hdf5_uint8):
    """L1 loss should accept images_target from dataset directly."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    # Simulate: decoder outputs a reconstruction for view 0
    pred = torch.randn_like(sample["images_target"][0:1])  # (1, 3, 224, 224)
    target = sample["images_target"][0:1]

    loss = l1_loss(pred, target)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_lpips_loss_on_dataset_output(synthetic_hdf5_uint8, lpips_net):
    """LPIPS loss should accept images_target from dataset.
    LPIPS internally converts [0,1] → [-1,1]."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    # Use smaller spatial size for speed (LPIPS handles it)
    target = sample["images_target"][0:1]  # (1, 3, 224, 224)
    pred = torch.randn_like(target).clamp(0, 1)

    loss = lpips_loss_fn(pred, target, lpips_net)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_discriminator_on_dataset_images(synthetic_hdf5_uint8):
    """Discriminator should accept images from dataset (already 224x224)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    disc = PatchDiscriminator(pretrained=False)
    # Feed one view to discriminator
    img = sample["images_target"][0:1]  # (1, 3, 224, 224)
    logits = disc(img)
    assert logits.shape == (1, 1)
    assert torch.isfinite(logits).all()


def test_recon_loss_on_dataset_output(synthetic_hdf5_uint8, lpips_net):
    """Full reconstruction_loss should work with dataset images."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    target = sample["images_target"][0:1]  # (1, 3, H, W)
    pred = torch.randn_like(target).clamp(0, 1)

    loss = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
    )
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# View-present masking
# ---------------------------------------------------------------------------

def test_masked_loss_excludes_padded_views(synthetic_hdf5_partial_views, lpips_net):
    """Loss should only be computed on views where view_present=True.
    This tests the masking pattern that the training loop will use."""
    ds = Stage1Dataset(synthetic_hdf5_partial_views, split="train")
    sample = ds[0]

    vp = sample["view_present"]  # [True, False, False, True]
    targets = sample["images_target"]  # (4, 3, H, W)

    # Simulate per-view loss computation
    losses = []
    for k in range(4):
        if vp[k]:
            pred = torch.randn_like(targets[k:k+1]).clamp(0, 1)
            loss = l1_loss(pred, targets[k:k+1])
            losses.append(loss)

    # Should only have 2 losses (views 0 and 3)
    assert len(losses) == 2
    avg_loss = torch.stack(losses).mean()
    assert torch.isfinite(avg_loss)


def test_masked_loss_all_views(synthetic_hdf5_uint8, lpips_net):
    """When all views are present, all should contribute to loss."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    vp = sample["view_present"]
    assert vp.all()

    losses = []
    for k in range(4):
        if vp[k]:
            pred = torch.randn_like(sample["images_target"][k:k+1]).clamp(0, 1)
            losses.append(l1_loss(pred, sample["images_target"][k:k+1]))

    assert len(losses) == 4


def test_batch_masked_loss_pattern(synthetic_hdf5_partial_views):
    """Test the batch-level masking pattern for training loop."""
    ds = Stage1Dataset(synthetic_hdf5_partial_views, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # (B, K, 3, H, W)
    targets = batch["images_target"]
    vp = batch["view_present"]  # (B, K)

    # Flatten real views into (N_real, 3, H, W)
    B, K = vp.shape
    real_mask = vp.view(-1)  # (B*K,)
    all_views = targets.view(B * K, 3, 224, 224)
    real_views = all_views[real_mask]

    # Should have B * 2 real views (slots 0, 3 per sample)
    assert real_views.shape[0] == 2 * 2  # 2 samples × 2 real views


# ---------------------------------------------------------------------------
# End-to-end with discriminator + adaptive lambda
# ---------------------------------------------------------------------------

def test_full_stage1_forward_pattern(synthetic_hdf5_uint8, lpips_net):
    """Simulate the full Stage 1 forward pass pattern:
    1. Load batch from dataset
    2. Encoder processes images_enc (simulated)
    3. Decoder reconstructs to [0,1] (simulated)
    4. Compute reconstruction loss against images_target
    5. Discriminator scores real (target) and fake (pred)
    6. Compute discriminator loss
    """
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    disc = PatchDiscriminator(pretrained=False)

    # For one view
    target = sample["images_target"][0:1]  # (1, 3, 224, 224)

    # Simulate decoder output
    last_w = nn.Parameter(torch.ones(1))
    raw_pred = torch.rand(1, 3, 224, 224)
    pred = raw_pred * last_w  # pred depends on last_w

    # --- Generator losses ---
    # Recon loss (no GAN yet — phase 1 of training)
    loss_recon = reconstruction_loss(
        pred, target, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
    )
    assert torch.isfinite(loss_recon)

    # --- Discriminator loss ---
    logits_real = disc(target)
    logits_fake = disc(pred.detach())
    d_loss = gan_discriminator_loss(logits_real, logits_fake)
    assert d_loss.item() >= 0


# ===========================================================================
# WAVE 3: Statistical and stress tests
# ===========================================================================

class TestStatistical:
    """Statistical tests across multiple samples."""

    @pytest.mark.parametrize("idx", range(20))
    def test_all_samples_have_valid_ranges(self, synthetic_hdf5_uint8, idx):
        """Every sample should have target in [0,1] and finite enc values."""
        ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
        sample = ds[idx]
        assert sample["images_target"].min() >= 0.0
        assert sample["images_target"].max() <= 1.0
        assert torch.isfinite(sample["images_enc"]).all()

    @pytest.mark.parametrize("idx", range(20))
    def test_normalization_consistent(self, synthetic_hdf5_uint8, idx):
        """ImageNet normalization should be consistent across all samples."""
        ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
        sample = ds[idx]

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        expected = (sample["images_target"] - mean) / std

        assert torch.allclose(sample["images_enc"], expected, atol=1e-5)


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_first_and_last_timestep(self, synthetic_hdf5_uint8):
        """First and last timesteps should work correctly."""
        ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
        first = ds[0]
        last = ds[len(ds) - 1]
        assert first["images_target"].shape == (4, 3, 224, 224)
        assert last["images_target"].shape == (4, 3, 224, 224)

    def test_valid_split(self, synthetic_hdf5_uint8):
        """Valid split should work independently of train split."""
        ds_train = Stage1Dataset(synthetic_hdf5_uint8, split="train")
        ds_valid = Stage1Dataset(synthetic_hdf5_uint8, split="valid")
        assert len(ds_train) != len(ds_valid)
        sample = ds_valid[0]
        assert sample["images_target"].shape == (4, 3, 224, 224)

    def test_different_samples_have_different_images(self, synthetic_hdf5_uint8):
        """Different timesteps should generally have different images."""
        ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
        s0 = ds[0]["images_target"]
        s5 = ds[5]["images_target"]
        # With random synthetic data, these should differ
        assert not torch.allclose(s0, s5)

    def test_partial_views_real_slots_have_content(self, synthetic_hdf5_partial_views):
        """Real camera slots should have non-zero images."""
        ds = Stage1Dataset(synthetic_hdf5_partial_views, split="train")
        sample = ds[0]
        vp = sample["view_present"]
        for k in range(4):
            if vp[k]:
                assert sample["images_target"][k].abs().sum() > 0

    def test_encoder_images_different_per_channel(self, synthetic_hdf5_uint8):
        """After ImageNet norm, different channels should have different stats
        (because ImageNet mean/std differ per channel)."""
        ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
        sample = ds[0]
        enc = sample["images_enc"][0]  # (3, H, W) for view 0
        # Channel means should differ (ImageNet subtracts different values)
        ch_means = [enc[c].mean().item() for c in range(3)]
        assert not (ch_means[0] == pytest.approx(ch_means[1], abs=0.01)
                    and ch_means[1] == pytest.approx(ch_means[2], abs=0.01))


# ---------------------------------------------------------------------------
# Consistency with base_dataset
# ---------------------------------------------------------------------------

def test_imagenet_norm_matches_base_dataset(synthetic_hdf5_uint8):
    """Our ImageNet normalization should produce same results as base_dataset.
    This ensures Stage 1 encoder gets identical inputs."""
    from data_pipeline.datasets.base_dataset import _imagenet_normalize

    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")

    # Read raw images directly from HDF5
    with h5py.File(synthetic_hdf5_uint8, "r") as f:
        raw = f["data/demo_0/images"][0]  # (K, H, W, 3) uint8

    # base_dataset's normalization
    base_result = _imagenet_normalize(raw)  # (K, 3, H, W)

    # Our dataset's normalization
    sample = ds[0]
    our_result = sample["images_enc"].numpy()

    np.testing.assert_allclose(our_result, base_result, atol=1e-5)


# ===========================================================================
# WAVE 4: Heavy stress / robustness / precision tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixture: variable-length demos
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_hdf5_variable_length(tmp_path):
    """HDF5 with demos of different timestep lengths."""
    path = str(tmp_path / "synthetic_varlen.hdf5")
    K, H, W = 4, 224, 224
    lengths = [3, 7, 1, 12, 5]  # 5 demos, different lengths

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "rlbench"
        f.attrs["task"] = "test_task"
        f.attrs["proprio_dim"] = 8
        f.attrs["action_dim"] = 8
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = 4

        f.create_group("data")
        f.create_group("mask")

        rng = np.random.RandomState(99)
        for i, T in enumerate(lengths):
            grp = f["data"].create_group(f"demo_{i}")
            grp.create_dataset("images", data=rng.randint(0, 256, (T, K, H, W, 3), dtype=np.uint8))
            grp.create_dataset("view_present", data=np.array([True, True, True, True]))
            grp.create_dataset("actions", data=rng.randn(T, 8).astype(np.float32))
            grp.create_dataset("proprio", data=rng.randn(T, 8).astype(np.float32))

        ns = f.create_group("norm_stats")
        a = ns.create_group("actions")
        a.create_dataset("mean", data=np.zeros(8, dtype=np.float32))
        a.create_dataset("std", data=np.ones(8, dtype=np.float32))

        dt = h5py.special_dtype(vlen=str)
        train_ds = f["mask"].create_dataset("train", shape=(4,), dtype=dt)
        train_ds[:] = [f"demo_{i}".encode() for i in range(4)]
        valid_ds = f["mask"].create_dataset("valid", shape=(1,), dtype=dt)
        valid_ds[:] = [b"demo_4"]

    return path, lengths


@pytest.fixture
def synthetic_hdf5_known_pixels(tmp_path):
    """HDF5 with exact known pixel values for precision testing."""
    path = str(tmp_path / "synthetic_known.hdf5")
    K, H, W = 4, 224, 224

    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "rlbench"
        f.attrs["task"] = "test_task"
        f.attrs["proprio_dim"] = 8
        f.attrs["action_dim"] = 8
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = 4

        f.create_group("data")
        f.create_group("mask")

        grp = f["data"].create_group("demo_0")

        # T=2, known pixel values
        imgs = np.zeros((2, K, H, W, 3), dtype=np.uint8)
        # Timestep 0: all pixels = [128, 64, 255]
        imgs[0, :, :, :, :] = np.array([128, 64, 255], dtype=np.uint8)
        # Timestep 1: gradient across width
        for w in range(W):
            imgs[1, :, :, w, :] = np.array([w % 256, 0, 0], dtype=np.uint8)

        grp.create_dataset("images", data=imgs)
        grp.create_dataset("view_present", data=np.array([True, True, True, True]))
        grp.create_dataset("actions", data=np.zeros((2, 8), dtype=np.float32))
        grp.create_dataset("proprio", data=np.zeros((2, 8), dtype=np.float32))

        ns = f.create_group("norm_stats")
        a = ns.create_group("actions")
        a.create_dataset("mean", data=np.zeros(8, dtype=np.float32))
        a.create_dataset("std", data=np.ones(8, dtype=np.float32))

        dt = h5py.special_dtype(vlen=str)
        train_ds = f["mask"].create_dataset("train", shape=(1,), dtype=dt)
        train_ds[:] = [b"demo_0"]
        valid_ds = f["mask"].create_dataset("valid", shape=(1,), dtype=dt)
        valid_ds[:] = [b"demo_0"]

    return path


# ---------------------------------------------------------------------------
# Variable-length demo tests
# ---------------------------------------------------------------------------

def test_variable_length_total_samples(synthetic_hdf5_variable_length):
    """Total samples = sum of all demo timestep counts."""
    path, lengths = synthetic_hdf5_variable_length
    ds = Stage1Dataset(path, split="train")
    assert len(ds) == sum(lengths[:4])  # first 4 demos are train


def test_variable_length_valid_split(synthetic_hdf5_variable_length):
    """Valid split with 1 demo of length 5."""
    path, lengths = synthetic_hdf5_variable_length
    ds = Stage1Dataset(path, split="valid")
    assert len(ds) == lengths[4]  # 5


def test_variable_length_all_accessible(synthetic_hdf5_variable_length):
    """Every index across variable-length demos should work."""
    path, _ = synthetic_hdf5_variable_length
    ds = Stage1Dataset(path, split="train")
    for i in range(len(ds)):
        sample = ds[i]
        assert sample["images_target"].shape == (4, 3, 224, 224)


def test_variable_length_cross_demo_boundary(synthetic_hdf5_variable_length):
    """Samples spanning demo boundaries should come from different demos."""
    path, lengths = synthetic_hdf5_variable_length
    ds = Stage1Dataset(path, split="train")
    # Sample at index lengths[0]-1 = last of demo_0
    # Sample at index lengths[0] = first of demo_1
    s_last = ds[lengths[0] - 1]
    s_first = ds[lengths[0]]
    # They should differ (different demos, different random data)
    assert not torch.allclose(s_last["images_target"], s_first["images_target"])


def test_single_timestep_demo(synthetic_hdf5_variable_length):
    """Demo with T=1 should produce exactly 1 sample."""
    path, lengths = synthetic_hdf5_variable_length
    # demo_2 has length 1; its samples start at sum(lengths[:2])
    offset = sum(lengths[:2])  # 3 + 7 = 10
    ds = Stage1Dataset(path, split="train")
    sample = ds[offset]
    assert sample["images_target"].shape == (4, 3, 224, 224)


# ---------------------------------------------------------------------------
# Known-pixel precision tests
# ---------------------------------------------------------------------------

def test_known_pixel_uint8_to_float32_precision(synthetic_hdf5_known_pixels):
    """Verify exact float32 values from known uint8 pixels."""
    ds = Stage1Dataset(synthetic_hdf5_known_pixels, split="train")
    sample = ds[0]

    # Known: [128, 64, 255] → [128/255, 64/255, 255/255]
    target = sample["images_target"]  # (K, 3, H, W)
    expected_r = 128.0 / 255.0
    expected_g = 64.0 / 255.0
    expected_b = 255.0 / 255.0

    # Check all views, all pixels (they're all the same for t=0)
    assert torch.allclose(target[:, 0, :, :], torch.full_like(target[:, 0, :, :], expected_r), atol=1e-6)
    assert torch.allclose(target[:, 1, :, :], torch.full_like(target[:, 1, :, :], expected_g), atol=1e-6)
    assert torch.allclose(target[:, 2, :, :], torch.full_like(target[:, 2, :, :], expected_b), atol=1e-6)


def test_known_pixel_imagenet_normalization(synthetic_hdf5_known_pixels):
    """Verify exact ImageNet-normalized values from known pixels."""
    ds = Stage1Dataset(synthetic_hdf5_known_pixels, split="train")
    sample = ds[0]

    # Raw: [128/255, 64/255, 255/255]
    raw_r = 128.0 / 255.0
    raw_g = 64.0 / 255.0
    raw_b = 255.0 / 255.0

    # ImageNet normalized
    exp_r = (raw_r - 0.485) / 0.229
    exp_g = (raw_g - 0.456) / 0.224
    exp_b = (raw_b - 0.406) / 0.225

    enc = sample["images_enc"]
    assert enc[:, 0, 0, 0].allclose(torch.tensor(exp_r), atol=1e-5)
    assert enc[:, 1, 0, 0].allclose(torch.tensor(exp_g), atol=1e-5)
    assert enc[:, 2, 0, 0].allclose(torch.tensor(exp_b), atol=1e-5)


def test_known_pixel_gradient_pattern(synthetic_hdf5_known_pixels):
    """Timestep 1 has a width-gradient in red channel. Verify it."""
    ds = Stage1Dataset(synthetic_hdf5_known_pixels, split="train")
    sample = ds[1]  # timestep 1

    target = sample["images_target"]  # (K, 3, H, W)
    # Red channel should increase across width (w % 256 / 255)
    red = target[0, 0, 0, :]  # width dimension
    for w in range(224):
        expected = (w % 256) / 255.0
        assert abs(red[w].item() - expected) < 1e-5, f"Mismatch at w={w}"


def test_known_pixel_green_blue_zero_in_gradient(synthetic_hdf5_known_pixels):
    """Timestep 1 gradient: green and blue channels should be zero."""
    ds = Stage1Dataset(synthetic_hdf5_known_pixels, split="train")
    sample = ds[1]
    target = sample["images_target"]
    assert target[0, 1, :, :].abs().sum() == 0  # green
    assert target[0, 2, :, :].abs().sum() == 0  # blue


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_reads(synthetic_hdf5_uint8):
    """Same index should return identical data every time."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    s1 = ds[5]
    s2 = ds[5]
    assert torch.equal(s1["images_enc"], s2["images_enc"])
    assert torch.equal(s1["images_target"], s2["images_target"])
    assert torch.equal(s1["view_present"], s2["view_present"])


def test_deterministic_after_other_accesses(synthetic_hdf5_uint8):
    """Reading other samples in between should not affect reproducibility."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    s_before = ds[3]
    _ = ds[0]
    _ = ds[19]
    _ = ds[7]
    s_after = ds[3]
    assert torch.equal(s_before["images_target"], s_after["images_target"])


def test_two_dataset_instances_agree(synthetic_hdf5_uint8):
    """Two Dataset instances on same file should return identical data."""
    ds1 = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    ds2 = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    for idx in [0, 5, 15, 19]:
        assert torch.equal(ds1[idx]["images_target"], ds2[idx]["images_target"])
        assert torch.equal(ds1[idx]["images_enc"], ds2[idx]["images_enc"])


# ---------------------------------------------------------------------------
# DataLoader stress tests
# ---------------------------------------------------------------------------

def test_dataloader_num_workers_zero(synthetic_hdf5_uint8):
    """num_workers=0 (main process) should work fine."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert batch["images_enc"].shape == (4, 4, 3, 224, 224)


def test_dataloader_batch_larger_than_dataset(synthetic_hdf5_uint8):
    """Batch size > dataset length should still work (partial last batch)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="valid")  # 10 samples
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    batch = next(iter(loader))
    assert batch["images_enc"].shape[0] == 10


def test_dataloader_drop_last(synthetic_hdf5_uint8):
    """drop_last=True should drop incomplete batches."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")  # 20 samples
    loader = torch.utils.data.DataLoader(ds, batch_size=6, drop_last=True)
    total = sum(b["images_enc"].shape[0] for b in loader)
    assert total == 18  # 3 full batches of 6


def test_dataloader_multiple_epochs(synthetic_hdf5_uint8):
    """Multiple epochs should produce identical data (no state leak)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    epoch1 = next(iter(loader))
    epoch2 = next(iter(loader))
    assert torch.equal(epoch1["images_target"], epoch2["images_target"])


# ---------------------------------------------------------------------------
# Gradient flow through dataset outputs
# ---------------------------------------------------------------------------

def test_gradient_flow_through_target(synthetic_hdf5_uint8):
    """Verify that dataset targets can participate in loss + backward."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    pred = torch.randn(1, 3, 224, 224, requires_grad=True)
    target = sample["images_target"][0:1]
    loss = l1_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == (1, 3, 224, 224)


def test_gradient_flow_through_encoder_images(synthetic_hdf5_uint8):
    """Encoder images should support gradient flow if needed."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    # Simulate adapter processing encoder tokens
    enc_input = sample["images_enc"][0:1]  # (1, 3, 224, 224)
    adapter = nn.Conv2d(3, 3, 1)
    output = adapter(enc_input)
    loss = output.mean()
    loss.backward()
    assert list(adapter.parameters())[0].grad is not None


def test_gradient_accumulation_across_views(synthetic_hdf5_uint8):
    """Gradients should accumulate correctly across multiple views."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    decoder = nn.Conv2d(3, 3, 1)
    total_loss = torch.tensor(0.0)

    for k in range(4):
        target = sample["images_target"][k:k+1]
        pred = decoder(torch.randn(1, 3, 224, 224))
        total_loss = total_loss + l1_loss(pred, target)

    total_loss.backward()
    grad = list(decoder.parameters())[0].grad
    assert grad is not None
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Batch-level statistics
# ---------------------------------------------------------------------------

def test_batch_target_statistics(synthetic_hdf5_uint8):
    """Batch of targets should have reasonable mean (around 0.5 for uniform uint8)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    batch = next(iter(loader))
    mean_val = batch["images_target"].mean().item()
    # Random uint8 uniform: expected mean ≈ 127.5/255 ≈ 0.5
    assert 0.3 < mean_val < 0.7, f"Unexpected batch mean: {mean_val}"


def test_batch_encoder_statistics(synthetic_hdf5_uint8):
    """Encoder images should have mean shifted away from 0.5 by ImageNet norm."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    batch = next(iter(loader))
    # After ImageNet norm of ~U(0,1), mean ≈ (0.5 - mean) / std per channel
    # Overall mean should NOT be ~0.5 (that would mean no normalization)
    mean_enc = batch["images_enc"].mean().item()
    mean_tgt = batch["images_target"].mean().item()
    assert abs(mean_enc - mean_tgt) > 0.1, "Encoder images not properly normalized"


def test_per_channel_stats(synthetic_hdf5_uint8):
    """Each channel should have distinct statistics after ImageNet norm."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=20, shuffle=False)
    batch = next(iter(loader))
    enc = batch["images_enc"]  # (B, K, 3, H, W)
    ch_means = [enc[:, :, c, :, :].mean().item() for c in range(3)]
    # ImageNet mean/std differ per channel, so normalized means should differ
    assert len(set(round(m, 2) for m in ch_means)) >= 2, "Channels too similar"


# ---------------------------------------------------------------------------
# HDF5 re-open safety
# ---------------------------------------------------------------------------

def test_multiple_sequential_reads(synthetic_hdf5_uint8):
    """Many sequential reads should all succeed (HDF5 open/close per read)."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    for _ in range(50):
        idx = np.random.randint(len(ds))
        sample = ds[idx]
        assert sample["images_target"].shape == (4, 3, 224, 224)


def test_interleaved_train_valid_reads(synthetic_hdf5_uint8):
    """Interleaved reads from train and valid datasets on same file."""
    ds_train = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    ds_valid = Stage1Dataset(synthetic_hdf5_uint8, split="valid")
    for _ in range(20):
        t_idx = np.random.randint(len(ds_train))
        v_idx = np.random.randint(len(ds_valid))
        st = ds_train[t_idx]
        sv = ds_valid[v_idx]
        assert st["images_target"].shape == sv["images_target"].shape


# ---------------------------------------------------------------------------
# Integration: full gen+disc step on dataset batch
# ---------------------------------------------------------------------------

def test_gen_disc_step_on_batch(synthetic_hdf5_uint8, lpips_net):
    """Simulate a full generator + discriminator training step on a batch."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    disc = PatchDiscriminator(pretrained=False)
    vp = batch["view_present"]  # (B, K)
    targets = batch["images_target"]  # (B, K, 3, H, W)

    B, K = vp.shape
    mask = vp.view(-1)
    all_targets = targets.view(B * K, 3, 224, 224)
    real_targets = all_targets[mask]

    # Simulate decoder predictions
    preds = torch.randn_like(real_targets).clamp(0, 1)

    # L1 loss
    loss_l1 = l1_loss(preds, real_targets)
    assert torch.isfinite(loss_l1)

    # Disc forward
    logits_real = disc(real_targets)
    logits_fake = disc(preds.detach())
    d_loss = gan_discriminator_loss(logits_real, logits_fake)
    assert torch.isfinite(d_loss)
    assert d_loss.item() >= 0


def test_full_recon_loss_on_batch(synthetic_hdf5_uint8, lpips_net):
    """reconstruction_loss (L1 + LPIPS, no GAN) on batched dataset output."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    # Process just one view from each sample
    targets = batch["images_target"][:, 0, :, :, :]  # (B, 3, H, W)
    preds = torch.randn_like(targets).clamp(0, 1)

    loss = reconstruction_loss(
        preds, targets, lpips_net,
        use_gan=False, logits_fake=None, last_layer_weight=None,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Parametrized: all fixtures × all properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("split", ["train", "valid"])
def test_both_splits_work(synthetic_hdf5_uint8, split):
    """Both train and valid splits should produce valid outputs."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split=split)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["images_target"].min() >= 0.0
    assert sample["images_target"].max() <= 1.0


@pytest.mark.parametrize("view_idx", [0, 1, 2, 3])
def test_each_view_independently(synthetic_hdf5_uint8, view_idx):
    """Each camera view should be independently valid."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]
    view_enc = sample["images_enc"][view_idx]  # (3, H, W)
    view_tgt = sample["images_target"][view_idx]  # (3, H, W)

    assert view_enc.shape == (3, 224, 224)
    assert view_tgt.shape == (3, 224, 224)
    assert torch.isfinite(view_enc).all()
    assert view_tgt.min() >= 0.0
    assert view_tgt.max() <= 1.0


@pytest.mark.parametrize("batch_size", [1, 2, 5, 10, 20])
def test_various_batch_sizes(synthetic_hdf5_uint8, batch_size):
    """DataLoader should handle various batch sizes."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    expected_b = min(batch_size, len(ds))
    assert batch["images_enc"].shape[0] == expected_b


# ---------------------------------------------------------------------------
# Encoder-input compatibility
# ---------------------------------------------------------------------------

def test_encoder_input_compatible_with_vit(synthetic_hdf5_uint8):
    """images_enc should be directly feedable to a ViT-style model.
    Verify shape and range are suitable for patch embedding."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    enc = sample["images_enc"]  # (K, 3, 224, 224)
    # ViT expects (B, 3, 224, 224)
    # Flatten views into batch dim
    B_views = enc.shape[0]  # K
    assert enc.shape[1:] == (3, 224, 224)

    # Simple patch embedding simulation (16x16 patches)
    patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
    tokens = patch_embed(enc)  # (K, 768, 14, 14)
    assert tokens.shape == (B_views, 768, 14, 14)


def test_target_compatible_with_pixel_decoder(synthetic_hdf5_uint8):
    """images_target should match expected decoder output shape."""
    ds = Stage1Dataset(synthetic_hdf5_uint8, split="train")
    sample = ds[0]

    target = sample["images_target"]  # (K, 3, 224, 224)
    # Simulate decoder output
    decoder_out = torch.sigmoid(torch.randn(4, 3, 224, 224))  # (K, 3, H, W)
    # L1 loss should work directly
    loss = l1_loss(decoder_out, target)
    assert torch.isfinite(loss)
