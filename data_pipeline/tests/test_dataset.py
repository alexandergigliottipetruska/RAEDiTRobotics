"""Smoke tests for MultiViewManipulationDataset (spec section 8.3).

Run with:
  pytest data_pipeline/tests/test_dataset.py -v

Uses the same --hdf5 / hdf5_path fixture from conftest.py.
Default: lift/ph unified HDF5.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.datasets.base_dataset import MultiViewManipulationDataset
from data_pipeline.conversion.unified_schema import NUM_CAMERA_SLOTS, IMAGE_SIZE


T_OBS  = 2
T_PRED = 50


@pytest.fixture(scope="session")
def dataset(hdf5_path):
    return MultiViewManipulationDataset(hdf5_path, split="train", T_obs=T_OBS, T_pred=T_PRED)


@pytest.fixture(scope="session")
def sample(dataset):
    return dataset[0]


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

def test_images_shape(sample, hdf5_path):
    H, W = IMAGE_SIZE
    K = NUM_CAMERA_SLOTS
    assert sample["images"].shape == (T_OBS, K, 3, H, W), (
        f"Expected images ({T_OBS}, {K}, 3, {H}, {W}), got {sample['images'].shape}"
    )


def test_actions_shape(sample, dataset):
    assert sample["actions"].shape == (T_PRED, dataset.action_dim), (
        f"Expected actions ({T_PRED}, {dataset.action_dim}), got {sample['actions'].shape}"
    )


def test_proprio_shape(sample, dataset):
    assert sample["proprio"].shape == (T_OBS, dataset.proprio_dim), (
        f"Expected proprio ({T_OBS}, {dataset.proprio_dim}), got {sample['proprio'].shape}"
    )


def test_view_present_shape(sample):
    assert sample["view_present"].shape == (NUM_CAMERA_SLOTS,), (
        f"Expected view_present ({NUM_CAMERA_SLOTS},), got {sample['view_present'].shape}"
    )


# ---------------------------------------------------------------------------
# Value range tests
# ---------------------------------------------------------------------------

def test_image_value_range(sample):
    """Images should be in approximately [-2.1, 2.6] after ImageNet normalization.
    Exact bounds: (0 - 0.406) / 0.225 ≈ -1.8, (1 - 0.406) / 0.225 ≈ 2.64
    Padded (zero) slots give negative values; real slots span a broader range.
    """
    imgs = sample["images"].numpy()
    assert imgs.min() >= -2.2, f"Image min {imgs.min():.3f} below expected floor"
    assert imgs.max() <= 2.7,  f"Image max {imgs.max():.3f} above expected ceiling"


def test_image_dtype(sample):
    assert sample["images"].dtype == torch.float32


def test_actions_dtype(sample):
    assert sample["actions"].dtype == torch.float32


# ---------------------------------------------------------------------------
# view_present flags
# ---------------------------------------------------------------------------

def test_view_present_flags(sample):
    """For robomimic: slots 0 and 3 are real, slots 1 and 2 are padded."""
    vp = sample["view_present"].numpy()
    assert vp[0] == True,  "Slot 0 (agentview) should be present"
    assert vp[1] == False, "Slot 1 should be absent"
    assert vp[2] == False, "Slot 2 should be absent"
    assert vp[3] == True,  "Slot 3 (wrist) should be present"


def test_imagenet_normalization_applied(dataset):
    """Verify ImageNet normalization was applied (not just [0,1] scaling).

    Unnormalized float32 images live strictly in [0, 1]. After ImageNet norm,
    any scene with real contrast will have pixels both above 1.0 (bright areas)
    and below 0.0 (dark areas) — this is impossible for raw [0,1] data.
    Only checks real camera slots.
    """
    sample = dataset[len(dataset) // 2]
    imgs = sample["images"].numpy()  # (T_obs, K, 3, H, W)
    vp   = sample["view_present"].numpy()
    real_slots = [i for i in range(NUM_CAMERA_SLOTS) if vp[i]]
    real_imgs  = imgs[:, real_slots]  # (T_obs, n_real, 3, H, W)
    # After ImageNet normalization, values must extend outside [0, 1]
    outside_unit = (real_imgs.min() < 0.0) or (real_imgs.max() > 1.0)
    assert outside_unit, (
        f"Image values in [{real_imgs.min():.3f}, {real_imgs.max():.3f}] — "
        f"still within [0,1], suggesting ImageNet normalization was NOT applied"
    )


def test_padded_slots_are_negative(sample):
    """Padded slots (zero pixels) normalize to ~[-1.8, -1.6] range."""
    imgs = sample["images"].numpy()
    vp   = sample["view_present"].numpy()
    for slot in range(NUM_CAMERA_SLOTS):
        if not vp[slot]:
            # Padded pixels are 0.0 in [0,1]; after ImageNet norm they're negative
            slot_imgs = imgs[:, slot]
            assert slot_imgs.max() < 0.0, (
                f"Padded slot {slot} has unexpected positive values: {slot_imgs.max():.3f}"
            )


# ---------------------------------------------------------------------------
# Dataset length and indexing
# ---------------------------------------------------------------------------

def test_dataset_length(dataset, hdf5_path):
    """Dataset length should equal total train timesteps across all demos."""
    import h5py
    from data_pipeline.conversion.unified_schema import read_mask
    with h5py.File(hdf5_path, "r") as f:
        keys = read_mask(f, "train")
        total_T = sum(f[f"data/{k}/actions"].shape[0] for k in keys)
    assert len(dataset) == total_T, (
        f"Dataset length {len(dataset)} != total train timesteps {total_T}"
    )


def test_last_sample_accessible(dataset):
    """Last sample should not raise."""
    _ = dataset[len(dataset) - 1]


def test_terminal_padding(dataset, hdf5_path):
    """At end of a demo, repeated action should match the last raw action."""
    import h5py
    from data_pipeline.conversion.unified_schema import read_mask
    from data_pipeline.conversion.compute_norm_stats import load_norm_stats

    with h5py.File(hdf5_path, "r") as f:
        keys = read_mask(f, "train")
        key = keys[0]
        T = f[f"data/{key}/actions"].shape[0]
        last_raw = f[f"data/{key}/actions"][T - 1].astype(np.float32)

    stats = load_norm_stats(hdf5_path)
    last_normalized = (last_raw - stats["actions"]["mean"]) / stats["actions"]["std"]

    # Find the sample index for the last timestep of demo 0
    last_idx = T - 1  # dataset is built in order, so first T entries are demo 0
    sample = dataset[last_idx]
    # The final action in the chunk should all equal last_normalized
    np.testing.assert_allclose(
        sample["actions"].numpy()[-1], last_normalized, rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# DataLoader compatibility
# ---------------------------------------------------------------------------

def test_dataloader_batch(dataset):
    """Should be able to collate a batch of 4 without errors."""
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    H, W = IMAGE_SIZE
    K = NUM_CAMERA_SLOTS
    assert batch["images"].shape  == (4, T_OBS, K, 3, H, W)
    assert batch["actions"].shape == (4, T_PRED, dataset.action_dim)
