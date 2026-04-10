"""Unit tests for unified HDF5 schema and conversion outputs.

Run with:
  pytest data_pipeline/tests/test_schema.py -v

Tests run against the lift/ph unified HDF5 by default.
Override path with: pytest --hdf5 path/to/other.hdf5
"""

import pytest
import numpy as np
import h5py

from data_pipeline.conversion.unified_schema import (
    NUM_CAMERA_SLOTS,
    IMAGE_SIZE,
    read_mask,
)
from data_pipeline.conversion.compute_norm_stats import load_norm_stats

@pytest.fixture(scope="session")
def hdf5_file(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        yield f


@pytest.fixture(scope="session")
def train_keys(hdf5_file):
    return read_mask(hdf5_file, "train")


@pytest.fixture(scope="session")
def valid_keys(hdf5_file):
    return read_mask(hdf5_file, "valid")


@pytest.fixture(scope="session")
def first_demo(hdf5_file, train_keys):
    return hdf5_file[f"data/{train_keys[0]}"]


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_image_shape(first_demo):
    T = first_demo["actions"].shape[0]
    H, W = IMAGE_SIZE
    K = NUM_CAMERA_SLOTS
    assert first_demo["images"].shape == (T, K, H, W, 3), (
        f"Expected images shape ({T}, {K}, {H}, {W}, 3), got {first_demo['images'].shape}"
    )


def test_action_shape(first_demo, hdf5_file):
    T = first_demo["actions"].shape[0]
    action_dim = hdf5_file.attrs["action_dim"]
    assert first_demo["actions"].shape == (T, action_dim), (
        f"Expected actions shape ({T}, {action_dim}), got {first_demo['actions'].shape}"
    )


def test_proprio_shape(first_demo, hdf5_file):
    T = first_demo["actions"].shape[0]
    D = hdf5_file.attrs["proprio_dim"]
    assert first_demo["proprio"].shape == (T, D), (
        f"Expected proprio shape ({T}, {D}), got {first_demo['proprio'].shape}"
    )


def test_view_present_shape(first_demo):
    assert first_demo["view_present"].shape == (NUM_CAMERA_SLOTS,), (
        f"Expected view_present shape ({NUM_CAMERA_SLOTS},), got {first_demo['view_present'].shape}"
    )


# ---------------------------------------------------------------------------
# view_present / camera content tests
# ---------------------------------------------------------------------------

def test_view_flags(hdf5_file, train_keys):
    """All demos must have identical view_present flags."""
    ref = hdf5_file[f"data/{train_keys[0]}"]["view_present"][:]
    for key in train_keys:
        flags = hdf5_file[f"data/{key}"]["view_present"][:]
        assert np.array_equal(flags, ref), (
            f"Demo {key} has different view_present flags: {flags} vs {ref}"
        )


def test_padded_cameras_are_zeros(first_demo):
    """Camera slots with view_present=False must be all zeros."""
    view_present = first_demo["view_present"][:]
    images = first_demo["images"][:]
    for slot in range(NUM_CAMERA_SLOTS):
        if not view_present[slot]:
            assert images[:, slot].sum() == 0.0, (
                f"Slot {slot} is padded but contains non-zero pixels"
            )


def test_real_cameras_nonzero(first_demo):
    """Camera slots with view_present=True must have non-zero pixels."""
    view_present = first_demo["view_present"][:]
    images = first_demo["images"][:]
    for slot in range(NUM_CAMERA_SLOTS):
        if view_present[slot]:
            assert images[:, slot].sum() > 0.0, (
                f"Slot {slot} is marked real but contains only zeros"
            )


def test_image_range(first_demo):
    """Images must be float32 in [0, 1] or uint8 in [0, 255]."""
    images = first_demo["images"][:]
    if images.dtype == np.float32:
        assert images.min() >= 0.0, f"Image min {images.min()} < 0"
        assert images.max() <= 1.0, f"Image max {images.max()} > 1"
    elif images.dtype == np.uint8:
        assert images.min() >= 0, f"Image min {images.min()} < 0"
        assert images.max() <= 255, f"Image max {images.max()} > 255"
    else:
        pytest.fail(f"Unexpected image dtype: {images.dtype}")


# ---------------------------------------------------------------------------
# Action range test
# ---------------------------------------------------------------------------

def test_action_ranges(hdf5_file, train_keys):
    """Actions must be finite (no NaN/Inf). Absolute actions can exceed [-1,1]."""
    for key in train_keys:
        actions = hdf5_file[f"data/{key}"]["actions"][:]
        assert np.isfinite(actions).all(), f"Demo {key}: actions contain NaN/Inf"


# ---------------------------------------------------------------------------
# Norm stats tests
# ---------------------------------------------------------------------------

def test_normalized_train_stats(hdf5_path, hdf5_file, train_keys):
    """After z-scoring with saved stats, train actions should have mean≈0, std≈1 per dim."""
    stats = load_norm_stats(hdf5_path)
    a_mean = stats["actions"]["mean"]
    a_std = stats["actions"]["std"]

    all_actions = np.concatenate(
        [hdf5_file[f"data/{k}"]["actions"][:] for k in train_keys], axis=0
    )
    normalized = (all_actions - a_mean) / a_std
    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(normalized.std(axis=0), 1.0, atol=1e-3)

def test_norm_stats_exist(hdf5_file):
    assert "norm_stats" in hdf5_file
    assert "norm_stats/actions/mean" in hdf5_file
    assert "norm_stats/actions/std" in hdf5_file
    assert "norm_stats/proprio/mean" in hdf5_file
    assert "norm_stats/proprio/std" in hdf5_file


def test_norm_stats_shapes(hdf5_file):
    action_dim = hdf5_file.attrs["action_dim"]
    proprio_dim = hdf5_file.attrs["proprio_dim"]
    assert hdf5_file["norm_stats/actions/mean"].shape == (action_dim,)
    assert hdf5_file["norm_stats/actions/std"].shape == (action_dim,)
    assert hdf5_file["norm_stats/proprio/mean"].shape == (proprio_dim,)
    assert hdf5_file["norm_stats/proprio/std"].shape == (proprio_dim,)


def test_norm_stats_std_positive(hdf5_file):
    """Std must be positive (clipped to 1e-6 min)."""
    assert (hdf5_file["norm_stats/actions/std"][:] > 0).all()
    assert (hdf5_file["norm_stats/proprio/std"][:] > 0).all()


def test_normalization_roundtrip(hdf5_path, hdf5_file, train_keys):
    """Z-score normalizing then denormalizing recovers original values."""
    stats = load_norm_stats(hdf5_path)
    a_mean = stats["actions"]["mean"]
    a_std = stats["actions"]["std"]

    demo = hdf5_file[f"data/{train_keys[0]}"]
    actions = demo["actions"][:]
    normalized = (actions - a_mean) / a_std
    recovered = normalized * a_std + a_mean
    np.testing.assert_allclose(recovered, actions, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Train/valid split tests
# ---------------------------------------------------------------------------

def test_split_sizes(hdf5_file, train_keys, valid_keys):
    total = len(hdf5_file["data"])
    assert len(train_keys) + len(valid_keys) == total, (
        f"train ({len(train_keys)}) + valid ({len(valid_keys)}) != total ({total})"
    )


def test_no_data_leakage(train_keys, valid_keys):
    """No demo key should appear in both train and valid splits."""
    overlap = set(train_keys) & set(valid_keys)
    assert len(overlap) == 0, f"Leaking keys: {overlap}"


def test_all_demos_accessible(hdf5_file, train_keys, valid_keys):
    """Every key in the masks must exist as a group in data/."""
    for key in train_keys + valid_keys:
        assert f"data/{key}" in hdf5_file, f"Missing demo group: data/{key}"
