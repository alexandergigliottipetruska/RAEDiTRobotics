"""Phase 0 validation script for a converted RLBench unified HDF5 file.

Checks every structural requirement needed for the training and evaluation
pipeline to load the file correctly. Prints a clear PASS / FAIL for each
check and exits with code 1 if anything fails.

Usage:
    python data_pipeline/conversion/validate_rlbench_hdf5.py \
        --hdf5 data/unified/rlbench/reach_target.hdf5

Run from the repo root.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Expected constants (must match converter and pipeline config)
# ---------------------------------------------------------------------------

EXPECTED_ACTION_DIM   = 8     # [pos(3), quat_xyzw(4), gripper(1)]
EXPECTED_PROPRIO_DIM  = 8     # [joint_positions(7), gripper_open(1)]
EXPECTED_NUM_CAMS     = 4     # front, left_shoulder, right_shoulder, wrist
EXPECTED_IMAGE_H      = 224
EXPECTED_IMAGE_W      = 224
EXPECTED_BENCHMARK    = "rlbench"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "  [PASS]"
_FAIL = "  [FAIL]"
_INFO = "  [INFO]"


def _check(label: str, condition: bool, detail: str = "") -> bool:
    tag  = _PASS if condition else _FAIL
    line = f"{tag}  {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return condition


def _header(title: str) -> None:
    bar = "-" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


# ---------------------------------------------------------------------------
# Check groups
# ---------------------------------------------------------------------------

def check_file_attrs(f: h5py.File) -> bool:
    _header("File-level attributes")
    ok = True

    benchmark = f.attrs.get("benchmark", "")
    ok &= _check(
        "benchmark == 'rlbench'",
        benchmark == EXPECTED_BENCHMARK,
        f"got '{benchmark}'"
    )

    task = f.attrs.get("task", "")
    ok &= _check(
        "task attr present",
        bool(task),
        f"'{task}'"
    )

    action_dim = int(f.attrs.get("action_dim", -1))
    ok &= _check(
        f"action_dim == {EXPECTED_ACTION_DIM}",
        action_dim == EXPECTED_ACTION_DIM,
        f"got {action_dim}"
    )

    proprio_dim = int(f.attrs.get("proprio_dim", -1))
    ok &= _check(
        f"proprio_dim == {EXPECTED_PROPRIO_DIM}",
        proprio_dim == EXPECTED_PROPRIO_DIM,
        f"got {proprio_dim}"
    )

    image_size = int(f.attrs.get("image_size", -1))
    ok &= _check(
        f"image_size == {EXPECTED_IMAGE_H}",
        image_size == EXPECTED_IMAGE_H,
        f"got {image_size}"
    )

    num_slots = int(f.attrs.get("num_cam_slots", -1))
    ok &= _check(
        f"num_cam_slots == {EXPECTED_NUM_CAMS}",
        num_slots == EXPECTED_NUM_CAMS,
        f"got {num_slots}"
    )

    return ok


def check_splits(f: h5py.File) -> tuple[bool, list[str], list[str]]:
    _header("Train / valid splits")
    ok = True

    has_mask = "mask" in f
    ok &= _check("mask group exists", has_mask)
    if not has_mask:
        return False, [], []

    train_keys = []
    valid_keys = []

    has_train = "mask/train" in f
    ok &= _check("mask/train dataset exists", has_train)
    if has_train:
        train_keys = [
            k.decode() if isinstance(k, bytes) else k
            for k in f["mask/train"][:]
        ]
        ok &= _check(
            "train split is non-empty",
            len(train_keys) > 0,
            f"{len(train_keys)} demos"
        )

    has_valid = "mask/valid" in f
    ok &= _check("mask/valid dataset exists", has_valid)
    if has_valid:
        valid_keys = [
            k.decode() if isinstance(k, bytes) else k
            for k in f["mask/valid"][:]
        ]
        ok &= _check(
            "valid split is non-empty",
            len(valid_keys) > 0,
            f"{len(valid_keys)} demos"
        )

    print(f"{_INFO}  Total demos: train={len(train_keys)}, valid={len(valid_keys)}")
    return ok, train_keys, valid_keys


def check_norm_stats(f: h5py.File) -> bool:
    _header("Normalisation statistics")
    ok = True

    has_ns = "norm_stats" in f
    ok &= _check("norm_stats group exists", has_ns)
    if not has_ns:
        return False

    for field in ("actions", "proprio"):
        for stat in ("mean", "std", "min", "max"):
            path = f"norm_stats/{field}/{stat}"
            present = path in f
            ok &= _check(f"{path} exists", present)
            if present:
                arr = f[path][:]
                expected_dim = (
                    EXPECTED_ACTION_DIM if field == "actions"
                    else EXPECTED_PROPRIO_DIM
                )
                ok &= _check(
                    f"{path} shape == ({expected_dim},)",
                    arr.shape == (expected_dim,),
                    f"got {arr.shape}"
                )
                ok &= _check(
                    f"{path} has no NaN/Inf",
                    bool(np.isfinite(arr).all()),
                    f"values: {np.round(arr, 4)}"
                )

    # Sanity: std > 0 for actions (catches zero-filled or constant actions)
    if "norm_stats/actions/std" in f:
        std = f["norm_stats/actions/std"][:]
        ok &= _check(
            "action std > 0 for all dims",
            bool((std > 0).all()),
            f"std: {np.round(std, 4)}"
        )

    return ok


def check_demo(
    f: h5py.File,
    demo_key: str,
    is_sample: bool = False,
) -> bool:
    """Check a single demo group for correct shapes and dtypes."""
    label = f"{'[sampled] ' if is_sample else ''}demo: {demo_key}"
    ok = True

    path = f"data/{demo_key}"
    if path not in f:
        print(f"{_FAIL}  {label}: group missing")
        return False

    grp = f[path]

    # --- images ---
    has_images = "images" in grp
    ok &= _check(f"{label}/images exists", has_images)
    if has_images:
        imgs = grp["images"]
        T = imgs.shape[0]
        ok &= _check(
            f"{label}/images ndim == 5",
            imgs.ndim == 5,
            f"shape {imgs.shape}"
        )
        if imgs.ndim == 5:
            _, K, H, W, C = imgs.shape
            ok &= _check(
                f"{label}/images cameras == {EXPECTED_NUM_CAMS}",
                K == EXPECTED_NUM_CAMS,
                f"got {K}"
            )
            ok &= _check(
                f"{label}/images spatial == {EXPECTED_IMAGE_H}x{EXPECTED_IMAGE_W}",
                H == EXPECTED_IMAGE_H and W == EXPECTED_IMAGE_W,
                f"got {H}x{W}"
            )
            ok &= _check(
                f"{label}/images channels == 3",
                C == 3,
                f"got {C}"
            )
        ok &= _check(
            f"{label}/images dtype == uint8",
            imgs.dtype == np.uint8,
            f"got {imgs.dtype}"
        )
        if is_sample and has_images and imgs.ndim == 5:
            # Check images are not all zeros (converter bug guard)
            sample_frame = imgs[0, 0]  # [H, W, 3]
            ok &= _check(
                f"{label}/images frame[0,cam0] not all-zero",
                bool(sample_frame.max() > 0),
            )
    else:
        T = None

    # --- actions ---
    has_actions = "actions" in grp
    ok &= _check(f"{label}/actions exists", has_actions)
    if has_actions:
        acts = grp["actions"]
        ok &= _check(
            f"{label}/actions ndim == 2",
            acts.ndim == 2,
            f"shape {acts.shape}"
        )
        if acts.ndim == 2:
            ok &= _check(
                f"{label}/actions dim == {EXPECTED_ACTION_DIM}",
                acts.shape[1] == EXPECTED_ACTION_DIM,
                f"got {acts.shape[1]}"
            )
            if T is not None:
                ok &= _check(
                    f"{label}/actions T matches images",
                    acts.shape[0] == T,
                    f"actions T={acts.shape[0]}, images T={T}"
                )
        ok &= _check(
            f"{label}/actions dtype == float32",
            acts.dtype == np.float32,
            f"got {acts.dtype}"
        )
        if is_sample and has_actions and acts.ndim == 2:
            ok &= _check(
                f"{label}/actions not all-zero",
                bool(np.abs(acts[:]).max() > 0),
            )
            # Quaternion norm check: quat_xyzw columns 3:7 should be unit vectors
            quats = acts[:, 3:7]
            norms = np.linalg.norm(quats, axis=1)
            ok &= _check(
                f"{label}/quaternions are unit-norm (tol 0.01)",
                bool(np.abs(norms - 1.0).max() < 0.01),
                f"max deviation: {np.abs(norms - 1.0).max():.4f}"
            )
            # Gripper column (last) should be binary {0, 1}
            gripper_vals = np.unique(acts[:, -1])
            ok &= _check(
                f"{label}/gripper values are binary {{0, 1}}",
                bool(np.all(np.isin(gripper_vals, [0.0, 1.0]))),
                f"unique: {gripper_vals}"
            )

    # --- proprio ---
    has_proprio = "proprio" in grp
    ok &= _check(f"{label}/proprio exists", has_proprio)
    if has_proprio:
        prop = grp["proprio"]
        ok &= _check(
            f"{label}/proprio ndim == 2",
            prop.ndim == 2,
            f"shape {prop.shape}"
        )
        if prop.ndim == 2:
            ok &= _check(
                f"{label}/proprio dim == {EXPECTED_PROPRIO_DIM}",
                prop.shape[1] == EXPECTED_PROPRIO_DIM,
                f"got {prop.shape[1]}"
            )

    # --- view_present ---
    has_vp = "view_present" in grp
    ok &= _check(f"{label}/view_present exists", has_vp)
    if has_vp:
        vp = grp["view_present"][:]
        ok &= _check(
            f"{label}/view_present shape == ({EXPECTED_NUM_CAMS},)",
            vp.shape == (EXPECTED_NUM_CAMS,),
            f"got {vp.shape}"
        )
        ok &= _check(
            f"{label}/view_present all True (all 4 cameras present)",
            bool(vp.all()),
            f"got {vp}"
        )

    return ok


def check_all_demos(
    f: h5py.File,
    train_keys: list[str],
    valid_keys: list[str],
) -> bool:
    _header("Demo shape/dtype checks (all demos, shapes only)")
    ok = True

    # Check all demos for shape/dtype — but only read actual data for one sample
    rng = np.random.default_rng(0)
    all_keys = train_keys + valid_keys
    sample_key = rng.choice(all_keys) if all_keys else None

    shape_failures = []
    for key in all_keys:
        passed = check_demo(f, key, is_sample=False)
        if not passed:
            shape_failures.append(key)

    ok &= _check(
        f"All {len(all_keys)} demos pass shape/dtype checks",
        len(shape_failures) == 0,
        f"failures: {shape_failures[:5]}{'...' if len(shape_failures) > 5 else ''}"
    )

    if sample_key is not None:
        _header(f"Deep data check on sampled demo: {sample_key}")
        ok &= check_demo(f, sample_key, is_sample=True)

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(hdf5_path: str) -> bool:
    path = Path(hdf5_path)

    print(f"\n{'='*60}")
    print(f"  RLBench HDF5 Validation")
    print(f"  File: {path}")
    print(f"{'='*60}")

    if not path.exists():
        print(f"\n{_FAIL}  File does not exist: {path}")
        return False

    results = []

    with h5py.File(str(path), "r") as f:
        results.append(check_file_attrs(f))
        ok_splits, train_keys, valid_keys = check_splits(f)
        results.append(ok_splits)
        results.append(check_norm_stats(f))
        results.append(check_all_demos(f, train_keys, valid_keys))

    all_passed = all(results)

    print(f"\n{'='*60}")
    if all_passed:
        print("  RESULT:  ALL CHECKS PASSED — file is ready for training")
    else:
        print("  RESULT:  ONE OR MORE CHECKS FAILED — see [FAIL] lines above")
    print(f"{'='*60}\n")

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a converted RLBench unified HDF5 file (Phase 0)."
    )
    parser.add_argument(
        "--hdf5", required=True,
        help="Path to the unified HDF5 file to validate.",
    )
    args = parser.parse_args()

    passed = validate(args.hdf5)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
