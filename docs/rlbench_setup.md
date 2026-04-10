# RLBench Setup Guide (PC03)

## Prerequisites

- Ubuntu 20.04+, NVIDIA GPU
- Python 3.10 venv at `~/venv_rlbench`
- CoppeliaSim V4.1.0 installed

## Environment Variables

Add these to your shell before any RLBench operation:

```bash
source ~/venv_rlbench/bin/activate
export COPPELIASIM_ROOT=/virtual/csc415user/coppeliasim/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
```

## Headless Display

CoppeliaSim needs a display. For **GT replay without cameras** (no rendering):

```bash
export QT_QPA_PLATFORM=offscreen
```

For **training eval with cameras** (needs OpenGL):

```bash
# Start Xvfb (once per session)
/virtual/csc415user/Xvfb :99 -screen 0 1024x768x24 &

export DISPLAY=:99
unset QT_QPA_PLATFORM

# Verify display and OpenGL are working
python -c "import pyrep; print('PyRep Display Boot OK')"
```

If Xvfb won't start ("Server is already active"), it's already running. If the lock is stale:

```bash
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99
/virtual/csc415user/Xvfb :99 -screen 0 1024x768x24 &
```

## Verify Installation

```bash
python -c "import pyrep; print('PyRep OK')"
python -c "import rlbench; print('RLBench OK')"
```

## Training

```bash
cd ~/RAEDiTRobotics

PYTHONPATH=. python training/train_v3_script.py \
    --hdf5 /virtual/csc415user/data/rlbench/open_drawer_tokens_bf16_none.hdf5 \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --eval_hdf5 /virtual/csc415user/data/rlbench/open_drawer.hdf5 \
    --eval_task open_drawer --eval_mode rlbench \
    --eval_exec_horizon 1 --eval_every_epoch 50 \
    --eval_episodes 10 \
    --ac_dim 10 --proprio_dim 8 --n_active_cams 4 \
    --norm_mode chi --no_amp --no_compile \
    --batch_size 64 --num_epochs 3000 \
    --save_dir checkpoints/v3_rlbench_open_drawer \
    --save_every_epoch 50
```

To resume from a checkpoint, add: `--resume checkpoints/v3_rlbench_open_drawer/epoch_249.pt`

## Standalone Evaluation (with video)

```bash
PYTHONPATH=. python training/eval_v3_rlbench.py \
    --checkpoint checkpoints/v3_rlbench_open_drawer/best.pt \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --eval_hdf5 /virtual/csc415user/data/rlbench/open_drawer.hdf5 \
    --task open_drawer \
    --num_episodes 10 \
    --ac_dim 10 --proprio_dim 8 --n_active_cams 4 --T_pred 10 \
    --save_video --video_dir checkpoints/v3_rlbench_open_drawer/media/eval

# Without video (faster): remove --save_video --video_dir flags
```

Videos are saved as `ep000_fail.mp4`, `ep001_success.mp4`, etc. (4 cameras side-by-side, 10fps).

## GT Replay (data verification)

Verifies HDF5 data chain is lossless. Requires raw demo pickles.

```bash
export QT_QPA_PLATFORM=offscreen  # no cameras needed

PYTHONPATH=. python training/gt_replay_rlbench.py \
    --task open_drawer \
    --hdf5 /virtual/csc415user/data/rlbench/open_drawer.hdf5 \
    --pickles /virtual/csc415user/data/rlbench/train/open_drawer/all_variations/episodes \
    --num_demos 20 --split train
```

Expected: ~50% success (CoppeliaSim physics non-determinism ceiling).

## Data Locations

| What | Path |
|------|------|
| HDF5 files | `/virtual/csc415user/data/rlbench/{task}.hdf5` |
| Precomputed tokens | `/virtual/csc415user/data/rlbench/{task}_tokens_bf16_none.hdf5` |
| Train pickles | `/virtual/csc415user/data/rlbench/train/{task}/all_variations/episodes/` |
| Valid pickles | `/virtual/csc415user/data/rlbench/valid/{task}/all_variations/episodes/` |
| Checkpoints | `checkpoints/v3_rlbench_{task}/` |
| Xvfb binary | `/virtual/csc415user/Xvfb` |

## Troubleshooting

| Error | Fix |
|-------|-----|
| `libcoppeliaSim.so.1: cannot open` | Set `LD_LIBRARY_PATH` to include `$COPPELIASIM_ROOT` |
| `Qt platform plugin "offscreen" not found` | Set `QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT` |
| OpenGL segfault / `createPlatformOpenGLContext` | Use Xvfb (`DISPLAY=:99`) instead of `QT_QPA_PLATFORM=offscreen` |
| `V-REP side. Return value: -1` | CoppeliaSim calls on wrong thread — use `signal.alarm` not `threading` |
| `SlideBlockToColorTarget` not found | Patch stepjam RLBench with peract fork `.py` + `.ttm` |
| Eval fails with `os referenced before assignment` | Pull latest — fixed in commit |
