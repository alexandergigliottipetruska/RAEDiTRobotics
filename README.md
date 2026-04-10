<div align="center">

# RAE-DP: Representation Autoencoder Regularization in Diffusion Policy

**Frozen DINOv3-L Encoder with RAE-Pretrained Adapter for Visuomotor Control**

*Alexander Gigliotti Petruska, Naqeeb Ali, Jean de Nassau*
*University of Toronto -- CSC415 Course Project, 2026*

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

RAE-DP replaces end-to-end trained CNN encoders in diffusion policies with a **frozen DINOv3-L** Vision Transformer, bridged to the policy via a lightweight trainable adapter. The adapter is pre-trained using an RAE-style ViT decoder with L1 + LPIPS + GAN reconstruction losses, ensuring it retains fine-grained spatial detail before policy training. Combined with **L1 Flow Matching** for efficient 2-step inference and a **spatial observation encoder** that preserves the 7x7 grid structure of multi-view visual tokens, RAE-DP matches the performance of end-to-end trained baselines while converging **9x faster** thanks to pre-computed token caching.

---

## Key Results

| Method | Lift | Can | Square |
|--------|------|-----|--------|
| DP-CNN (e2e ResNet-18, Chi et al.) | 1.00 / 1.00 | 1.00 / 0.96 | 0.98 / 0.92 |
| DP-T (e2e ResNet-18, Chi et al.) | 1.00 / 1.00 | 1.00 / 0.98 | 1.00 / 0.90 |
| Frozen ViT-CLIP (Chi et al.) | -- | -- | 0.70 |
| Finetuned ViT-CLIP (Chi et al.) | -- | -- | 0.98 |
| **RAE-DP (ours, frozen DINOv3-L)** | **1.00 / 1.00** | **1.00 / 0.99** | **1.00 / 0.90** |

*Reported as max / avg of last 10 checkpoints. Chi et al. use 3 seeds, 3000 epochs. RAE-DP uses 5 seeds, 100 epochs.*

**Key findings:**
- Warm-start adapter pre-training provides a 19-point improvement on Can (79.5% -> 98.5% at epoch 19)
- RAE-DP reaches 95% on Can by epoch 5 vs epoch 50 for DP-T (10x faster convergence)
- d=512 denoiser needed for harder tasks (RLBench open_drawer: d=256 gets 0%, d=512 gets 80%)
- 7x7 spatial tokens with linear projection outperform MLP and Q-Former alternatives
- Pre-computed tokens provide **9x training speedup** (66s vs 597s per epoch on Square)
- 2-step flow matching inference is 50x faster than 100-step DDIM

---

## Architecture

```
Phase 1 -- Representation Pre-Training (RAE):
  Camera views -> [Frozen DINOv3-L] -> 196 tokens x 1024D
                    -> [Cancel-Affine LN] -> [Adapter MLP 1024->1024->512]
                    -> [ViT Decoder] -> Reconstruct image (L1 + LPIPS + GAN loss)

Phase 2 -- Policy Training (Flow Matching):
  Adapted tokens -> [AdaptiveAvgPool2d(7)] -> 49 tokens x 512D per camera
    -> [Linear(512, d)] + camera embeddings + spatial position embeddings
    -> [4 self-attention conditioning layers] (cross-camera reasoning)
    -> [Cross-attention Transformer Denoiser] (8 layers, 4 heads)
    -> L1 Flow Matching (sample prediction, logistic-normal timestep sampling)
    -> 2-step inference: half-step to midpoint, then direct prediction
```

### Module Map

```
RAEDiTRobotics/
  models/
    encoder.py              # FrozenMultiViewEncoder (DINOv3-L + cancel-affine LN)
    adapter.py              # TrainableAdapter (1024 -> 1024 -> 512, 2-layer MLP)
    decoder.py              # ViTDecoder (8-layer transformer, RAE reconstruction)
    discriminator.py        # GAN discriminator (Phase 1 auxiliary)
    stage1_bridge.py        # Stage1Bridge (loads Phase 1 weights, manages encoding)
    obs_encoder_v3.py       # ObservationEncoder (7x7 spatial pool + camera/pos embeddings)
    denoiser_transformer.py # TransformerDenoiser (Chi cross-attention architecture)
    denoiser_dit.py         # DiTDenoiser (adaLN-Zero, experimental)
    policy_v3.py            # PolicyDiTv3 (end-to-end policy with flow matching)
    ema_model.py            # EMAModel (adaptive power schedule)
    losses.py               # L1, LPIPS, GAN loss utilities
  training/
    train_v3.py             # Main training loop (V3Config dataclass)
    train_v3_script.py      # CLI entry point for Phase 2 training
    train_stage1_script.py  # CLI entry point for Phase 1 training
    eval_v3_robomimic.py    # Robomimic evaluation (Chi's protocol)
    eval_v3_rlbench.py      # RLBench evaluation (IK + temporal ensemble)
    precompute_tokens.py    # Cache DINOv3 tokens to HDF5
    plot_metrics.py         # Training curve visualization
    slim_checkpoints.py     # Strip frozen encoder from old EMA checkpoints
  data_pipeline/
    conversion/             # Raw -> unified HDF5 converters
    datasets/               # Stage1Dataset, Stage3Dataset
    envs/                   # Environment wrappers
    evaluation/             # Rollout, metrics, visualization
```

---

## Installation

> **Requirements:** Python >= 3.10, CUDA >= 11.8, ~16 GB VRAM (RTX 4080 or equivalent).

### Option A: venv (recommended)

```bash
git clone https://github.com/alexandergigliottipetruska/RAE-DP.git
cd RAEDiTRobotics
python -m venv venv
source venv/bin/activate
```

### Option B: conda

```bash
git clone https://github.com/alexandergigliottipetruska/RAE-DP.git
cd RAEDiTRobotics
conda create -n raedit python=3.10 -y
conda activate raedit
```

### Install dependencies

```bash
# PyTorch (install first, matching your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
pip install -r requirements.txt

# Simulation environments (required for training and evaluation)
pip install robosuite==1.5.2
pip install -e "git+https://github.com/ARISE-Initiative/robomimic.git@e10526b#egg=robomimic"

# robomimic pins old transformers/diffusers — force-reinstall our versions:
pip install transformers>=5.0.0 diffusers>=0.37.0 huggingface-hub>=1.0.0

# RLBench (optional, requires CoppeliaSim + PyRep)
pip install git+https://github.com/stepjam/PyRep.git
pip install -e git+https://github.com/stepjam/RLBench.git#egg=rlbench
pip install git+https://github.com/stepjam/gymnasium.git
```

### HuggingFace token (required for DINOv3-L)

DINOv3-L is a gated model. You must first request access at:
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m

Once approved (typically 5-10 minutes), create the secrets file:

```bash
mkdir -p configs
echo 'huggingface_token: "hf_YOUR_TOKEN_HERE"' > configs/secrets.yaml
```

Get your token from https://huggingface.co/settings/tokens.

---

## Data Preparation (Robomimic)

The pipeline converts raw robomimic demos into training-ready HDF5 with pre-computed DINOv3 tokens:

```
demo_v15.hdf5 (35 MB, raw states, no images)
  -> [Step 1: extract observations]  (needs MuJoCo renderer)
image_v15.hdf5 (~500 MB, 84x84 images, delta actions)
  -> [Step 2: convert to absolute actions]  (needs robosuite 1.5)
image_abs_v15.hdf5 (~500 MB, absolute EE poses)
  -> [Step 3: convert to unified schema]  (pure Python)
ph_abs_v15.hdf5 (~513 MB, 224x224, 7D absolute actions)
  -> [Step 4: pre-compute tokens]  (needs GPU + DINOv3)
ph_abs_v15_tokens_fp32_none.hdf5 (~25 GB, cached encoder tokens)
```

### Step 0: Download raw demos

```bash
# Using robomimic's download script (recommended):
python -m robomimic.scripts.download_datasets \
    --tasks lift can square --dataset_types ph --hdf5_types raw \
    --download_dir data/raw/robomimic

# Or download manually from HuggingFace: robomimic/robomimic_datasets
```

### Step 1: Extract observations from states

Replays each demo through robosuite to render camera images and extract proprioception. Requires a display or headless rendering (e.g., `xvfb-run`).

```bash
python -m robomimic.scripts.dataset_states_to_obs \
    --dataset data/raw/robomimic/lift/demo_v15.hdf5 \
    --output_name image_v15.hdf5 \
    --done_mode 2 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 84 --camera_width 84
```

This creates `image_v15.hdf5` in the same directory. Actions remain in delta format.

### Step 2: Convert delta actions to absolute EE poses

Our system requires absolute end-effector poses (not deltas). This replays demos through the robosuite 1.5 OSC_POSE controller and extracts world-frame goal positions/orientations.

```bash
PYTHONPATH=. python training/generate_abs_actions_v15.py \
    --input data/raw/robomimic/lift/image_v15.hdf5 \
    --output data/raw/robomimic/lift/image_abs_v15.hdf5
```

Output actions are 7D: `[world_pos(3), axis_angle(3), gripper(1)]`.

### Step 3: Convert to unified HDF5 schema

Resizes images from 84x84 to 224x224, maps cameras to slots, extracts proprioception, and computes normalization statistics.

```bash
PYTHONPATH=. python -m data_pipeline.conversion.convert_robomimic \
    data/raw/robomimic/lift/image_abs_v15.hdf5 \
    data/robomimic/lift/ph_abs_v15.hdf5 \
    --task lift
```

### Step 4: Pre-compute DINOv3 tokens (9x training speedup)

Runs the frozen DINOv3-L encoder once on all images and caches the 196x1024D tokens. Training then reads tokens directly instead of running the 303M-param encoder each step (66s vs 597s per epoch).

```bash
PYTHONPATH=. python training/precompute_tokens.py \
    --hdf5 data/robomimic/lift/ph_abs_v15.hdf5 \
    --preset fp32-none \
    --rot6d
```

The `--rot6d` flag recomputes normalization statistics in 10D format (required for training). The script creates a `*_tokens_fp32_none.hdf5` file alongside the original. Use `--preset bf16-none` for half the disk space. The training data loader auto-detects cached tokens.

## Data Preparation (RLBench)

```bash
# Convert RLBench demos to unified HDF5:
PYTHONPATH=. python data_pipeline/conversion/convert_rlbench.py \
    --task open_drawer \
    --input /path/to/rlbench/train/open_drawer \
    --val-input /path/to/rlbench/val/open_drawer \
    --output data/rlbench/open_drawer/open_drawer_dense.hdf5

# Pre-compute tokens (same as Robomimic step 4):
PYTHONPATH=. python training/precompute_tokens.py \
    --hdf5 data/rlbench/open_drawer/open_drawer_dense.hdf5 \
    --preset fp32-none --rot6d
```

---

## Training

### Phase 1 -- RAE Pretraining (Adapter + Decoder)

Pre-trains the adapter (1024->512 MLP) and ViT decoder using L1 + LPIPS + GAN reconstruction losses. This ensures the adapter retains spatial detail before policy training.

**Option A: Download pre-trained weights (recommended)**

We provide pre-trained Phase 1 weights trained on 4 Robomimic + 8 RLBench tasks for 24 epochs:

```bash
# TODO: Replace with actual HuggingFace link
wget -O checkpoints/stage1/epoch_024.pt <HUGGINGFACE_LINK>
```

**Option B: Train from scratch**

Requires unified HDF5 files for all tasks you want to train on. Uses a 3-phase schedule: reconstruction-only (epochs 0-5), discriminator warm-up (6-7), full GAN (8+). Approximately 1.5 hours per epoch on an RTX 4080.

```bash
PYTHONPATH=. python training/train_stage1_script.py \
    --hdf5 data/rlbench/close_jar/close_jar_dense.hdf5 \
            data/rlbench/open_drawer/open_drawer_dense.hdf5 \
            data/rlbench/sweep_to_dustpan/sweep_to_dustpan_dense.hdf5 \
            data/robomimic/lift/ph_abs_v15.hdf5 \
            data/robomimic/can/ph_abs_v15.hdf5 \
            data/robomimic/square/ph_abs_v15.hdf5 \
    --save_dir checkpoints/stage1 \
    --num_epochs 25 \
    --batch_size 10
```

For single-task or quick experiments, even 7 epochs of reconstruction-only training (before GAN) provides a good adapter initialization.

### Phase 2 -- Policy Training

The recommended configuration (matching our best results):

```bash
PYTHONPATH=. python training/train_v3_script.py \
    --hdf5 data/robomimic/can/ph_abs_v15_tokens_fp32_none.hdf5 \
    --eval_hdf5 data/robomimic/can/ph_abs_v15.hdf5 \
    --stage1_checkpoint checkpoints/stage1/epoch_024.pt \
    --eval_task can \
    --d_model 512 \
    --num_epochs 3000 --stop_after_epochs 100 \
    --eval_full_every_epoch 2 \
    --no_amp --no_compile \
    --save_dir checkpoints/v3_can_d512
```

Key flags and their defaults:

| Flag | Default | Description |
|------|---------|-------------|
| `--d_model` | `256` | Denoiser hidden dim (256 for easy tasks, 512 for harder ones) |
| `--spatial_pool_size` | `7` | Spatial pool: 7x7 tokens per camera (recommended) |
| `--n_cond_layers` | `4` | Self-attention conditioning layers (cross-camera reasoning) |
| `--use_flow_matching` | `True` | L1 Sample Flow with 2-step inference |
| `--no_flow_matching` | -- | Fall back to DDPM/DDIM |
| `--norm_mode` | `chi` | Chi normalization (pos minmax, rot6d/grip identity) |
| `--p_drop_attn` | `0.05` | Attention dropout |
| `--eval_mode` | `robomimic` | Chi's evaluation pipeline |
| `--lambda_recon` | `0.0` | Reconstruction co-training weight (0 = warm-start only) |
| `--stage1_checkpoint` | -- | Phase 1 checkpoint for adapter warm-start |
| `--stop_after_epochs` | `0` | Early termination (0 = disabled) |

### Resume from checkpoint

```bash
PYTHONPATH=. python training/train_v3_script.py \
    --resume checkpoints/v3_can_d512/latest_backup_0099.pt \
    [... other flags ...]
```

---

## Evaluation

```bash
PYTHONPATH=. python training/eval_v3_robomimic.py \
    --checkpoint checkpoints/v3_can_d512/best_success.pt \
    --hdf5 data/robomimic/can/ph_abs_v15.hdf5 \
    --num_episodes 50 --n_envs 25 \
    --d_model 512 --spatial_pool_size 7 --n_cond_layers 4 \
    --use_flow_matching
```

For RLBench:

```bash
PYTHONPATH=. python training/eval_v3_rlbench.py \
    --checkpoint checkpoints/v3_sweep/best_success.pt \
    --eval_hdf5 data/rlbench/sweep_to_dustpan/sweep_to_dustpan_dense.hdf5 \
    --task sweep_to_dustpan_of_size
```

---

## Benchmarks

| Benchmark | Tasks | Demos | Cameras | Resolution |
|-----------|-------|-------|---------|------------|
| **Robomimic** | Lift, Can, Square | 200/task | 2 RGB | 84x84 |
| **RLBench** | sweep_to_dustpan, open_drawer, close_jar | 100+25/task | 4 RGB | 224x224 |

---

## Key Hyperparameters

Recommended configuration (matching best results):

```python
# Architecture
d_model           = 512     # 256 for easy tasks, 512 for harder ones
n_head            = 4
n_layers          = 8
n_cond_layers     = 4       # Self-attention conditioning encoder
spatial_pool_size = 7       # 7x7 spatial tokens per camera
adapter_dim       = 512     # DINOv3 1024 -> 1024 -> 512

# Flow Matching (replaces DDPM/DDIM)
use_flow_matching = True    # L1 sample prediction, 2-step inference
                            # 50x faster than 100-step DDIM

# Normalization
norm_mode         = "chi"   # Position minmax [-1,1], rotation/gripper identity

# Optimizer (Chi et al. recipe)
lr                = 1e-4
betas             = (0.9, 0.95)
weight_decay      = 1e-3    # Denoiser weights
warmup_steps      = 1000
p_drop_attn       = 0.05

# EMA (adaptive power schedule)
ema_power         = 0.75
ema_max_decay     = 0.9999

# Action representation
ac_dim            = 10      # pos(3) + rot6d(6) + grip(1), absolute EE pose
```

---

## Checkpoint Management

Old checkpoints may contain the frozen DINOv3-L encoder in the EMA (~1.2 GB of waste per checkpoint). To slim them:

```bash
# Dry run (see how much space you'd save):
python training/slim_checkpoints.py checkpoints/v3_*/best_success.pt --dry-run

# Slim in-place:
python training/slim_checkpoints.py $(find checkpoints -name "*.pt")
```

---

## Citation

```bibtex
@misc{gigliottialidenassau2026raedp,
  title        = {{RAE-DP}: Representation Autoencoder Regularization in Diffusion Policy},
  author       = {Gigliotti Petruska, Alexander and Ali, Naqeeb and de Nassau, Jean},
  year         = {2026},
  institution  = {University of Toronto},
  note         = {CSC415 Course Project}
}
```

---

## Acknowledgements

- **Diffusion Policy** -- Chi et al., IJRR 2024
- **Diffusion Transformers with Representation Autoencoders** -- Zheng et al., 2025
- **L1 Sample Flow** -- Song et al., 2025
- **DINOv3** -- Simeoni et al., 2025
- **Robomimic** -- Mandlekar et al., CoRL 2021
- **RLBench** -- James et al., IEEE RA-L 2020

---

<div align="center">
<sub>University of Toronto -- CSC415 -- 2026</sub>
</div>
