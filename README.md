<div align="center">

# RAEDiT: Robust Diffusion Transformer Policy with Representation Autoencoder Visual Encoding

**Robust Multi-View Diffusion Policy with Frozen ViT and RAE-Decoder Regularization**

*Alexander Gigliotti Petruska · Naqeeb Ali · Jean de Nassau*
*University of Toronto — CSC415 Course Project, 2026*

---

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Branch](https://img.shields.io/badge/branch-ViT__experiments-brightgreen)](https://github.com/)

</div>

---

## Overview

RAEDiT investigates whether **Representation Autoencoder (RAE)** visual encoding improves robotic manipulation policies built on **Diffusion Transformer (DiT) Block** architectures. Standard approaches encode camera observations with trained CNNs or VAEs, each with known limitations: CNNs discard the rich prior of large-scale vision models, and VAEs impose a Gaussian bottleneck that blurs fine-grained spatial detail. We replace both with a **frozen DINOv3-L encoder** regularised by a lightweight trainable adapter and an RAE-style decoder reconstruction objective, then feed the resulting tokens into a faithful reproduction of Chi et al.'s cross-attention diffusion policy transformer.

The central question: *does an RAE-regularised visual representation yield measurably better task success, sample efficiency, or generalisation compared to standard trained-encoder baselines on established manipulation benchmarks?*

---

## Key Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Frozen foundation encoder** | DINOv3-L replaces trained CNN/ResNet encoders; weights never updated during policy training |
| 2 | **RAE-regularised adapter** | A 2-layer MLP adapter projects DINOv3 patch tokens (1024→512) and is jointly trained with a ViT decoder reconstruction loss to prevent representation collapse |
| 3 | **Faithful DiT baseline** | Chi et al. (2023) cross-attention transformer denoiser reproduced with exact hyperparameters (betas, weight decay, AdaLN-Zero) for clean ablation comparisons |
| 4 | **Unified multi-benchmark data pipeline** | Single HDF5 schema covering Robomimic, RLBench, and ManiSkill with precomputed token caching for 50% training speedup |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 1 — Visual Encoder Pretraining (RAE)                             │
│                                                                         │
│   I^(v)_t ──► [Frozen DINOv3-L] ──► z^(v)_t ∈ R^{196×1024}              │
│                                          │                              │
│                              [Trainable Adapter A_φ]                    │
│                                          │                              │
│                              z̄^(v)_t ∈ R^{196×512}                      │
│                                    │         │                          │
│                             [ViT Decoder] [GAN Disc.]                   │
│                                    │                                    │
│              L_recon = Σ_v ‖D_ψ(z̄^(v)_t) − I^(v)_t‖²                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
              Adapter checkpoint  ──┘  (frozen at Stage 2)
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│  Stage 2 — Policy Training (DiT Block Policy)                           │
│                                                                         │
│   Observation buffer [T_obs frames × K cameras]                         │
│          │                                                              │
│   [Stage1Bridge]  ──── online encode OR load cached tokens              │
│          │                                                              │
│   [ObservationEncoder]  ──  pool 196 tokens → 512D per view             │
│          │                  concat proprioception → conditioning seq.   │
│          ▼                                                              │
│   [Cross-Attention TransformerDenoiser]  (Chi et al., 2023)             │
│          │  ┌─ noisy actions x_t                                        │
│          │  ├─ time embedding (sinusoidal)                              │
│          │  └─ memory = [t_emb, obs_0, obs_1, ...]                      │
│          │     causal self-attn → cross-attn → FFN                      │
│          ▼                                                              │
│   ε̂  ──► DDPM loss  ──► DDIM inference at evaluation                    │
│                                                                         │
│   L_total = L_DDPM  +  λ · L_recon   (λ annealed during policy train)   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Module Map

```
RAEDiTRoboticsMain/
├── models/
│   ├── encoder.py              # FrozenMultiViewEncoder (DINOv3-L wrapper)
│   ├── adapter.py              # TrainableAdapter  (1024 → 512, 2-layer MLP)
│   ├── decoder.py              # ViTDecoder        (RAE reconstruction head)
│   ├── discriminator.py        # GAN discriminator (Stage 1 auxiliary)
│   ├── stage1_bridge.py        # Stage1Bridge      (load & freeze Stage 1)
│   ├── obs_encoder_v3.py       # ObservationEncoder (pool + concat proprio)
│   ├── denoiser_transformer.py # TransformerDenoiser (Chi cross-attention)
│   ├── denoiser_dit.py         # DiTDenoiser       (adaLN-Zero ablation)
│   ├── policy_v3.py            # PolicyDiTv3       (end-to-end policy)
│   ├── ema_model.py            # EMAModel          (adaptive power schedule)
│   └── losses.py               # Custom loss utilities
├── training/
│   ├── train_v3.py             # Main training loop  (V3Config dataclass)
│   ├── train_v3_script.py      # CLI entry point
│   ├── eval_v3.py              # Receding-horizon rollout evaluator
│   ├── eval_v3_async.py        # Parallel async evaluation
│   ├── eval_v3_robomimic.py    # Robomimic-specific evaluation
│   └── precompute_tokens.py    # Cache DINOv3 tokens to HDF5
├── data_pipeline/
│   ├── conversion/             # Raw → unified HDF5 converters
│   ├── datasets/               # Stage1Dataset, Stage3Dataset
│   ├── envs/                   # Robomimic / RLBench / ManiSkill wrappers
│   ├── evaluation/             # Rollout, metrics, visualisation
│   └── gym_util/               # AsyncVectorEnv utilities
└── requirements.txt
```

---

## Baselines

We design four baselines to isolate each contribution:

| ID | Name | Encoder | Decoder Reg. | Denoiser |
|----|------|---------|-------------|---------|
| **B1** | Diffusion Policy (Chi et al., 2023) | ResNet-18 (trained) | — | U-Net |
| **B2** | DiT-Block Policy (Dasari et al., 2024) | ResNet-26 (trained) | — | Cross-attention DiT |
| **B3** | RAEDiT — no decoder | DINOv3-L (frozen) | None | Cross-attention DiT |
| **Ours** | **RAEDiT** | DINOv3-L (frozen) | RAE recon. | Cross-attention DiT |

---

## Benchmarks

| Benchmark | Tasks | # Demos | Cameras | Eval Episodes |
|-----------|-------|---------|---------|---------------|
| **Robomimic** (Mandlekar et al., 2021) | Lift, Can, Square | 200 / task | 2 RGB | 50 |
| **RLBench** (James et al., 2020) | reach\_target, push\_button, pick\_and\_lift, slide\_block\_to\_target, put\_item\_in\_drawer | 100 / task | 4 RGB | 25 |
| **ManiSkill2** | PickCube, StackCube | 1000 / task | 2 RGB | 50 |

---

## Installation

> **Requirements:** Python ≥ 3.10, CUDA ≥ 11.8, ~16 GB VRAM recommended for full pipeline.

### 1. Clone & create environment

```bash
git clone https://github.com/<your-org>/RAEDiTRoboticsMain.git
cd RAEDiTRoboticsMain
conda create -n raedit python=3.10 -y
conda activate raedit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Robomimic** (required for Robomimic benchmarks):
```bash
pip install git+https://github.com/ARISE-Initiative/robomimic.git
```

**RLBench** (required for RLBench benchmarks):
```bash
# Follow the official RLBench install guide (requires CoppeliaSim/PyRep)
# https://github.com/stepjam/RLBench
```

### 3. Configure data paths

```bash
cp data_pipeline/configs/paths_template.yaml data_pipeline/configs/paths.yaml
# Edit paths.yaml to point to your raw demo directories and unified HDF5 output location
```

---

## Data Preparation

### Convert raw demonstrations to unified HDF5

```bash
# Robomimic
python data_pipeline/conversion/convert_robomimic.py \
    --input /path/to/robomimic/lift.hdf5 \
    --output /data/unified/robomimic_lift.hdf5

# RLBench
python data_pipeline/conversion/convert_rlbench.py \
    --task reach_target \
    --input /path/to/rlbench/reach_target \
    --output /data/unified/rlbench_reach_target.hdf5
```

### Compute normalisation statistics

```bash
python data_pipeline/conversion/compute_norm_stats.py \
    --dataset /data/unified/robomimic_lift.hdf5
```

### (Optional) Precompute DINOv3 tokens — ~50% training speedup

```bash
python training/precompute_tokens.py \
    --dataset /data/unified/robomimic_lift.hdf5 \
    --stage1_ckpt /checkpoints/stage1/best.pt \
    --output /data/cached_tokens/robomimic_lift_tokens.hdf5
```

---

## Training

### Stage 1 — RAE Pretraining (Adapter + Decoder)

Train the adapter and ViT decoder jointly on the reconstruction objective:

```bash
python training/train_stage1.py \
    --dataset /data/unified/robomimic_lift.hdf5 \
    --output_dir /checkpoints/stage1 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

The checkpoint saved under `output_dir/best.pt` contains keys `adapter`, `decoder`, and `discriminator`.

### Stage 2 — Policy Training (DiT Block Policy)

```bash
python training/train_v3_script.py \
    --dataset     /data/unified/robomimic_lift.hdf5 \
    --stage1_ckpt /checkpoints/stage1/best.pt \
    --output_dir  /checkpoints/policy/lift \
    --task        lift \
    --denoiser    cross_attn \
    --epochs      100 \
    --batch_size  64 \
    --n_active_cams 2
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--denoiser` | `cross_attn` | `cross_attn` (Chi) or `dit` (adaLN-Zero ablation) |
| `--n_active_cams` | `2` | Active cameras (2 for Robomimic, 4 for RLBench) |
| `--use_cached_tokens` | `False` | Load precomputed tokens instead of running encoder online |
| `--T_obs` | `2` | Observation history horizon |
| `--T_pred` | `10` | Action chunk prediction horizon |
| `--T_act` | `8` | Steps executed per planning cycle |
| `--lambda_recon` | `0.1` | Weight for reconstruction auxiliary loss during policy training |

### Resume from checkpoint

```bash
python training/train_v3_script.py \
    --resume_from /checkpoints/policy/lift/epoch_050.pt \
    [... other flags ...]
```

---

## Evaluation

### Single-checkpoint rollout

```bash
python training/eval_v3_robomimic.py \
    --policy_ckpt /checkpoints/policy/lift/best_ema.pt \
    --task lift \
    --n_episodes 50 \
    --n_envs 10
```

### Async parallel evaluation across all tasks

```bash
python training/eval_v3_async.py \
    --checkpoint_dir /checkpoints/policy \
    --tasks lift can square \
    --n_episodes 50
```

---

## Key Hyperparameters

The full configuration is managed by the `V3Config` dataclass in [training/train_v3.py](training/train_v3.py). The most important entries:

```python
# Architecture
d_model        = 256    # Transformer hidden dimension
n_head         = 4      # Attention heads
n_layers       = 8      # Transformer decoder layers
adapter_dim    = 512    # Adapter output dimension (DINOv3 1024 → 512)

# Diffusion
train_diffusion_steps = 100   # DDPM training timesteps
eval_diffusion_steps  = 100   # DDIM inference steps

# Optimiser (Chi et al. recipe)
lr                    = 1e-4
betas                 = (0.9, 0.95)   # NOT diffusers default (0.9, 0.999)
weight_decay_denoiser = 1e-3
weight_decay_encoder  = 1e-6
warmup_steps          = 1000

# EMA (adaptive power schedule)
ema_power     = 0.75           # Reaches 0.9999 at ~1M steps
ema_max_decay = 0.9999

# Action representation
ac_dim        = 10      # 10D rot6d (converted from 7D delta-EE in dataset)
proprio_dim   = 9       # Proprioceptive state dimension
```

---

## Results

> Experiments are ongoing. Results will be populated as runs complete.

| Method | Lift | Can | Square | RLBench (avg) |
|--------|------|-----|--------|---------------|
| B1 — Diffusion Policy (Chi et al.) | — | — | — | — |
| B2 — DiT-Block Policy (Dasari et al.) | — | — | — | — |
| B3 — RAEDiT (no decoder reg.) | — | — | — | — |
| **RAEDiT (ours)** | — | — | — | — |

*Table entries will be filled with mean ± std success rate (%) over 3 seeds × 50 episodes.*

---

## Project Structure Details

### Data Schema

Each unified HDF5 file follows:

```
/data/<demo_key>/
    images        uint8   [T, K=4, H=224, W=224, 3]
    view_present  bool    [K]
    actions       float32 [T, 7]     # 7D delta end-effector
    proprio       float32 [T, D_prop]
    states        float32 [T, D_s]   # optional, for GT replay
/norm_stats/
    actions/{mean, std, min, max}
    proprio/{mean, std, min, max}
```

### Rotation Convention

Demonstrations are stored in 7D delta end-effector format `[dx, dy, dz, rx, ry, rz, gripper]` (axis-angle rotation). The dataset class converts to 10D `rot6d` for training; the evaluator converts back to axis-angle before sending actions to the environment.

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{gigliottialidenassau2026raedit,
  title        = {Robust Multi-View Diffusion Policy with Frozen {ViT} and {RAE}-Decoder Regularization},
  author       = {Gigliotti Petruska, Alexander and Ali, Naqeeb and de Nassau, Jean},
  year         = {2026},
  institution  = {University of Toronto},
  note         = {CSC415 Course Project}
}
```

---

## Acknowledgements

This project builds on and is deeply indebted to the following works:

- **Diffusion Policy** — Chi et al., RSS 2023 — cross-attention transformer denoiser and DDPM/DDIM recipe
- **Ingredients for Robotic Diffusion Transformers** — Dasari et al., 2024 — DiT-Block Policy architecture and adaLN-Zero conditioning
- **Diffusion Transformers with Representation Autoencoders** — Zheng et al., 2025 — RAE concept and motivation
- **Scaling Text-to-Image DiTs with RAEs** — Tong et al., 2026 — ViT decoder design
- **DINOv3** — Simeoni et al., 2025 — frozen foundation visual encoder
- **Robomimic** — Mandlekar et al., CoRL 2021 — benchmark environments and evaluation protocol
- **RLBench** — James et al., IEEE RA-L 2020 — manipulation task suite

---

<div align="center">
<sub>University of Toronto · CSC415 · 2026</sub>
</div>
