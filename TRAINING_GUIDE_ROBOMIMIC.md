# RoboMimic Training Guide

## Setup (UTM dh2026pcXX machines)


```bash
cd /virtual/csc415user/RAEDiTRobotics
source ~/venv_rlbench/bin/activate
```

See `requirements_venv_rlbench.txt` for environment setup on a new machine.

---

## Step 1: Precompute tokens (one-time per task)

```bash
# fp32 tokens, no compression (need to test how much of an effect bf16 tokens have)
python training/precompute_tokens.py \
    --hdf5 data/robomimic/square/ph_abs_v15.hdf5 \
    --preset fp32-none --rot6d

# Output: data/robomimic/square/ph_abs_v15_tokens_fp32_none.hdf5
```

Available presets: `fp32-none` (fastest), `fp32-lzf` (compressed), `bf16-none` (smaller).
Use `--rot6d` to recompute norm stats in 10D rot6d format (required for V3).

## Step 2: Train

### Recommended: L1 Flow + spatial tokens + warm-start

```bash
python -m training.train_v3_script \
    --hdf5 /virtual/csc415user/data/robomimic/lift/ph_abs_v15_tokens_fp32_none.hdf5 \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090_0312_0400/best.pt \
    --save_dir checkpoints/v3_lift_d256_0405 \
    --eval_task lift --eval_mode robomimic \
    --no_amp --no_compile --norm_mode chi \
    --use_flow_matching \
    --spatial_pool_size 7 --n_cond_layers 4 \
    --p_drop_attn 0.05 \
    --d_model 256 \
    --num_epochs 3000 --batch_size 64 --seed 42 \
    --eval_full_every_epoch 1 --eval_full_episodes 50 \
    --eval_n_envs 25 --val_ratio 0.02  --save_every_epoch 0 --no_save_best  \
    --save_rolling_every 10
```

### Standard DDPM training (Chi's recipe)

```bash
python -m training.train_v3_script \
    --hdf5 data/robomimic/square/ph_abs_v15_tokens_fp32_none.hdf5 \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090_0312_0400/best.pt \
    --save_dir checkpoints/v3_square \
    --eval_task square --eval_mode robomimic \
    --no_amp --no_compile --norm_mode chi \
    --spatial_pool_size 7 --n_cond_layers 4 \
    --p_drop_attn 0.05 \
    --num_epochs 3000 --batch_size 64 --seed 42 \
    --eval_every_epoch 10 --eval_full_every_epoch 50 \
    --eval_episodes 10 --eval_full_episodes 100 \
    --eval_n_envs 20 --val_ratio 0.02
```

## Step 3: Standalone eval

```bash
# Auto-detects architecture from checkpoint config:
python -m training.eval_v3_robomimic \
    --checkpoint checkpoints/v3_square_fm/best.pt \
    --hdf5 data/robomimic/square/ph_abs_v15_tokens_fp32_none.hdf5 \
    --num_episodes 200 --n_envs 20

# For old checkpoints without saved config, specify architecture:
python -m training.eval_v3_robomimic \
    --checkpoint checkpoints/old/best.pt \
    --hdf5 data/robomimic/square/ph_abs_v15_tokens_fp32_none.hdf5 \
    --spatial_pool_size 7 --n_cond_layers 4 --use_flow_matching \
    --num_episodes 200 --n_envs 20
```

## Resuming training

```bash
python -m training.train_v3_script \
    --hdf5 ... [same args as original run] \
    --resume checkpoints/v3_square_fm/best.pt
```

---

## Architecture Options

| Flag | Default | Description |
|------|---------|-------------|
| `--d_model` | 256 | Transformer hidden dimension |
| `--n_head` | 4 | Attention heads |
| `--n_layers` | 8 | Decoder layers |
| `--spatial_pool_size` | 7 | 7=spatial tokens (recommended), 1=avg pool (Chi), 4/14=alternatives |
| `--n_cond_layers` | 4 | 4=self-attention encoder (recommended), 0=MLP encoder (Chi) |
| `--denoiser_type` | transformer | `transformer` (Chi) or `dit` (adaLN-Zero) |
| `--use_flow_matching` | off | L1 Sample Flow (2-step inference) |
| `--grad_accum_steps` | 1 | Gradient accumulation (effective batch = batch_size * accum) |

## Training Hyperparameters (Chi's defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `--lr` | 1e-4 | AdamW |
| `--lr_adapter` | 0 | Separate adapter LR (0 = use main `--lr`) |
| `--betas` | (0.9, 0.95) | Chi's transformer recipe |
| `--weight_decay_denoiser` | 1e-3 | Transformer weight decay |
| `--weight_decay_encoder` | 1e-6 | Obs encoder weight decay |
| `--warmup_steps` | 1000 | Linear warmup then cosine decay |
| `--p_drop_attn` | 0.05 | Attention dropout (0.05 works better with frozen DINOv3 tokens) |
| `--ema_power` | 0.75 | Adaptive EMA schedule |
| `--train_diffusion_steps` | 100 | DDPM noise levels |
| `--eval_diffusion_steps` | 100 | DDIM inference steps |
| `--T_pred` | 10 | Prediction horizon |
| `--T_act` | 8 | Execution horizon (eval) |
| `--pad_before` | 1 | Allow windows starting before episode |

---

## Available tasks

`lift`, `can`, `square`, `tool_hang`
Data lives at `data/robomimic/{task}/ph_abs_v15.hdf5`.

---

## tmux Cheat Sheet

| Action | Keys / Command |
|--------|---------------|
| Detach | `Ctrl+B`, then `D` |
| Reattach | `tmux attach -t train` |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t train` |
| Scroll up | `Ctrl+B`, then `[`, arrow keys (`q` to exit) |
