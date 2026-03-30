# Stage 1 Training Guide

## Quick Start (with tmux)

```bash
# 1. Start a named tmux session
tmux new -s train

# 2. Activate environment
source ~/venv_rlbench/bin/activate
cd ~/RAEDiTRobotics

# 3. Launch training
python3 training/train_stage1_script.py \
    --hdf5 data/rlbench/close_jar.hdf5 \
        data/rlbench/meat_off_grill.hdf5 \
        data/rlbench/open_drawer.hdf5 \
        data/rlbench/place_wine_at_rack_location.hdf5 \
        data/rlbench/push_buttons.hdf5 \
        data/rlbench/slide_block_to_color_target.hdf5 \
        data/rlbench/sweep_to_dustpan_of_size.hdf5 \
        data/rlbench/turn_tap.hdf5 \
        data/robomimic/lift/ph.hdf5 \
        data/robomimic/can/ph.hdf5 \
        data/robomimic/square/ph.hdf5 \
        data/robomimic/tool_hang/ph.hdf5 \
    --batch_size 6 \
    --epoch_start_disc 6 --epoch_start_gan 8 \
    --save_dir checkpoints/stage1 \
    --save_every 1

# 4. Detach from tmux (training keeps running):
#    Press Ctrl+B, then D

# 5. Reattach later (after reconnecting via SSH):
tmux attach -t train

# 6. Watch GPU
watch -n 1 nvidia-smi
```

## Resuming from a checkpoint

```bash
python3 training/train_stage1_script.py \
    --hdf5 data/rlbench/close_jar.hdf5 \
        data/rlbench/meat_off_grill.hdf5 \
        data/rlbench/open_drawer.hdf5 \
        data/rlbench/place_wine_at_rack_location.hdf5 \
        data/rlbench/push_buttons.hdf5 \
        data/rlbench/slide_block_to_color_target.hdf5 \
        data/rlbench/sweep_to_dustpan_of_size.hdf5 \
        data/rlbench/turn_tap.hdf5 \
        data/robomimic/lift/ph.hdf5 \
        data/robomimic/can/ph.hdf5 \
        data/robomimic/square/ph.hdf5 \
        data/robomimic/tool_hang/ph.hdf5 \
    --batch_size 4 \
    --epoch_start_disc 6 --epoch_start_gan 8 \
    --save_dir checkpoints/stage1 \
    --save_every 1 \
    --resume checkpoints/stage1/epoch_007.pt
```

---

# Stage 3 Training Guide

Stage 3 trains a diffusion policy (per task) on top of the frozen Stage 1 encoder.
Training uses precomputed encoder tokens to skip the expensive DINOv3 forward pass.

## Step 1: Precompute tokens (one-time per task)

```bash
# Precompute frozen encoder tokens (~5 min per task on GPU)
python training/precompute_tokens.py \
    --hdf5 data/robomimic/lift/ph.hdf5 \
    --batch_size 64 --device cuda
# Output: data/robomimic/lift/ph_tokens.hdf5

# Decompress for faster training (gzip hurts DataLoader throughput)
python -c "
import h5py
src = 'data/robomimic/lift/ph_tokens.hdf5'
dst = 'data/robomimic/lift/ph_tokens_fast.hdf5'
with h5py.File(src, 'r') as s, h5py.File(dst, 'w') as d:
    for attr in s.attrs: d.attrs[attr] = s.attrs[attr]
    if 'mask' in s: s.copy('mask', d)
    if 'norm_stats' in s: s.copy('norm_stats', d)
    for key in s['data']:
        g = d.create_group(f'data/{key}')
        for ds in s[f'data/{key}']:
            g.create_dataset(ds, data=s[f'data/{key}/{ds}'][:])
print('Done')
"
```

## Step 2: Train (per task)

```bash
tmux new -s stage3

python training/train_stage3_script.py \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --hdf5 data/robomimic/lift/ph_tokens_fast.hdf5 \
    --num_epochs 2000 \
    --batch_size 44 \
    --num_workers 8 \
    --save_every_epoch 50 \
    --save_dir checkpoints/stage3_lift_2k \
    --norm_mode minmax \
    --warmup_steps 1000 \
    --p_view_drop 0.0 \
    --eval_video_task lift \
    --eval_video_hdf5 data/robomimic/lift/ph.hdf5 \
    --eval_video_episodes 1 \
    --eval_video_steps 100 \
    --eval_video_dir eval_videos
```

## Resuming

```bash
python training/train_stage3_script.py \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --hdf5 data/robomimic/lift/ph_tokens_fast.hdf5 \
    --num_epochs 2000 \
    --batch_size 44 \
    --num_workers 8 \
    --save_every_epoch 50 \
    --save_dir checkpoints/stage3_lift_2k \
    --norm_mode minmax \
    --warmup_steps 1000 \
    --p_view_drop 0.0 \
    --eval_video_task lift \
    --eval_video_hdf5 data/robomimic/lift/ph.hdf5 \
    --eval_video_episodes 1 \
    --eval_video_steps 100 \
    --eval_video_dir eval_videos \
    --resume checkpoints/stage3_lift_2k/epoch_0999.pt
```

## Manual eval with video

```bash
python training/eval_stage3_video.py \
    --checkpoint checkpoints/stage3_lift_2k/epoch_0999.pt \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --hdf5 data/robomimic/lift/ph.hdf5 \
    --task lift \
    --num_episodes 5 \
    --eval_steps 100 \
    --output_dir eval_videos/manual_epoch0999 \
    --norm_mode minmax --device cuda
```

## Full eval (success rate)

```bash
python training/eval_stage3.py \
    --checkpoint checkpoints/stage3_lift_2k/best.pt \
    --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
    --hdf5 data/robomimic/lift/ph.hdf5 \
    --task lift \
    --num_episodes 25 \
    --norm_mode minmax
```

## Expected Performance

- ~44 sec/epoch on RTX 4080 (16GB), batch_size=44 (with torch.compile + decompressed tokens)
- 2000 epochs ≈ 24 hours
- Checkpoints + eval video saved every 50 epochs
- Eval videos appear in `eval_videos/epoch_XXXX/`
- Use `--eval_steps 100` for early checkpoints, reduce to 10 once well-trained

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--num_epochs` | 2000 | Chi et al. uses 3000 for image-based tasks |
| `--batch_size` | 44 | Fill GPU memory (RTX 4080 16GB) |
| `--lr` | 1e-4 | Noise net + token assembly |
| `--lr_adapter` | 1e-5 | 10x lower to prevent adapter drift |
| `--warmup_steps` | 1000 | Linear warmup then cosine decay |
| `--ema_decay` | 0.9999 | Smoothed weights for eval |
| `--p_view_drop` | 0.0 | 0 for single-task, 0.15 for multi-task |
| `--train_diffusion_steps` | 100 | DDPM noise levels |
| `--eval_diffusion_steps` | 10 | DDIM inference steps |
| `--eval_video_steps` | 100 | More steps = cleaner actions for early checkpoints |

---

## tmux Cheat Sheet

| Action              | Keys / Command          |
|---------------------|-------------------------|
| Detach              | `Ctrl+B`, then `D`     |
| Reattach            | `tmux attach -t stage3` |
| List sessions       | `tmux ls`               |
| Kill session        | `tmux kill-session -t stage3` |
| Scroll up           | `Ctrl+B`, then `[`, then arrow keys (press `q` to exit) |

## Training Schedule (Stage 1, from Zheng et al. 2025, Table 12)

- Epochs 0-5: Pure reconstruction (L1 + LPIPS)
- Epochs 6-7: Discriminator warmup (disc trains, no adversarial loss to generator)
- Epochs 8+: Full GAN with adaptive lambda

## Expected Performance (Stage 1)

- ~6 it/s on RTX 4080 (16GB), batch_size=6
- ~90 min/first 5-7 epochs, ~320 min/epoch 8 onwards. 50 epochs total (~11 days!)
- Checkpoints: ~300-400MB each (saved every epoch)

## New Machine Setup

```bash
# Clone repo
git clone https://github.com/alexandergigliottipetruska/RAEDiTRobotics.git
cd RAEDiTRobotics

# Python environment
virtualenv --python=python3.10 ~/venv_rlbench
source ~/venv_rlbench/bin/activate
pip install -r requirements_fixed.txt

# HuggingFace auth (for DINOv3-L)
mkdir -p configs
echo 'huggingface_token: "YOUR_TOKEN"' > configs/secrets.yaml

# Pre-download models
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov3-vitl16-pretrain-lvd1689m', trust_remote_code=True)"
python -c "import torch; torch.hub.load('facebookresearch/dino', 'dino_vits8')"

# Copy data from existing machine
scp -r user@other_pc:~/RAEDiTRobotics/data/ ./data/
```
