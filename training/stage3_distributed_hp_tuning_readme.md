# 🐝 Robotics Stage 3: Distributed Swarm Trainer (DiT Policy)

This setup orchestrates Optuna Hyperparameter Sweeps across multiple UTM Lab PCs to train a Diffusion Transformer (DiT) policy for robotic control (Task: `lift`). 

Building on the Phase 1 infrastructure, Stage 3 introduces **offline token precomputation** to bypass the DinoV3 encoder during training, saving massive amounts of VRAM and compute. It continues to use `/tmp` for high-speed local caching and securely pushes all successful checkpoints to Hugging Face.

---

## 🏗️ Architecture
* **Primary Node (Head PC):** Hosts the PostgreSQL database (`142.1.46.5:5433`) and the Optuna Dashboard.
* **Worker Nodes (Lab PCs):** Pulls the `stage3_hp-tuning` branch automatically, pulls a trial, and trains the DiT policy.
* **Precomputed Tokens:** Raw images are encoded via DinoV3 *before* the sweep starts. The DiT trains directly on these lightweight patches.
* **DDP-Safe Pruning:** If Rank 0 detects a trial is failing or overfitting, it broadcasts a kill signal to all GPUs in the DDP group to gracefully abort, saving compute time.
* **Extended Dashboard:** The dashboard tracks the validation loss as the primary objective, but also logs `lr`, `train_loss`, and other metrics at the *best* epoch for deeper analysis.

---

## 🛠️ Step 0: Precompute the Tokens (CRITICAL)
Before starting the swarm, you **must** precompute the visual tokens. The Stage 3 dataset loader expects `(B, T, num_patches, d)` embeddings, not raw `(B, T, 3, 224, 224)` images.

Run this command locally or on a capable lab PC:
```bash
python3 training/precompute_tokens.py \
    --hdf5 data/complete_unified_data/lift.hdf5 \
    --output data/stage3_tokens/lift_tokens.hdf5 \
    --batch_size 64 \
    --device cuda
```

---

## ⚡ Step 0.5: Decompress for Speed (Optimization)
Standard HDF5 files are compressed with GZIP, which creates a CPU bottleneck. Run this to create a "Fast" version that allows for higher `it/s` and better GPU utilization:

```bash
python3 -c "
import h5py
src = 'data/stage3_tokens/lift_tokens.hdf5'
dst = 'data/stage3_decompressed_tokens/lift_tokens_fast.hdf5'
with h5py.File(src, 'r') as s, h5py.File(dst, 'w') as d:
    for attr in s.attrs: d.attrs[attr] = s.attrs[attr]
    if 'mask' in s: s.copy('mask', d)
    if 'norm_stats' in s: s.copy('norm_stats', d)
    for key in s['data']:
        g = d.create_group(f'data/{key}')
        for ds in s[f'data/{key}']:
            g.create_dataset(ds, data=s[f'data/{key}/{ds}'][:])
print('Optimization Complete.')
"
```
**Important:** Update your `swarm_stage3_config.yaml` to point to `lift_tokens_fast.hdf5`.

---

## ⚙️ Step 1: Configuration

**1. Database & Secrets (`configs/secrets.yaml`)**
Ensure your secrets file exists (DO NOT commit this to Git). It shares the same database as Phase 1:
```yaml
ssh_password: "YOUR_LAB_PASSWORD"
db_url: "postgresql://csc415user@142.1.46.5:5433/optuna_db"
hf_token: "your_huggingface_write_token"
```

**2. Swarm Grid (`configs/swarm_stage3_config.yaml`)**
Edit this file to select which lab PCs to use. **Do not** hardcode nodes in the manager script. Uncomment the machines you want to recruit:
```yaml
nodes:
  - dh2020pc00
  - dh2020pc01
  # - dh2020pc02
```
*Note: The manager automatically SSHs in, checks out the `stage3_hp-tuning` branch, and runs `git pull` before launching the workers.*

---

## 🚀 Step 2: How to Run the Swarm

**1. Start the Database (Head PC Only)**
```bash
pg_ctl -D ~/optuna_pg_data -l ~/optuna_pg_data/pg.log start
```

**2. Manage the Swarm (From any PC)**
Use `swarm_manager_stage3.py` to orchestrate the lab machines. 

* **Start the workers:**
```bash
python3 training/swarm_manager_stage3.py start
```
* **Check who is training:**
```bash
python3 training/swarm_manager_stage3.py status
```
* **Restart idle machines:** (Finds nodes that crashed or finished their queue and restarts them)
```bash
python3 swarm_manager_stage3.py restart_idle
```
* **Graceful Stop:**
```bash
python3 training/swarm_manager_stage3.py stop
```

---

## 📊 Step 3: Monitoring & Evaluation

**1. The Optuna Dashboard**
Start the UI on the Head PC:
```bash
optuna-dashboard "postgresql://csc415user@142.1.46.5:5433/optuna_db"
```
Create an SSH tunnel on your personal laptop to view it:
```bash
ssh -N -L 8080:localhost:8080 csc415user@dh2026pc02.utm.utoronto.ca
```
Navigate to `http://localhost:8080`. You will see `robotics_stage3_lift_test`.

**2. Evaluation Videos (`/tmp/eval_videos`)**
Every 50 epochs, the policy runs inside the Robomimic simulator and records an MP4 rollout and an action trajectory graph. 
* Because videos eat disk space, they are saved locally to `/tmp/eval_videos` on the specific lab PC running the trial.
* To watch them, you must SSH into that specific PC and pull them *before* the trial finishes (the worker script wipes `/tmp` upon completion to protect the drive).

---

## 🧹 Maintenance & Clean Slate Protocol
If you need to wipe the study and start over:

**1. Delete the Optuna Study:**
```bash
python3 -c "import optuna, yaml; db_url = yaml.safe_load(open('configs/secrets.yaml'))['db_url']; optuna.delete_study(study_name='robotics_stage3_lift_dropout_change', storage=db_url)"
```

**2. Clear Zombie Processes & Temp Files:**
```bash
for node in dh2026pc16 dh2026pc17 dh2026pc18 dh2026pc19; do
    ssh -o StrictHostKeyChecking=no $node.utm.utoronto.ca "pkill -9 -u \$USER python && rm -rf /tmp/denassau_stage3/* && rm -rf /tmp/eval_videos/*"
done
```

**3. Check Specific Worker Logs:**
```bash
for node in dh2026pc06 dh2026pc08 dh2026pc09 dh2026pc10; do
    echo "=== LOGS FOR $node ==="
    ssh -o ConnectTimeout=5 $node.utm.utoronto.ca "tail -n 50 /tmp/swarm_logs/worker_${node}_stage3.log"
    echo ""
done
```