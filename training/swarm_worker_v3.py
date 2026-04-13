"""Optuna worker for distributed HP tuning of Stage 3 policy.

Each worker connects to a shared Optuna DB (PostgreSQL), samples HPs via TPE,
runs train_v3() with those HPs, and reports eval success rate as the objective.

Pruning: train_v3() reports success rate to the trial after each full eval epoch.
Optuna's MedianPruner kills underperforming trials early.

Usage (launched by swarm_manager_stage3.py, or manually):
    cd ~/RAEDiTRobotics
    python training/swarm_worker_v3.py
"""

import gc
import glob
import json
import logging
import os
import shutil
import sys
from datetime import datetime

import optuna
import torch

os.environ["PYOPENGL_PLATFORM"] = "egl"

# Ensure project root is on sys.path (worker may be launched from training/ dir)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_swarm_configs():
    """Load swarm config + secrets (secrets.yaml is git-ignored)."""
    import yaml
    base_path = os.path.join("configs", "swarm_stage3_config.yaml")
    secrets_path = os.path.join("configs", "secrets.yaml")

    with open(base_path, "r") as f:
        cfg = yaml.safe_load(f)
    if os.path.exists(secrets_path):
        with open(secrets_path, "r") as f:
            secrets = yaml.safe_load(f)
            if secrets:
                cfg.update(secrets)
    return cfg


SWARM_CFG = load_swarm_configs()

# Redirect caches to /tmp to avoid NFS quota issues
os.environ["HF_HOME"] = SWARM_CFG["project"].get("hf_cache_dir", "/tmp/hf_cache")
os.environ["TORCH_HOME"] = SWARM_CFG["project"].get("torch_cache_dir", "/tmp/torch_cache")

from training.train_v3 import V3Config, train_v3

SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M")

# d_model -> n_head mapping (must divide evenly)
D_MODEL_TO_N_HEAD = {256: 4, 384: 6, 512: 8}


# ---------------------------------------------------------------------------
# HuggingFace upload + cleanup
# ---------------------------------------------------------------------------

def upload_to_hf_and_clean(trial_dir, trial_number):
    """Upload trial artifacts to HuggingFace, then wipe local directory."""
    from huggingface_hub import HfApi

    upload_token = SWARM_CFG.get("hf_upload_token", SWARM_CFG.get("hf_token", SWARM_CFG.get("huggingface_token")))
    if not upload_token:
        log.warning("No HF token found — skipping upload, cleaning up locally")
        shutil.rmtree(trial_dir, ignore_errors=True)
        return

    api = HfApi(token=upload_token)
    repo_id = SWARM_CFG["project"].get("hf_repo_id", "Denass04/RAEDiTRobotics-stage3-sweeps")
    study_name = SWARM_CFG["project"]["study_name"]
    path_in_repo = os.path.join(f"{study_name}_{SESSION_TS}", f"trial_{trial_number}")

    # Exclude media files (eval rollout videos) — they're ~1000 files per trial and
    # blow through HF's 100k-files-per-repo soft limit. We never use them downstream;
    # all numeric data is in metrics.jsonl and the Optuna DB. Keep checkpoints (.pt),
    # logs, and metrics.
    try:
        api.upload_folder(
            folder_path=trial_dir,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            ignore_patterns=["media/**", "*.mp4", "*.gif"],
        )
        log.info("Uploaded Trial %d to %s/%s", trial_number, repo_id, path_in_repo)
    except Exception as e:
        log.warning("HF upload failed for Trial %d: %r", trial_number, e)
    finally:
        shutil.rmtree(trial_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# HP sampling
# ---------------------------------------------------------------------------

def sample_from_config(trial, search_space):
    """Sample hyperparameters using Optuna trial methods defined in YAML."""
    sampled = {}
    for name, cfg in search_space.items():
        kwargs = cfg["kwargs"].copy()
        # Resolve string references to previously sampled HPs
        for k, v in kwargs.items():
            if isinstance(v, str) and v in sampled:
                kwargs[k] = sampled[v]
        suggest_fn = getattr(trial, cfg["method"])
        sampled[name] = suggest_fn(name, **kwargs)
    return sampled


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def extract_eval_success_rates(save_dir):
    """Parse metrics JSONL to get all eval success rates in order."""
    candidates = glob.glob(os.path.join(save_dir, "metrics*.jsonl"))
    if not candidates:
        return []

    rates = []
    for path in candidates:
        with open(path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "eval_success_rate" in data:
                        rates.append(data["eval_success_rate"])
                except json.JSONDecodeError:
                    continue
    return rates


def extract_weighted_success_rate(save_dir, peak_weight=0.3, tail_n=6):
    """Compute weighted objective: peak_weight * peak + (1 - peak_weight) * last_N_avg.

    The tail window (last tail_n evals) skips the warmup phase where eval SR
    is dominated by noise, so the metric rewards final-plateau quality rather
    than convergence speed. For pruned trials with fewer than tail_n evals,
    the whole trajectory is used.
    """
    rates = extract_eval_success_rates(save_dir)
    if not rates:
        return 0.0

    peak = max(rates)
    tail = rates[-tail_n:]  # last N evals; falls back to all if len(rates) < tail_n
    tail_avg = sum(tail) / len(tail)
    return peak_weight * peak + (1 - peak_weight) * tail_avg


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial):
    scratch_dir = SWARM_CFG["project"].get("scratch_dir", "/tmp/optuna_trials")
    trial_dir = os.path.join(scratch_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # 1. Sample hyperparameters from search space
    hps = sample_from_config(trial, SWARM_CFG["hyperparameters_search"])

    # 2. Build V3Config with static overrides (expand ~ in paths)
    config = V3Config()
    if "training_static" in SWARM_CFG:
        for key, value in SWARM_CFG["training_static"].items():
            if isinstance(value, str):
                value = os.path.expanduser(value)
            elif isinstance(value, list):
                value = [os.path.expanduser(v) if isinstance(v, str) else v for v in value]
            setattr(config, key, value)

    # 3. Apply dynamic Optuna overrides
    for key, value in hps.items():
        setattr(config, key, value)

    # 4. Handle d_model -> n_head constraint
    if "d_model" in hps:
        config.n_head = D_MODEL_TO_N_HEAD.get(hps["d_model"], 4)

    # 5. Force save directory to isolated trial folder
    config.save_dir = trial_dir

    log.info("Trial %d starting with: %s", trial.number, hps)

    try:
        train_v3(config=config, device="cuda", trial=trial)

        # Compute weighted objective from metrics (0.3 * peak + 0.7 * last_6_avg)
        objective_value = extract_weighted_success_rate(trial_dir)
        rates = extract_eval_success_rates(trial_dir)
        peak = max(rates) if rates else 0.0
        tail = rates[-6:] if rates else []
        tail_avg = sum(tail) / len(tail) if tail else 0.0

        log.info("Trial %d finished — peak: %.1f%%, last6_avg: %.1f%%, weighted: %.4f",
                 trial.number, peak * 100, tail_avg * 100, objective_value)

        upload_to_hf_and_clean(trial_dir, trial.number)
        return objective_value

    except optuna.exceptions.TrialPruned:
        objective_value = extract_weighted_success_rate(trial_dir)
        log.info("Trial %d pruned — weighted objective: %.4f",
                 trial.number, objective_value)
        upload_to_hf_and_clean(trial_dir, trial.number)
        raise  # re-raise so Optuna marks it as PRUNED

    except Exception as e:
        log.error("Trial %d failed: %s", trial.number, e)
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)
        raise

    finally:
        # Free GPU memory between trials
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pruner = optuna.pruners.PercentilePruner(
        percentile=SWARM_CFG["project"].get("pruning_percentile", 25.0),
        n_startup_trials=SWARM_CFG["project"].get("n_startup_trials", 10),
        n_warmup_steps=SWARM_CFG["project"].get("n_warmup_steps", 19),
    )

    study = optuna.create_study(
        study_name=SWARM_CFG["project"]["study_name"],
        storage=SWARM_CFG["db_url"],
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
    )

    study.optimize(
        objective,
        n_trials=SWARM_CFG["project"].get("n_trials_per_worker", 5),
        catch=(Exception,),
    )

    log.info("Worker finished. Best trial: %s (value=%.3f)",
             study.best_trial.number, study.best_value)
