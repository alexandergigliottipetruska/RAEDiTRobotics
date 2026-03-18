import os
import sys
import yaml
import torch
import optuna
import shutil
from huggingface_hub import HfApi
from datetime import datetime

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 1. THE PROJECT ROOT FIX
# Ensures 'models' and 'training' imports work from the 'training/' subdir
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. LOAD CONFIGS FROM YAML
def load_swarm_configs():
    # Note: Pointing to the new Stage 3 config file!
    base_path = os.path.join("configs", "swarm_stage3_config.yaml")
    secrets_path = os.path.join("configs", "secrets.yaml")
    
    with open(base_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            cfg.update(yaml.safe_load(f))
    return cfg

SWARM_CFG = load_swarm_configs()

# 3. STORAGE REDIRECTION (Safety for university disk limits)
os.environ['HF_HOME'] = SWARM_CFG["project"]["hf_cache_dir"]
os.environ['TORCH_HOME'] = SWARM_CFG["project"]["torch_cache_dir"]

# 4. IMPORT PHASE 3 SPECIFIC TRAINING LOGIC
from train_stage3_hp_distributed import Stage3Config, train_stage3

# Create a unique session ID based on when the worker started
SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M")

def upload_to_hf_and_clean(trial_dir, trial_number):
    """Sends best.pt to HF and wipes local /tmp."""
    api = HfApi()
    
    # Defaults to a stage 3 specific repo if not defined
    repo_id = SWARM_CFG["project"].get("hf_repo_id", "Denass04/RAEDiTRobotics-stage3-sweeps") 
    study_name = SWARM_CFG["project"]["study_name"]
    
    # Nested folder structure: StudyName_Timestamp / Trial_X
    path_in_repo = os.path.join(study_name + "_" + SESSION_TS, f"trial_{trial_number}")

    try:
        api.upload_folder(
            folder_path=trial_dir,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model"
        )
        shutil.rmtree(trial_dir)
        print(f"✅ Successfully uploaded Trial {trial_number} to {repo_id}/{path_in_repo}")
    except Exception as e:
        print(f"❌ HF Upload failed for Trial {trial_number}: {e}")


def sample_from_config(trial, search_space):
    """Dynamically calls trial methods and unpacks YAML kwargs."""
    sampled_hps = {}
    
    for name, config in search_space.items():
        # 1. Make a copy of the kwargs for THIS specific parameter
        kwargs = config["kwargs"].copy()
        
        # 2. Check if any kwarg is a string referencing a previous HP
        for k, v in kwargs.items():
            if isinstance(v, str) and v in sampled_hps:
                kwargs[k] = sampled_hps[v]  # Swap string for the actual integer
                
        # 3. Call Optuna using the UPDATED kwargs
        suggest_method = getattr(trial, config["method"])
        sampled_hps[name] = suggest_method(name, **kwargs)

    return sampled_hps


def objective(trial):
    # 1. Setup scratch space dynamically from config
    scratch_dir = SWARM_CFG["project"]["scratch_dir"]
    trial_dir = f"{scratch_dir}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # 2. Get the sampled values using our helper
    hps = sample_from_config(trial, SWARM_CFG["hyperparameters_search"])

    # 3. Unpack into the Stage3Config dataclass!
    config = Stage3Config(
        **SWARM_CFG["training_static"], 
        **hps, 
        save_dir=trial_dir
    )

    try:
        # 4. Run the Training (No need to init models here, train_stage3 handles it!)
        best_val_loss = train_stage3(
            config=config,
            device="cuda",
            trial=trial
        )
        
        # SUCCESS: Upload to HF and clean up.
        upload_to_hf_and_clean(trial_dir, trial.number)
        return best_val_loss

    except optuna.exceptions.TrialPruned:
        # PRUNED: Optuna killed it early. Clean local /tmp, DO NOT upload.
        print(f"Trial {trial.number} pruned. Cleaning up local scratch space.")
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
        raise  # Re-raise so the DB marks it as PRUNED

    except Exception as e:
        # CRASH: Something broke (OOM, NaN loss, etc). Clean /tmp, DO NOT upload.
        print(f"Trial {trial.number} failed: {e}")
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
        raise e 


if __name__ == "__main__":
    # This pruner only kills trials in the bottom 25th percentile
    pruner = optuna.pruners.PercentilePruner(
        percentile=SWARM_CFG["project"]["pruning_percentile"], 
        n_startup_trials=SWARM_CFG["project"]["n_startup_trials"], 
        n_warmup_steps=SWARM_CFG["project"]["n_warmup_steps"]
    )

    study = optuna.create_study(
        study_name=SWARM_CFG["project"]["study_name"],
        storage=SWARM_CFG["db_url"],
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=SWARM_CFG["project"]["n_startup_trials"], 
            n_warmup_steps=SWARM_CFG["project"]["n_warmup_steps"]
        )
    )

    study.optimize(
        objective,
        n_trials=SWARM_CFG["project"]["n_trials_per_worker"],
        catch=(Exception,)
    )