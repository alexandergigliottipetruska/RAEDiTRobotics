import os
import sys
import yaml
import json
import glob
import shutil
from datetime import datetime
import optuna
from huggingface_hub import HfApi

os.environ["PYOPENGL_PLATFORM"] = "egl"

# 1. PATH FIX
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. LOAD CONFIGS
def load_swarm_configs():
    base_path = os.path.join("configs", "swarm_stage3_config.yaml")
    secrets_path = os.path.join("configs", "secrets.yaml")
    
    with open(base_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            cfg.update(yaml.safe_load(f))
    return cfg

SWARM_CFG = load_swarm_configs()

# 3. STORAGE REDIRECTION
os.environ['HF_HOME'] = SWARM_CFG["project"].get("hf_cache_dir", "/tmp/hf_cache")
os.environ['TORCH_HOME'] = SWARM_CFG["project"].get("torch_cache_dir", "/tmp/torch_cache")

# 4. IMPORT TEAMMATE'S V3 LOGIC
from training.train_v3 import V3Config, train_v3

SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M")

def upload_to_hf_and_clean(trial_dir, trial_number):
    """Sends best.pt and logs to HF, then wipes local /tmp."""
    api = HfApi()
    repo_id = SWARM_CFG["project"].get("hf_repo_id", "Denass04/RAEDiTRobotics-stage3-sweeps") 
    study_name = SWARM_CFG["project"]["study_name"]
    path_in_repo = os.path.join(f"{study_name}_{SESSION_TS}", f"trial_{trial_number}")

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
        kwargs = config["kwargs"].copy()
        for k, v in kwargs.items():
            if isinstance(v, str) and v in sampled_hps:
                kwargs[k] = sampled_hps[v] 
        suggest_method = getattr(trial, config["method"])
        sampled_hps[name] = suggest_method(name, **kwargs)
    return sampled_hps

def extract_best_success_rate(save_dir):
    """Zero-touch metric extraction: parses the teammate's JSONL log."""
    metrics_files = glob.glob(os.path.join(save_dir, "metrics_*.jsonl"))
    if not metrics_files:
        print("⚠️ Warning: No metrics file found. Returning 0.0 for success rate.")
        return 0.0
    
    latest_file = max(metrics_files, key=os.path.getctime)
    best_sr = -1.0
    
    with open(latest_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "eval_success_rate" in data:
                    sr = data["eval_success_rate"]
                    if sr > best_sr:
                        best_sr = sr
            except json.JSONDecodeError:
                continue
                
    return max(best_sr, 0.0)

def objective(trial):
    scratch_dir = SWARM_CFG["project"].get("scratch_dir", "/tmp/denassau_stage3")
    trial_dir = f"{scratch_dir}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # 1. Sample hyperparameters
    hps = sample_from_config(trial, SWARM_CFG["hyperparameters_search"])

    # 2. Base Configuration (Teammate's defaults)
    config = V3Config()
    
    # 3. Apply static overrides from your swarm config
    if "training_static" in SWARM_CFG:
        for key, value in SWARM_CFG["training_static"].items():
            setattr(config, key, value)
            
    # 4. Apply dynamic Optuna overrides
    for key, value in hps.items():
        setattr(config, key, value)
        
    # Force the save directory to our isolated trial folder
    config.save_dir = trial_dir

    try:
        # 5. Run the Unmodified Training Loop
        print(f"🚀 Starting Trial {trial.number} with config overrides: {hps}")
        train_v3(config=config, device="cuda")
        
        # 6. Extract the ultimate success rate
        best_success_rate = extract_best_success_rate(trial_dir)
        print(f"🏆 Trial {trial.number} finished with Best Success Rate: {best_success_rate * 100:.1f}%")
        
        upload_to_hf_and_clean(trial_dir, trial.number)
        return best_success_rate

    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
        raise e 

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=SWARM_CFG["project"]["study_name"],
        storage=SWARM_CFG["db_url"],
        load_if_exists=True,
        direction="maximize" 
    )

    study.optimize(
        objective,
        n_trials=SWARM_CFG["project"]["n_trials_per_worker"],
        catch=(Exception,)
    )