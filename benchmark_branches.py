import subprocess
import time
import yaml
import os
import shutil
import sys

# --- CONFIGURATION ---
STABLE_BRANCH = "feature/distributed_stage1"
PERF_BRANCH = "feature/distributed_stage1-performance_boost"
REAL_CONFIG = "configs/swarm_config.yaml"
BACKUP_CONFIG = "configs/swarm_config.yaml.bak"
NUM_EPOCHS = 3
DATA_PATH = "data/complete_unified_data/can.hdf5"
NUM_TRIALS_PER_WORKER = 1

def cleanup_optuna_study(cfg):
    """Optional: Deletes the benchmark study from PostgreSQL to keep the DB clean."""
    try:
        import optuna
        study_name = cfg['project']['study_name']
        db_url = cfg['db_url']
        print(f">>> Cleaning up Optuna study: {study_name}...")
        optuna.delete_study(study_name=study_name, storage=db_url)
    except Exception as e:
        print(f"⚠️ Could not delete Optuna study: {e}")

def run_bench_session(branch_name):
    print("\n" + "="*60)
    print(f"STARTING BENCHMARK FOR: {branch_name}")
    print("="*60)
    
    # 1. Checkout the branch
    subprocess.run(["git", "checkout", branch_name], check=True)
    
    # 2. Swap the config
    if not os.path.exists(REAL_CONFIG):
        print(f"❌ Error: {REAL_CONFIG} not found!")
        return None

    # Create a backup of the original
    shutil.copy(REAL_CONFIG, BACKUP_CONFIG)
    
    try:
        # Load the data from the backup
        with open(BACKUP_CONFIG, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print(f">>> Freezing Hyperparameters for {branch_name}...")
        search_space = cfg.get('hyperparameters_search', {})
        for hp_name, settings in search_space.items():
            if 'default' in settings:
                val = settings['default']
            elif 'choices' in settings:
                val = settings['choices'][0]
            elif 'low' in settings and 'high' in settings:
                # Calculate the midpoint
                low = settings['low']
                high = settings['high']
                
                if settings.get('log', False):
                    # For log scales (like Learning Rate), midpoint is geometric
                    import math
                    val = math.pow(10, (math.log10(low) + math.log10(high)) / 2)
                else:
                    # For linear scales, midpoint is arithmetic
                    val = (low + high) / 2
            else:
                val = settings.get('low', 0)

            if 'epoch_start' in hp_name:
                if 'disc' in hp_name:
                    val = int(NUM_EPOCHS * (1/3)) # Start Discriminator at Epoch 1
                elif 'gan' in hp_name:
                    val = int(NUM_EPOCHS * (2/3))  # Start GAN at Epoch 2
                else:
                    val = 1  # Default to starting at Epoch 1 if not specified
            
            cfg['training_static'][hp_name] = val
            print(f"   > Freezing {hp_name} to: {val}")

        # Clear the search space so Optuna is bypassed
        cfg['hyperparameters_search'] = {}
        
        # Hardcoded single file and 6 epochs
        cfg['training_static']['hdf5_paths'] = [DATA_PATH]
        cfg['training_static']['num_epochs'] = NUM_EPOCHS
        cfg['project']['study_name'] = f"bench_run_{branch_name.replace('/', '_')}"
        cfg['project']['n_trials_per_worker'] = NUM_TRIALS_PER_WORKER
        
        # Overwrite the real config with these benchmark settings
        with open(REAL_CONFIG, 'w') as f:
            yaml.dump(cfg, f)
            
        print(f">>> Running {NUM_EPOCHS} epochs (1 trial) on {branch_name}...")
        start_time = time.time()
        
        # Launch the local worker process
        # This will read the REAL_CONFIG file we just overwrote
        subprocess.run([sys.executable, "training/swarm_worker.py"], check=True)
        
        duration = time.time() - start_time
        print(f"\n✅ {branch_name} completed in {duration:.2f}s")
        
        # Cleanup the temporary study in the DB
        cleanup_optuna_study(cfg)
        
        return duration

    finally:
        # 3. ALWAYS restore the real config, even if the training crashed
        if os.path.exists(BACKUP_CONFIG):
            shutil.move(BACKUP_CONFIG, REAL_CONFIG)
            print(">>> Restored original swarm_config.yaml")

if __name__ == "__main__":
    results = {}
    
    # Get current branch so we can return to it at the end
    try:
        original_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode().strip()
    except Exception:
        original_branch = PERF_BRANCH

    try:
        # Step 1: Run Baseline
        results[STABLE_BRANCH] = run_bench_session(STABLE_BRANCH)
        
        # Step 2: Run Performance
        results[PERF_BRANCH] = run_bench_session(PERF_BRANCH)
        
        # FINAL REPORTING
        print("\n" + "#"*40)
        print("         BENCHMARK SUMMARY")
        print("#"*40)
        for branch, duration in results.items():
            if duration:
                print(f"{branch:<40} : {duration:.2f}s")
        
        if STABLE_BRANCH in results and PERF_BRANCH in results:
            diff = results[STABLE_BRANCH] - results[PERF_BRANCH]
            percent = (diff / results[STABLE_BRANCH]) * 100
            print("-" * 40)
            print(f"Speedup: {percent:.2f}% ({diff:.2f}s saved)")
        print("#"*40)

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
    finally:
        # Ensure we return to where we started
        print(f"\n>>> Returning to original branch: {original_branch}")
        subprocess.run(["git", "checkout", original_branch], check=True)