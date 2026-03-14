## RUN LIKE THIS:
# nohup python3 -u benchmark_branches.py > benchmark_final.log 2>&1 &

## kill all the background things after stopping a run:
# ps -ux | grep -E "benchmark_branches.py|swarm_worker.py" | grep -v "ssh" | awk '{print $2}' | xargs kill -9

## verify they are stopped:
# ps -f -u denassau | grep -E "benchmark_branches.py|swarm_worker.py" | grep -v "ssh"
import subprocess
import time
import yaml
import os
import shutil
import sys
import threading
import os

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
            if 'lr' in hp_name:
                val = 1e-4
            elif 'omega' in hp_name:
                val = 1.0
            elif 'weight_decay' in hp_name:
                val = 1e-5
            
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

        # Start the background spy
        monitor = BottleneckMonitor(interval=15)
        monitor.start()
            
        print(f">>> Running {NUM_EPOCHS} epochs (1 trial) on {branch_name}...")
        start_time = time.time()

        # This tells tqdm to only update once every 3600 seconds (basically once per epoch)
        env_vars = os.environ.copy()
        env_vars["TQDM_MININTERVAL"] = "3600" 

        subprocess.run(
            ["python3", "training/swarm_worker.py"], 
            check=True,
            env=env_vars # Pass the quiet environment
        )

        duration = time.time() - start_time
        print(f"\n✅ {branch_name} completed in {duration:.2f}s")

        # STOP AND REPORT HERE
        monitor.stop()
        monitor.join()
        monitor.report(branch_name)
        
        # Cleanup the temporary study in the DB
        cleanup_optuna_study(cfg)
        
        return duration

    finally:
        # 3. ALWAYS restore the real config, even if the training crashed
        if os.path.exists(BACKUP_CONFIG):
            shutil.move(BACKUP_CONFIG, REAL_CONFIG)
            print(">>> Restored original swarm_config.yaml")


class BottleneckMonitor(threading.Thread):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.stats = []
        self.daemon = True 

    def run(self):
        query = "utilization.gpu,utilization.memory,power.draw"
        # The loop runs until stop_event is set
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL).decode().strip().split(",")
                self.stats.append([int(x) for x in out])
            except Exception:
                pass
            
            # This is the elegant part: wait for 'interval' seconds 
            # UNLESS the stop_event is triggered, in which case it returns immediately.
            self.stop_event.wait(self.interval)

    def stop(self):
        self.stop_event.set()

    def report(self, branch_name):
        if not self.stats:
            print(f">>> [!] No bottleneck data collected for {branch_name}")
            return
        
        avg_sm = sum(s[0] for s in self.stats) / len(self.stats)
        avg_mem = sum(s[1] for s in self.stats) / len(self.stats)
        peak_pwr = max(s[2] for s in self.stats)
        
        print(f"\n--- BOTTLECHECK SUMMARY: {branch_name} ---")
        print(f"   > Avg Compute (SM) Util : {avg_sm:.1f}%")
        print(f"   > Avg Memory (BW) Util  : {avg_mem:.1f}%")
        print(f"   > Peak Power Draw       : {peak_pwr}W")
        
        if avg_sm > 85 and avg_mem < 60:
            print("   [!] DIAGNOSIS: COMPUTE BOUND. Your optimizations (TF32/Compile) are targeting the right area.")
        elif avg_mem > 75:
            print("   [!] DIAGNOSIS: IO/MEMORY BOUND. Data movement is the bottleneck. Caching DINO is your next best move.")
        elif avg_sm < 50:
            print("   [!] DIAGNOSIS: CPU/SUBPROCESS BOUND. The GPU is waiting on Python/Dataloader logic.")
        print("-" * 45)


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
        # Step 2: Run Performance
        results[PERF_BRANCH] = run_bench_session(PERF_BRANCH)

        time.sleep(30)  # Short pause between runs to ensure clean state

        # Step 1: Run Baseline - currently commented out to save time and replaced with the value gotten last time.
        results[STABLE_BRANCH] = run_bench_session(STABLE_BRANCH)
        # results[STABLE_BRANCH] = 2089.02
        

        
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