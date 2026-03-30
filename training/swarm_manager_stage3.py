import yaml
import subprocess
import os
import sys

def load_configs():
    """Merges base config and secrets into one dictionary."""
    base_path = os.path.join("configs", "swarm_stage3_config.yaml")
    if not os.path.exists(base_path):
        print(f"Error: Base config {base_path} not found. Exiting.")
        sys.exit(1)

    secrets_path = os.path.join("configs", "secrets.yaml")
    with open(base_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            secrets = yaml.safe_load(f)
            if secrets:
                config.update(secrets)
    else:
        print(f"⚠️ Warning: {secrets_path} not found. SSH might ask for password.")
        
    return config


def run_remote(cfg, node_name, domain, command, password=None, is_start_cmd=False):
    # Safely handle the domain dot whether it was included in the config or not
    formatted_domain = f".{domain.lstrip('.')}" if domain else ""
    full_address = f"{node_name}{formatted_domain}"
    
    env = os.environ.copy()
    if password:
        # SECURE: Use -e flag to read from the SSHPASS environment variable
        local_sshpass = cfg["manager"]["sshpass_path"]

        ssh_cmd = [
            local_sshpass, "-e", 
            "ssh", "-o", "ConnectTimeout=5", 
            "-o", "StrictHostKeyChecking=no", full_address, command
        ]
        env["SSHPASS"] = str(password) 
    else:
        ssh_cmd = ["ssh", "-o", "ConnectTimeout=5", full_address, command]
        
    if is_start_cmd:
        return subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    else:
        return subprocess.run(ssh_cmd, capture_output=True, text=True, env=env)


def start_swarm(cfg, target_nodes=None):
    domain = cfg['project'].get('domain', '')
    password = cfg.get('ssh_password')
    nodes_to_start = target_nodes if target_nodes is not None else cfg['nodes']
    
    print(f"Launching swarm on {len(nodes_to_start)} machines in {domain}...")
    
    # Git commands removed! Just CD, Source, Mkdir, and Run.
    remote_cmd = (
        f"export PYTORCH_ALLOC_CONF=expandable_segments:True && "
        f"cd {cfg['project']['project_root']} && "
        f"source {cfg['project']['venv_path']} && "
        f"mkdir -p {cfg['project']['log_directory']} && "
        f"nohup python3 {cfg['project']['worker_script']} > {cfg['project']['log_directory']}/worker_$(hostname)_stage3.log 2>&1 &"
    )

    for node in nodes_to_start:
        print(f"  → Starting {node}...", end=" ", flush=True)
        run_remote(cfg, node, domain, remote_cmd, password, is_start_cmd=True)
        print({"status": "Launched", "node": node})


def stop_swarm(cfg):
    domain = cfg['project'].get('domain', '')
    password = cfg.get('ssh_password')
    print("Stopping all workers...")
    stop_cmd = f"pkill -f {cfg['project']['worker_script']}"
    for node in cfg['nodes']:
        print(f"  → Stopping {node}...", end=" ", flush=True)
        run_remote(cfg, node, domain, stop_cmd, password)
        print("Stopped")


def check_status(cfg):
    domain = cfg['project'].get('domain', '')
    password = cfg.get('ssh_password')
    worker_script = cfg['project'].get('worker_script', 'training/swarm_worker_stage3.py')
    
    print(f"\n{'NODE':<20} | {'STATUS':<10} | {'GPU USAGE'}")
    print("-" * 50)

    node_statuses = {}
    
    for node in cfg['nodes']:
        check_cmd = (
            f"ps aux | grep {worker_script} | grep -v grep > /dev/null && echo 'RUNNING' || echo 'IDLE'; "
            f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'N/A'"
        )
        
        result = run_remote(cfg, node, domain, check_cmd, password, is_start_cmd=False)
    
        if result.returncode == 255:
            print(f"{node:<20} | OFFLINE  | N/A")
            node_statuses[node] = "OFFLINE"
            continue

        lines = result.stdout.strip().split('\n')
        status_text = lines[0] if len(lines) > 0 else "UNKNOWN"
        gpu_mem = lines[1] if len(lines) > 1 else "N/A"

        status_icon = " ACTIVE" if status_text == "RUNNING" else " IDLE"
        gpu_display = f"{gpu_mem} MiB" if gpu_mem != "N/A" else "N/A"

        print(f"{node:<20} | {status_icon:<10} | {gpu_display}")
        node_statuses[node] = status_text

    return node_statuses


def restart_idle(cfg):
    print("Checking cluster status to find idle nodes...")
    statuses = check_status(cfg)
    
    idle_nodes = [node for node, status in statuses.items() if status == "IDLE"]
    
    if idle_nodes:
        print(f"\nFound {len(idle_nodes)} IDLE nodes. Initiating restart...")
        start_swarm(cfg, target_nodes=idle_nodes)
    else:
        print("\nNo IDLE nodes found.")


if __name__ == "__main__":
    config = load_configs()
    
    if len(sys.argv) < 2:
        print("Usage: python3 swarm_manager_stage3.py [start|stop|status|restart_idle]")
    elif sys.argv[1] == "start":
        start_swarm(config)
    elif sys.argv[1] == "stop":
        stop_swarm(config)
    elif sys.argv[1] == "status":
        check_status(config)
    elif sys.argv[1] == "restart_idle":
        restart_idle(config)