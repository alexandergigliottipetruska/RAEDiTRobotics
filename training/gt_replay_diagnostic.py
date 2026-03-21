"""GT replay diagnostic: compare custom vs robomimic eval pipelines.

Replays recorded demo actions through BOTH eval pipelines and compares
step-by-step EE trajectories, rewards, and success rates. This isolates
whether the eval pipeline itself has bugs (independent of the policy).

Usage:
  PYTHONPATH=. python training/gt_replay_diagnostic.py \
    --hdf5 data/unified/robomimic/lift/ph_abs_v15.hdf5 \
    --num_demos 10 \
    --output_dir checkpoints/gt_replay_diagnostic
"""

import argparse
import logging
import os
import sys
import warnings

import h5py
import numpy as np

warnings.filterwarnings("ignore", module="robosuite")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def replay_custom_wrapper(actions_7d, initial_state, task="lift"):
    """Replay through our custom RobomimicWrapper. Returns per-step trajectory."""
    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

    env = RobomimicWrapper(task=task, abs_action=True)
    env.reset()
    env._env.sim.set_state_from_flattened(initial_state)
    env._env.sim.forward()
    env._last_obs = env._env._get_observations()

    trajectory = []
    for t in range(len(actions_7d)):
        action = actions_7d[t]
        obs, reward, done, info = env.step(action)
        proprio = env.get_proprio()  # (1, 9)
        eef_pos = proprio[0, :3].copy()
        eef_quat = proprio[0, 3:7].copy()
        trajectory.append({
            "step": t,
            "action": action.copy(),
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "reward": float(reward),
            "success": bool(info["success"]),
        })
    env.close()
    return trajectory


def replay_robomimic(actions_7d, initial_state, hdf5_path):
    """Replay through robomimic pipeline (Chi's wrappers). Returns per-step trajectory."""
    from training.eval_v3_robomimic import create_robomimic_env

    # Use n_obs_steps=1, n_action_steps=1 for step-by-step replay
    env = create_robomimic_env(
        hdf5_path, n_obs_steps=1, n_action_steps=1, max_steps=9999
    )
    # Set init_state on inner RobomimicImageWrapper
    env.env.init_state = initial_state
    obs = env.reset()

    trajectory = []
    for t in range(len(actions_7d)):
        action = actions_7d[t]
        # MultiStepWrapper expects (n_action_steps, action_dim)
        obs, reward, done, info = env.step(action.reshape(1, -1))

        # Extract proprio from obs dict
        eef_pos = obs['robot0_eef_pos'][-1, :3].copy()
        eef_quat = obs['robot0_eef_quat'][-1].copy()
        trajectory.append({
            "step": t,
            "action": action.copy(),
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "reward": float(reward),
            "success": bool(reward > 0.5),
        })
    env.close()
    return trajectory


def plot_comparison(traj_custom, traj_robomimic, demo_key, output_dir):
    """Plot side-by-side EE trajectories, action diffs, and rewards."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [d["step"] for d in traj_custom]
    T = len(steps)

    # Extract data
    ee_custom = np.array([d["eef_pos"] for d in traj_custom])
    ee_robomimic = np.array([d["eef_pos"] for d in traj_robomimic])
    act_custom = np.array([d["action"] for d in traj_custom])
    act_robomimic = np.array([d["action"] for d in traj_robomimic])
    rew_custom = np.array([d["reward"] for d in traj_custom])
    rew_robomimic = np.array([d["reward"] for d in traj_robomimic])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GT Replay Diagnostic: {demo_key}', fontsize=14, fontweight='bold')

    labels_xyz = ['X', 'Y', 'Z']
    colors = ['tab:blue', 'tab:orange']

    # Row 1: EE position per axis
    for i in range(3):
        ax = axes[0, i]
        ax.plot(steps, ee_custom[:, i], '-', color=colors[0], label='Custom', alpha=0.8)
        ax.plot(steps, ee_robomimic[:, i], '--', color=colors[1], label='Robomimic', alpha=0.8)
        ax.set_title(f'EE Position {labels_xyz[i]}')
        ax.set_xlabel('Step')
        ax.set_ylabel(f'{labels_xyz[i]} (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2, Col 0: EE position error (L2 norm)
    ax = axes[1, 0]
    ee_diff = np.linalg.norm(ee_custom - ee_robomimic, axis=1)
    ax.plot(steps, ee_diff * 1000, 'r-', linewidth=1.5)
    ax.set_title('EE Position Diff (mm)')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 error (mm)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1mm')
    ax.legend(fontsize=8)

    # Row 2, Col 1: Action diff (L2 norm)
    ax = axes[1, 1]
    act_diff = np.linalg.norm(act_custom - act_robomimic, axis=1)
    ax.plot(steps, act_diff, 'purple', linewidth=1.5)
    ax.set_title('Action Diff (L2 norm)')
    ax.set_xlabel('Step')
    ax.set_ylabel('L2 error')
    ax.grid(True, alpha=0.3)

    # Row 2, Col 2: Rewards
    ax = axes[1, 2]
    ax.plot(steps, rew_custom, '-', color=colors[0], label='Custom', linewidth=2)
    ax.plot(steps, rew_robomimic, '--', color=colors[1], label='Robomimic', linewidth=2)
    ax.set_title('Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, f"{demo_key}.png")
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="GT replay diagnostic: compare custom vs robomimic eval pipelines"
    )
    parser.add_argument("--hdf5", required=True,
                        help="Unified HDF5 with actions, states, and env_args")
    parser.add_argument("--task", default="lift")
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--output_dir", default="checkpoints/gt_replay_diagnostic")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Open HDF5
    f = h5py.File(args.hdf5, "r")
    demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])

    # Check states exist
    if "states" not in f[f"data/{demo_keys[0]}"]:
        log.error("No 'states' dataset in HDF5 — cannot do GT replay")
        f.close()
        return

    demo_keys = demo_keys[:args.num_demos]
    log.info("GT replay diagnostic: %d demos from %s", len(demo_keys), args.hdf5)

    # Check action dim
    ac_dim = f[f"data/{demo_keys[0]}/actions"].shape[-1]
    rot6d = ac_dim == 10
    log.info("Action dim: %d%s", ac_dim, " (rot6d → will convert to 7D)" if rot6d else "")

    if rot6d:
        from data_pipeline.utils.rotation import convert_actions_from_rot6d

    # Results tracking
    results_custom = []
    results_robomimic = []

    for demo_key in demo_keys:
        actions_raw = f[f"data/{demo_key}/actions"][:]
        initial_state = f[f"data/{demo_key}/states"][0]

        if rot6d:
            actions_7d = convert_actions_from_rot6d(actions_raw)
        else:
            actions_7d = actions_raw.copy()

        log.info("--- %s (%d steps, action range [%.3f, %.3f]) ---",
                 demo_key, len(actions_7d),
                 actions_7d.min(), actions_7d.max())

        # Pipeline A: Custom wrapper
        log.info("  Running custom RobomimicWrapper...")
        traj_custom = replay_custom_wrapper(actions_7d, initial_state, args.task)
        success_custom = any(d["success"] for d in traj_custom)
        first_success_custom = next(
            (d["step"] for d in traj_custom if d["success"]), -1
        )

        # Pipeline B: Robomimic
        log.info("  Running robomimic pipeline...")
        traj_robomimic = replay_robomimic(actions_7d, initial_state, args.hdf5)
        success_robomimic = any(d["success"] for d in traj_robomimic)
        first_success_robomimic = next(
            (d["step"] for d in traj_robomimic if d["success"]), -1
        )

        # Compare
        ee_custom = np.array([d["eef_pos"] for d in traj_custom])
        ee_robomimic = np.array([d["eef_pos"] for d in traj_robomimic])
        ee_diff = np.linalg.norm(ee_custom - ee_robomimic, axis=1)

        log.info("  Custom:    %s (step %d)", "SUCCESS" if success_custom else "FAIL", first_success_custom)
        log.info("  Robomimic: %s (step %d)", "SUCCESS" if success_robomimic else "FAIL", first_success_robomimic)
        log.info("  EE diff:   mean=%.3fmm max=%.3fmm", ee_diff.mean() * 1000, ee_diff.max() * 1000)

        results_custom.append(success_custom)
        results_robomimic.append(success_robomimic)

        # Plot
        plot_path = plot_comparison(traj_custom, traj_robomimic, demo_key, args.output_dir)
        log.info("  Plot saved: %s", plot_path)

    f.close()

    # Summary
    n_custom = sum(results_custom)
    n_robomimic = sum(results_robomimic)
    log.info("")
    log.info("=" * 60)
    log.info("GT REPLAY SUMMARY")
    log.info("=" * 60)
    log.info("%-15s | %-10s | %-10s", "Demo", "Custom", "Robomimic")
    log.info("-" * 40)
    for i, dk in enumerate(demo_keys):
        c = "PASS" if results_custom[i] else "FAIL"
        r = "PASS" if results_robomimic[i] else "FAIL"
        match = "  " if results_custom[i] == results_robomimic[i] else " !"
        log.info("%-15s | %-10s | %-10s%s", dk, c, r, match)
    log.info("-" * 40)
    log.info("%-15s | %d/%d       | %d/%d", "TOTAL",
             n_custom, len(demo_keys), n_robomimic, len(demo_keys))
    log.info("=" * 60)

    if n_custom == n_robomimic == len(demo_keys):
        log.info("BOTH PIPELINES PASS — eval pipeline is correct")
    elif n_custom > n_robomimic:
        log.warning("ROBOMIMIC PIPELINE WORSE — check action/obs conversion in eval_v3_robomimic.py")
    elif n_robomimic > n_custom:
        log.warning("CUSTOM PIPELINE WORSE — robomimic controller config is better")
    else:
        log.warning("MIXED RESULTS — check per-demo plots for divergence patterns")


if __name__ == "__main__":
    main()
