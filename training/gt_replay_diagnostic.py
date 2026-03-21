"""GT replay diagnostic: compare custom vs robomimic eval pipelines.

Replays recorded demo actions through BOTH eval pipelines (and optionally
a "Chi-style" normalized pipeline) and compares step-by-step EE trajectories,
rewards, and success rates.

Usage:
  PYTHONPATH=. python training/gt_replay_diagnostic.py \
    --hdf5 data/unified/robomimic/lift/ph_abs_v15.hdf5 \
    --num_demos 200 \
    --output_dir checkpoints/gt_replay_diagnostic
"""

import argparse
import json
import logging
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import numpy as np

warnings.filterwarnings("ignore", module="robosuite")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# ── Replay functions (take pre-created env, no create/close) ──


def replay_custom(env, actions_7d, initial_state):
    """Pipeline A: our custom RobomimicWrapper. Env must already exist."""
    env.reset()
    env._env.sim.set_state_from_flattened(initial_state)
    env._env.sim.forward()
    env._last_obs = env._env._get_observations()

    trajectory = []
    for t in range(len(actions_7d)):
        obs, reward, done, info = env.step(actions_7d[t])
        proprio = env.get_proprio()
        trajectory.append({
            "step": t,
            "eef_pos": proprio[0, :3].copy(),
            "reward": float(reward),
            "success": bool(info["success"]),
        })
    return trajectory


def replay_robomimic_denorm(env, actions_7d, initial_state):
    """Pipeline B: robomimic env with denormalized actions. Env must already exist."""
    env.env.init_state = initial_state
    obs = env.reset()

    trajectory = []
    for t in range(len(actions_7d)):
        obs, reward, done, info = env.step(actions_7d[t].reshape(1, -1))
        trajectory.append({
            "step": t,
            "eef_pos": obs['robot0_eef_pos'][-1, :3].copy(),
            "reward": float(reward),
            "success": bool(reward > 0.5),
        })
    return trajectory


def replay_robomimic_normalized(env, actions_7d, initial_state, norm_stats):
    """Pipeline C: Chi-style normalized actions (input_min=-1). Env must already exist."""
    action_min = norm_stats["actions"]["min"]
    action_max = norm_stats["actions"]["max"]
    a_range = action_max - action_min
    a_range = np.where(np.abs(a_range) < 1e-6, 1.0, a_range)

    env.env.init_state = initial_state
    obs = env.reset()

    trajectory = []
    for t in range(len(actions_7d)):
        norm_action = 2.0 * (actions_7d[t] - action_min) / a_range - 1.0
        obs, reward, done, info = env.step(norm_action.reshape(1, -1).astype(np.float32))
        trajectory.append({
            "step": t,
            "eef_pos": obs['robot0_eef_pos'][-1, :3].copy(),
            "reward": float(reward),
            "success": bool(reward > 0.5),
        })
    return trajectory


# ── Environment factories (called ONCE) ──


def create_custom_env(task="lift"):
    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
    return RobomimicWrapper(task=task, abs_action=True)


def create_robomimic_denorm_env(hdf5_path):
    from training.eval_v3_robomimic import create_robomimic_env
    return create_robomimic_env(hdf5_path, n_obs_steps=1, n_action_steps=1, max_steps=9999)


def create_robomimic_chinorm_env(hdf5_path):
    """Create robomimic env with Chi-style config: control_delta=False, input_min=-1."""
    from collections import defaultdict
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    from data_pipeline.envs.robomimic_image_wrapper import RobomimicImageWrapper
    from data_pipeline.envs.multistep_wrapper import MultiStepWrapper
    from training.eval_v3_robomimic import LIFT_SHAPE_META

    shape_meta = LIFT_SHAPE_META
    modality_mapping = defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    env_meta['env_kwargs']['use_object_obs'] = False

    # Chi-style: only set control_delta=False, keep input_min=-1
    ctrl = env_meta['env_kwargs']['controller_configs']
    if 'body_parts' in ctrl:
        for part in ctrl['body_parts'].values():
            part['control_delta'] = False
    else:
        ctrl['control_delta'] = False

    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False,
        render_offscreen=True, use_image_obs=True,
    )
    robomimic_env.env.hard_reset = False

    return MultiStepWrapper(
        RobomimicImageWrapper(
            env=robomimic_env, shape_meta=shape_meta,
            init_state=None, render_obs_key='agentview_image',
        ),
        n_obs_steps=1, n_action_steps=1, max_episode_steps=9999,
    )


# ── Plotting ──


def plot_comparison(traj_custom, traj_robomimic, demo_key, output_dir):
    """Plot side-by-side EE trajectories and rewards."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = [d["step"] for d in traj_custom]
    ee_a = np.array([d["eef_pos"] for d in traj_custom])
    ee_b = np.array([d["eef_pos"] for d in traj_robomimic])
    rew_a = np.array([d["reward"] for d in traj_custom])
    rew_b = np.array([d["reward"] for d in traj_robomimic])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GT Replay: {demo_key}', fontsize=14, fontweight='bold')
    colors = ['tab:blue', 'tab:orange']

    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = axes[0, i]
        ax.plot(steps, ee_a[:, i], '-', color=colors[0], label='Custom', alpha=0.8)
        ax.plot(steps, ee_b[:, i], '--', color=colors[1], label='Robomimic', alpha=0.8)
        ax.set_title(f'EE {label}'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ee_diff = np.linalg.norm(ee_a - ee_b, axis=1)
    ax.plot(steps, ee_diff * 1000, 'r-'); ax.set_title('EE Diff (mm)'); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(steps, rew_a, '-', color=colors[0], label='Custom', linewidth=2)
    ax.plot(steps, rew_b, '--', color=colors[1], label='Robomimic', linewidth=2)
    ax.set_title('Reward'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[1, 1].axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"{demo_key}.png"), dpi=100, bbox_inches='tight')
    plt.close()


def plot_summary(results, output_dir):
    """Plot success step comparison across all demos."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pipelines = list(results[list(results.keys())[0]].keys())
    demo_keys = list(results.keys())
    n_demos = len(demo_keys)

    success_steps = {p: [] for p in pipelines}
    for dk in demo_keys:
        for p in pipelines:
            step = results[dk][p]["first_success_step"]
            success_steps[p].append(step if step >= 0 else np.nan)

    colors = {'custom': 'tab:blue', 'robomimic_denorm': 'tab:orange', 'robomimic_normalized': 'tab:green'}
    labels = {'custom': 'Custom', 'robomimic_denorm': 'Robomimic (denorm)', 'robomimic_normalized': 'Chi-norm'}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'GT Replay Summary: {n_demos} demos', fontsize=14, fontweight='bold')

    ax = axes[0]
    for p in pipelines:
        n = sum(1 for s in success_steps[p] if not np.isnan(s))
        ax.bar(labels.get(p, p), n, color=colors.get(p, 'gray'), alpha=0.8)
        ax.text(labels.get(p, p), n + 0.5, f"{n}/{n_demos}", ha='center', fontsize=10)
    ax.set_ylabel('Demos Succeeded'); ax.set_title('Success Count'); ax.set_ylim(0, n_demos + 5)

    if 'robomimic_denorm' in pipelines:
        s_a = np.array(success_steps['custom'])
        s_b = np.array(success_steps['robomimic_denorm'])
        valid = ~np.isnan(s_a) & ~np.isnan(s_b)

        ax = axes[1]
        ax.scatter(s_a[valid], s_b[valid], alpha=0.5, s=20)
        lim = max(np.nanmax(s_a[valid]), np.nanmax(s_b[valid])) + 5
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
        ax.set_xlabel('Custom (step)'); ax.set_ylabel('Robomimic (step)')
        ax.set_title('Success Step Comparison')
        earlier = np.sum(s_b[valid] < s_a[valid])
        later = np.sum(s_b[valid] > s_a[valid])
        same = np.sum(s_b[valid] == s_a[valid])
        ax.text(0.05, 0.95, f"Robomimic earlier: {earlier}\nSame: {same}\nCustom earlier: {later}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax = axes[2]
        diff = s_b[valid] - s_a[valid]
        ax.hist(diff, bins=range(int(diff.min()) - 1, int(diff.max()) + 2), alpha=0.7, color='purple')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step diff (robomimic - custom)'); ax.set_ylabel('Count')
        ax.set_title(f'Step Difference (mean={diff.mean():.2f})')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(output_dir, "summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    log.info("Summary plot saved: %s", os.path.join(output_dir, "summary.png"))


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="GT replay diagnostic")
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--task", default="lift")
    parser.add_argument("--num_demos", type=int, default=200)
    parser.add_argument("--output_dir", default="checkpoints/gt_replay_diagnostic")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--chi_norm", action="store_true",
                        help="Also test Chi-style normalized action pipeline")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel env instances (default: 4)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    f = h5py.File(args.hdf5, "r")
    demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo")])

    if "states" not in f[f"data/{demo_keys[0]}"]:
        log.error("No 'states' in HDF5"); f.close(); return

    demo_keys = demo_keys[:args.num_demos]
    ac_dim = f[f"data/{demo_keys[0]}/actions"].shape[-1]
    rot6d = ac_dim == 10
    log.info("GT replay: %d demos, action_dim=%d%s", len(demo_keys), ac_dim,
             " (rot6d)" if rot6d else "")

    if rot6d:
        from data_pipeline.utils.rotation import convert_actions_from_rot6d

    # Compute chi-norm stats if needed (before env creation)
    norm_stats_7d = None
    if args.chi_norm:
        all_actions_7d = []
        for dk in demo_keys:
            acts = f[f"data/{dk}/actions"][:]
            if rot6d:
                acts = convert_actions_from_rot6d(acts)
            all_actions_7d.append(acts)
        all_actions_7d = np.concatenate(all_actions_7d, axis=0)
        norm_stats_7d = {
            "actions": {
                "min": all_actions_7d.min(axis=0).astype(np.float32),
                "max": all_actions_7d.max(axis=0).astype(np.float32),
            }
        }
        log.info("Chi-norm stats: min=%s max=%s",
                 np.array2string(norm_stats_7d["actions"]["min"], precision=3),
                 np.array2string(norm_stats_7d["actions"]["max"], precision=3))

    pipelines = ['custom', 'robomimic_denorm']
    if args.chi_norm:
        pipelines.append('robomimic_normalized')

    results = {}

    # Pre-load all demo data
    log.info("Loading demo data...")
    demo_data = []
    for demo_key in demo_keys:
        actions_raw = f[f"data/{demo_key}/actions"][:]
        initial_state = f[f"data/{demo_key}/states"][0].copy()
        if rot6d:
            actions_7d = convert_actions_from_rot6d(actions_raw)
        else:
            actions_7d = actions_raw.copy()
        demo_data.append((demo_key, actions_7d, initial_state))

    # Parallel replay: create N env pairs, each running a demo independently.
    n_workers = args.workers
    log.info("Creating %d worker env pairs for parallel replay...", n_workers)

    # Each worker needs its own env instances (MuJoCo sim state not shareable)
    worker_envs = []
    for _ in range(n_workers):
        we = {"custom": create_custom_env(args.task)}
        we["robomimic"] = create_robomimic_denorm_env(args.hdf5)
        if args.chi_norm:
            we["chinorm"] = create_robomimic_chinorm_env(args.hdf5)
        worker_envs.append(we)

    log.info("All %d workers ready. Starting parallel replay...", n_workers)

    def run_single_demo(worker_id, demo_idx):
        demo_key, actions_7d, initial_state = demo_data[demo_idx]
        we = worker_envs[worker_id]

        traj_a = replay_custom(we["custom"], actions_7d, initial_state)
        step_a = next((d["step"] for d in traj_a if d["success"]), -1)

        traj_b = replay_robomimic_denorm(we["robomimic"], actions_7d, initial_state)
        step_b = next((d["step"] for d in traj_b if d["success"]), -1)

        step_c = None
        traj_c = None
        if "chinorm" in we:
            traj_c = replay_robomimic_normalized(we["chinorm"], actions_7d, initial_state, norm_stats_7d)
            step_c = next((d["step"] for d in traj_c if d["success"]), -1)

        ee_a = np.array([d["eef_pos"] for d in traj_a])
        ee_b = np.array([d["eef_pos"] for d in traj_b])
        ee_diff = np.linalg.norm(ee_a - ee_b, axis=1).mean() * 1000

        return demo_idx, demo_key, step_a, step_b, step_c, ee_diff, traj_a, traj_b

    # Submit all demos to thread pool, each assigned a worker round-robin
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for i in range(len(demo_data)):
            worker_id = i % n_workers
            fut = executor.submit(run_single_demo, worker_id, i)
            futures[fut] = i

        for fut in as_completed(futures):
            _, demo_key, step_a, step_b, step_c, ee_diff, traj_a, traj_b = fut.result()

            results[demo_key] = {
                'custom': {"success": step_a >= 0, "first_success_step": step_a},
                'robomimic_denorm': {"success": step_b >= 0, "first_success_step": step_b},
            }
            if step_c is not None:
                results[demo_key]['robomimic_normalized'] = {
                    "success": step_c >= 0, "first_success_step": step_c,
                }

            completed += 1
            status = f"A={'OK' if step_a>=0 else 'X'}({step_a}) B={'OK' if step_b>=0 else 'X'}({step_b})"
            if step_c is not None:
                status += f" C={'OK' if step_c>=0 else 'X'}({step_c})"
            status += f" diff={ee_diff:.1f}mm"
            log.info("[%d/%d] %s: %s", completed, len(demo_data), demo_key, status)

            if not args.skip_plots:
                plot_comparison(traj_a, traj_b, demo_key, args.output_dir)

    # Cleanup worker envs
    for we in worker_envs:
        we["custom"].close()
        try:
            we["robomimic"].env.env.close()
        except Exception:
            pass
        if "chinorm" in we:
            try:
                we["chinorm"].env.env.close()
            except Exception:
                pass

    f.close()

    # ── Summary ──
    log.info("")
    log.info("=" * 70)
    log.info("GT REPLAY SUMMARY (%d demos)", len(demo_keys))
    log.info("=" * 70)

    for p in pipelines:
        n = sum(1 for dk in results if results[dk][p]["success"])
        log.info("  %-25s: %d/%d (%.1f%%)", p, n, len(demo_keys), 100 * n / len(demo_keys))

    steps_a = [results[dk]['custom']['first_success_step'] for dk in results]
    steps_b = [results[dk]['robomimic_denorm']['first_success_step'] for dk in results]
    both = [(a, b) for a, b in zip(steps_a, steps_b) if a >= 0 and b >= 0]
    if both:
        diffs = [b - a for a, b in both]
        log.info("")
        log.info("Step comparison (robomimic_denorm - custom) where both succeed:")
        log.info("  Mean diff: %.2f steps", np.mean(diffs))
        log.info("  Robomimic earlier: %d", sum(1 for d in diffs if d < 0))
        log.info("  Same step: %d", sum(1 for d in diffs if d == 0))
        log.info("  Custom earlier: %d", sum(1 for d in diffs if d > 0))

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, 'w') as jf:
        json.dump(results, jf, indent=2, default=str)
    log.info("Results saved: %s", json_path)

    try:
        plot_summary(results, args.output_dir)
    except Exception as e:
        log.warning("Plot failed: %s", e)

    log.info("=" * 70)


if __name__ == "__main__":
    main()
