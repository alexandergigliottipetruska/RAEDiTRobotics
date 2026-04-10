"""Sweep eval-time hyperparameters to find what helps closed-loop performance.

Tests combinations of:
  - temporal_ensemble: on/off
  - ensemble_gain: 0.01 (current), 0.1, 0.5, 1.0
  - exec_horizon (T_a): 1, 4, 5, 8, 10
  - DDIM inference steps: 100 (current), 50, 20

Loads model + env ONCE, reuses across all configs.
Runs N episodes per config on the same demo scenes for fair comparison.

Usage:
    PYTHONPATH=. python -u training/sweep_eval_configs.py \
        --checkpoint checkpoints/v3_rlbench_open_drawer/epoch_699.pt \
        --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
        --eval_hdf5 /virtual/csc415user/data/rlbench/open_drawer.hdf5 \
        --task open_drawer \
        --demo_pickles /virtual/csc415user/data/rlbench/train/open_drawer/all_variations/episodes \
        --episodes_per_config 5
"""
import argparse
import logging
import os
import pickle
import signal
import sys
import time
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.conversion.compute_norm_stats import load_norm_stats
from data_pipeline.utils.rotation import convert_actions_rot6d_to_quat
from models.policy_v3 import PolicyDiTv3
from models.stage1_bridge import Stage1Bridge
from training.eval_v3 import V3PolicyWrapper
from training.eval_v3_rlbench import _denorm_and_convert, _temporal_ensemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Episode runner (inline, supports all config knobs) ──────────────────────

class _EpisodeTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _EpisodeTimeout()


def run_episode(
    policy_wrapper,
    env,
    action_stats,
    proprio_stats,
    max_steps,
    demo,
    *,
    temporal_ensemble: bool = True,
    ensemble_gain: float = 0.01,
    exec_horizon: int = 1,
    obs_horizon: int = 2,
    timeout: int = 180,
):
    """Run one episode with the given config. Returns dict with success/steps."""
    # Reset to demo scene
    if demo is not None:
        try:
            desc, obs = env._task.reset_to_demo(demo)
        except (KeyError, AttributeError):
            env._task.set_variation(demo.variation_number)
            desc, obs = env._task.reset()
        env._last_obs = obs
    else:
        env.reset()

    # Observation buffers
    init_images = env.get_multiview_images()
    init_proprio = env.get_proprio()
    img_buffer = deque([init_images] * obs_horizon, maxlen=obs_horizon)
    proprio_buffer = deque([init_proprio] * obs_horizon, maxlen=obs_horizon)
    view_present = env.get_view_present()

    p_min, p_max = proprio_stats["min"], proprio_stats["max"]

    T_pred = None
    action_history = deque(maxlen=50)
    pending_actions = []
    step_count = 0
    info = {}

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)

    try:
        while step_count < max_steps:
            if pending_actions:
                action = pending_actions.pop(0)
            else:
                # Build observation
                images_seq = np.concatenate(list(img_buffer), axis=0)
                proprio_seq = np.concatenate(list(proprio_buffer), axis=0)

                proprio_norm = proprio_seq.copy()
                pos_range = np.clip(p_max[:3] - p_min[:3], 1e-6, None)
                proprio_norm[..., :3] = 2.0 * (proprio_seq[..., :3] - p_min[:3]) / pos_range - 1.0
                g_min, g_max = p_min[7:], p_max[7:]
                g_range = np.clip(g_max - g_min, 1e-6, None)
                proprio_norm[..., 7:] = 2.0 * (proprio_seq[..., 7:] - g_min) / g_range - 1.0

                with torch.no_grad():
                    pred = policy_wrapper.predict(
                        torch.from_numpy(images_seq).unsqueeze(0),
                        torch.from_numpy(proprio_norm).unsqueeze(0),
                        torch.from_numpy(view_present).unsqueeze(0),
                    )

                raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
                actions_8d = _denorm_and_convert(raw, action_stats)

                if T_pred is None:
                    T_pred = actions_8d.shape[0]

                if temporal_ensemble:
                    action_history.append((step_count, actions_8d))
                    action = _temporal_ensemble(
                        list(action_history), step_count, T_pred, gain=ensemble_gain,
                    )
                    if exec_horizon > 1:
                        for j in range(1, min(exec_horizon, T_pred)):
                            a = _temporal_ensemble(
                                list(action_history), step_count + j, T_pred,
                                gain=ensemble_gain,
                            )
                            pending_actions.append(a)
                else:
                    action = actions_8d[0]
                    if exec_horizon > 1:
                        for j in range(1, min(exec_horizon, len(actions_8d))):
                            pending_actions.append(actions_8d[j])

            try:
                _, reward, done, info = env.step(action)
            except Exception:
                break

            step_count += 1
            img_buffer.append(env.get_multiview_images())
            proprio_buffer.append(env.get_proprio())

            if done:
                break

    except _EpisodeTimeout:
        log.warning("    Episode timed out after %ds", timeout)
        return {"success": False, "steps": step_count}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    success = info.get("success", False) if isinstance(info, dict) else False
    return {"success": success, "steps": step_count}


# ── Main sweep ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep eval-time configs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--eval_hdf5", required=True)
    parser.add_argument("--task", default="open_drawer")
    parser.add_argument("--demo_pickles", required=True)
    parser.add_argument("--episodes_per_config", type=int, default=5,
                        help="Episodes per config (more = more reliable, slower)")
    parser.add_argument("--ac_dim", type=int, default=10)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--n_active_cams", type=int, default=4)
    parser.add_argument("--T_pred", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ONCE ──
    log.info("Loading model...")
    bridge = Stage1Bridge(checkpoint_path=args.stage1_checkpoint)
    policy = PolicyDiTv3(
        bridge=bridge, ac_dim=args.ac_dim,
        proprio_dim=args.proprio_dim, n_active_cams=args.n_active_cams,
        T_pred=args.T_pred,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt:
        ema = ckpt["ema"]
        if "averaged_model" in ema:
            policy.load_state_dict(ema["averaged_model"], strict=False)
        elif "denoiser" in ema:
            def _strip(sd):
                prefix = "_orig_mod."
                return {k.removeprefix(prefix): v for k, v in sd.items()} if any(
                    k.startswith(prefix) for k in sd) else sd
            policy.denoiser.load_state_dict(_strip(ema["denoiser"]))
            policy.obs_encoder.load_state_dict(_strip(ema["obs_encoder"]))
            policy.bridge.adapter.load_state_dict(_strip(ema["adapter"]))
        log.info("Loaded EMA weights")
    policy.eval()

    wrapper = V3PolicyWrapper(policy, device=str(device))
    norm_stats = load_norm_stats(args.eval_hdf5)
    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    # ── Load demo pickles ONCE ──
    ep_root = Path(args.demo_pickles)
    ep_dirs = sorted(
        [d for d in ep_root.iterdir()
         if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
        key=lambda d: int(d.name.replace("episode", "")),
    )
    demos = []
    for d in ep_dirs[:args.episodes_per_config]:
        with open(d / "low_dim_obs.pkl", "rb") as fh:
            demos.append(pickle.load(fh))
    log.info("Loaded %d demo pickles", len(demos))

    # ── Create env ONCE ──
    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper
    max_steps = 150  # open_drawer

    log.info("Creating RLBenchWrapper for %s...", args.task)
    env = RLBenchWrapper(task_name=args.task, headless=True)

    # ── Define sweep configs ──
    # Format: (label, temporal_ensemble, ensemble_gain, exec_horizon, ddim_steps)
    configs = []

    # 1. No temporal ensemble, varying exec_horizon
    for ta in [1, 4, 5, 10]:
        configs.append((f"no_ens_ta{ta}", False, 0.0, ta, 100))

    # 2. Temporal ensemble with different gains, T_a=1
    for gain in [0.01, 0.1, 0.5, 1.0]:
        configs.append((f"ens_g{gain}_ta1", True, gain, 1, 100))

    # 3. Temporal ensemble with different gains, T_a=4
    for gain in [0.01, 0.1, 0.5, 1.0]:
        configs.append((f"ens_g{gain}_ta4", True, gain, 4, 100))

    # 4. Temporal ensemble with different gains, T_a=5
    for gain in [0.1, 0.5]:
        configs.append((f"ens_g{gain}_ta5", True, gain, 5, 100))

    # 5. DDIM steps sweep (with best-guess ensemble settings)
    for ddim in [50, 20, 10]:
        configs.append((f"ens_g0.1_ta4_ddim{ddim}", True, 0.1, 4, ddim))

    log.info("=" * 70)
    log.info("SWEEP: %d configs × %d episodes = %d total episodes",
             len(configs), args.episodes_per_config,
             len(configs) * args.episodes_per_config)
    log.info("Estimated time: %.0f - %.0f minutes",
             len(configs) * args.episodes_per_config * 1.0,
             len(configs) * args.episodes_per_config * 2.0)
    log.info("=" * 70)

    # ── Run sweep ──
    all_results = {}
    original_ddim_steps = policy.eval_diffusion_steps

    for cfg_idx, (label, use_ens, gain, ta, ddim) in enumerate(configs):
        # Override DDIM steps on the policy
        policy.eval_diffusion_steps = ddim

        successes = 0
        total_steps = 0
        t0 = time.time()

        for ep_idx in range(args.episodes_per_config):
            demo = demos[ep_idx] if ep_idx < len(demos) else None
            try:
                result = run_episode(
                    wrapper, env, action_stats, proprio_stats,
                    max_steps, demo,
                    temporal_ensemble=use_ens,
                    ensemble_gain=gain,
                    exec_horizon=ta,
                )
            except Exception as e:
                log.warning("    %s ep%d crashed: %s", label, ep_idx, e)
                result = {"success": False, "steps": 0}
                # Recreate env on crash
                try:
                    env.close()
                except Exception:
                    pass
                env = RLBenchWrapper(task_name=args.task, headless=True)

            tag = "OK" if result["success"] else "  "
            successes += int(result["success"])
            total_steps += result["steps"]

        elapsed = time.time() - t0
        sr = successes / args.episodes_per_config
        avg_steps = total_steps / args.episodes_per_config

        all_results[label] = {
            "success_rate": sr,
            "successes": successes,
            "total": args.episodes_per_config,
            "avg_steps": avg_steps,
            "elapsed_s": elapsed,
            "config": {
                "temporal_ensemble": use_ens,
                "ensemble_gain": gain,
                "exec_horizon": ta,
                "ddim_steps": ddim,
            },
        }

        log.info("[%2d/%2d] %-25s  %d/%d (%.0f%%)  avg_steps=%.0f  %.1fs",
                 cfg_idx + 1, len(configs), label,
                 successes, args.episodes_per_config, sr * 100,
                 avg_steps, elapsed)

    # Restore original DDIM steps
    policy.eval_diffusion_steps = original_ddim_steps

    # ── Summary table ──
    print("\n" + "=" * 80)
    print(f"SWEEP RESULTS — {args.checkpoint}")
    print("=" * 80)
    print(f"{'Config':<28} {'Ens':>3} {'Gain':>5} {'T_a':>3} {'DDIM':>4}  "
          f"{'SR':>5}  {'Avg Steps':>9}  {'Time':>6}")
    print("-" * 80)

    for label, r in all_results.items():
        c = r["config"]
        ens_str = "Y" if c["temporal_ensemble"] else "N"
        gain_str = f"{c['ensemble_gain']:.2f}" if c["temporal_ensemble"] else "  - "
        print(f"{label:<28} {ens_str:>3} {gain_str:>5} {c['exec_horizon']:>3} "
              f"{c['ddim_steps']:>4}  "
              f"{r['successes']}/{r['total']:>1}  "
              f"{r['avg_steps']:>9.1f}  "
              f"{r['elapsed_s']:>5.0f}s")

    print("=" * 80)

    # Highlight any successes
    any_success = any(r["successes"] > 0 for r in all_results.values())
    if any_success:
        print("\nCONFIGS WITH SUCCESS:")
        for label, r in all_results.items():
            if r["successes"] > 0:
                c = r["config"]
                print(f"  {label}: {r['successes']}/{r['total']} "
                      f"(ens={c['temporal_ensemble']}, gain={c['ensemble_gain']}, "
                      f"T_a={c['exec_horizon']}, ddim={c['ddim_steps']})")
    else:
        print("\nNo successes across any configuration.")
        print("This confirms the issue is compounding error, not eval-time hyperparameters.")
        print("Next steps: observation noise augmentation or L1 loss (require retraining).")

    env.close()


if __name__ == "__main__":
    main()
