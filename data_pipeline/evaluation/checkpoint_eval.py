"""Checkpoint evaluation protocol.

Following Chi et al. and robomimic conventions, reports two numbers:
1. Best checkpoint: Highest success rate across all saved checkpoints.
2. Average of last N: Mean success rate of the last 10 checkpoints.

Also supports multi-seed evaluation with pooled Wilson confidence intervals.
"""

from glob import glob

import numpy as np

from data_pipeline.evaluation.metrics import aggregate_seeds, wilson_ci
from data_pipeline.evaluation.rollout import evaluate_policy


def evaluate_all_checkpoints(
    policy_class,
    ckpt_dir: str,
    env,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
    ckpt_pattern: str = "ckpt_*.pt",
    last_n: int = 10,
    **eval_kwargs,
) -> dict:
    """Evaluate all saved checkpoints and return best + last-N avg.

    Args:
        policy_class: Class with a .load(ckpt_path) classmethod that
                      returns a policy with .predict().
        ckpt_dir: Directory containing checkpoint files.
        env: BaseManipulationEnv instance.
        action_mean, action_std: Normalization stats for actions.
        proprio_mean, proprio_std: Normalization stats for proprio.
        ckpt_pattern: Glob pattern for checkpoint files.
        last_n: Number of last checkpoints to average.
        **eval_kwargs: Passed to evaluate_policy (num_episodes, max_steps, etc.)

    Returns:
        Dict with keys: best_sr, best_ckpt, last_n_avg, per_checkpoint.
    """
    ckpts = sorted(glob(f"{ckpt_dir}/{ckpt_pattern}"))

    if not ckpts:
        return {
            "best_sr": 0.0,
            "best_ckpt": None,
            "last_n_avg": 0.0,
            "per_checkpoint": {},
        }

    results = {}
    for ckpt_path in ckpts:
        policy = policy_class.load(ckpt_path)
        sr, episode_results = evaluate_policy(
            policy,
            env,
            action_mean=action_mean,
            action_std=action_std,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            **eval_kwargs,
        )

        n_total = len(episode_results)
        n_successes = sum(1 for r in episode_results if r["success"])
        ci_lo, ci_hi = wilson_ci(n_successes, n_total)

        results[ckpt_path] = {
            "success_rate": sr,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "episodes": episode_results,
        }

    all_srs = [r["success_rate"] for r in results.values()]
    best_sr = max(all_srs)
    best_ckpt = max(results, key=lambda k: results[k]["success_rate"])
    last_n_avg = float(np.mean(all_srs[-last_n:])) if len(all_srs) >= last_n else float(np.mean(all_srs))

    return {
        "best_sr": best_sr,
        "best_ckpt": best_ckpt,
        "last_n_avg": last_n_avg,
        "per_checkpoint": results,
    }


def evaluate_multi_seed(
    policy_class,
    ckpt_path: str,
    env_fn,
    seeds: list[int],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    proprio_mean: np.ndarray,
    proprio_std: np.ndarray,
    **eval_kwargs,
) -> dict:
    """Evaluate a single checkpoint across multiple seeds.

    Args:
        policy_class: Class with .load(ckpt_path).
        ckpt_path: Path to checkpoint.
        env_fn: Callable(seed) -> BaseManipulationEnv.
        seeds: List of random seeds.
        action_mean, action_std: Normalization stats.
        proprio_mean, proprio_std: Normalization stats.
        **eval_kwargs: Passed to evaluate_policy.

    Returns:
        Dict from aggregate_seeds plus 'ckpt_path'.
    """
    per_seed_outcomes = []

    for seed in seeds:
        env = env_fn(seed)
        policy = policy_class.load(ckpt_path)

        _, episode_results = evaluate_policy(
            policy,
            env,
            action_mean=action_mean,
            action_std=action_std,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            **eval_kwargs,
        )

        outcomes = [r["success"] for r in episode_results]
        per_seed_outcomes.append(outcomes)
        env.close()

    result = aggregate_seeds(per_seed_outcomes)
    result["ckpt_path"] = ckpt_path
    return result
