"""V3 evaluation: PolicyDiTv3 rollout with validated RobomimicWrapper.

Key differences from Stage3PolicyWrapper (V1/V2):
  - NO external ImageNet normalization — policy handles it internally via Stage1Bridge
  - Images passed as float [0,1] (NOT ImageNet-normalized)
  - rot6d→axis_angle conversion for 10D→7D actions
  - 50 episodes with seeds 100000+i (Chi's scheme)
  - Proper env seeding: env.seed(seed) before env.reset()

Usage:
    from training.eval_v3 import V3PolicyWrapper, evaluate_v3

    wrapper = V3PolicyWrapper(policy, ema_model=ema, device="cuda")
    success_rate, results = evaluate_v3(
        wrapper, norm_stats, num_episodes=50, task="lift",
    )
"""

import logging
import threading

import numpy as np
import torch

from data_pipeline.evaluation.rollout import evaluate_policy
from data_pipeline.utils.rotation import convert_actions_from_rot6d

log = logging.getLogger(__name__)


class V3PolicyWrapper:
    """Wraps PolicyDiTv3 for the evaluation harness (rollout.py).

    Unlike Stage3PolicyWrapper, this does NOT apply ImageNet normalization.
    Images are passed as float [0,1] to the policy, which handles resize +
    ImageNet norm internally via Stage1Bridge.

    Args:
        policy: PolicyDiTv3 instance.
        ema_model: diffusers.EMAModel instance (optional). If provided,
            copies EMA weights into policy for inference, then restores.
        device: Device for inference.
    """

    def __init__(self, policy, ema_model=None, device="cpu"):
        self.policy = policy
        self.ema_model = ema_model
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.eval()
        self._inference_lock = threading.Lock()  # for thread-safe parallel eval

    def predict(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        view_present: torch.Tensor,
    ) -> torch.Tensor:
        """Run inference for rollout.py evaluate_policy().

        Args:
            images:       (1, T_o, K, 3, H, W) float [0,1] or uint8 — from env wrapper
            proprio:      (1, T_o, D_prop) normalized proprio
            view_present: (1, K) bool

        Returns:
            actions: (T_p, ac_dim) normalized actions (no batch dim)
        """
        images = images.to(self.device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        # NOTE: NO ImageNet normalization here — policy handles it internally

        proprio = proprio.float().to(self.device)
        view_present = view_present.bool().to(self.device)

        obs = {
            "images_enc": images,
            "proprio": proprio,
            "view_present": view_present,
        }

        with self._inference_lock:
            if self.ema_model is not None:
                self.ema_model.store(self.policy.parameters())
                self.ema_model.copy_to(self.policy.parameters())
                try:
                    actions = self.policy.predict_action(obs)
                finally:
                    self.ema_model.restore(self.policy.parameters())
            else:
                actions = self.policy.predict_action(obs)

        # Remove batch dim: (1, T_p, ac_dim) → (T_p, ac_dim)
        return actions[0]


def evaluate_v3(
    wrapper: V3PolicyWrapper,
    norm_stats: dict,
    num_episodes: int = 50,
    task: str = "lift",
    max_steps: int = 400,
    seed_start: int = 100000,
    exec_horizon: int = 8,
    obs_horizon: int = 2,
    image_size: int = 84,
    control_freq: int = 20,
    use_rot6d: bool = True,
) -> tuple:
    """Run V3 evaluation with validated RobomimicWrapper.

    Args:
        wrapper:      V3PolicyWrapper wrapping PolicyDiTv3.
        norm_stats:   dict with 'actions' and 'proprio' stats (from load_norm_stats).
        num_episodes: Number of eval episodes.
        task:         Robosuite task name.
        max_steps:    Max steps per episode.
        seed_start:   First episode seed (Chi uses 100000).
        exec_horizon: Actions executed before re-planning.
        obs_horizon:  Past frames to condition on.
        image_size:   Env render size (84 for robomimic).
        control_freq: Env control frequency.
        use_rot6d:    Convert 10D rot6d → 7D axis-angle for env.

    Returns:
        (success_rate, per_episode_results)
    """
    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

    env = RobomimicWrapper(
        task=task,
        abs_action=True,
        image_size=image_size,
    )

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    success_rate, results = evaluate_policy(
        policy=wrapper,
        env=env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        norm_mode="minmax",
        action_min=action_stats["min"],
        action_max=action_stats["max"],
        proprio_min=proprio_stats["min"],
        proprio_max=proprio_stats["max"],
        exec_horizon=exec_horizon,
        obs_horizon=obs_horizon,
        rot6d=use_rot6d,
    )

    env.close()
    log.info("V3 eval: %.1f%% success (%d/%d episodes)",
             success_rate * 100, int(success_rate * num_episodes), num_episodes)

    return success_rate, results


def evaluate_v3_parallel(
    wrapper: V3PolicyWrapper,
    norm_stats: dict,
    num_episodes: int = 50,
    num_workers: int = 4,
    task: str = "lift",
    max_steps: int = 400,
    seed_start: int = 100000,
    exec_horizon: int = 8,
    obs_horizon: int = 2,
    image_size: int = 84,
    use_rot6d: bool = True,
) -> tuple:
    """Run V3 evaluation with parallel episode execution.

    Each thread gets its own RobomimicWrapper instance. GPU inference is
    serialized via a lock in V3PolicyWrapper, but env steps (MuJoCo physics)
    run truly in parallel since mj_step releases the GIL.

    Args:
        wrapper:      V3PolicyWrapper (thread-safe with inference lock).
        norm_stats:   dict with 'actions' and 'proprio' stats.
        num_episodes: Total episodes to run.
        num_workers:  Number of parallel threads.
        Other args:   Same as evaluate_v3.

    Returns:
        (success_rate, per_episode_results)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    def run_single_episode(ep_id):
        env = RobomimicWrapper(task=task, abs_action=True, image_size=image_size)
        env.seed(seed_start + ep_id)

        _, ep_results = evaluate_policy(
            policy=wrapper,
            env=env,
            num_episodes=1,
            max_steps=max_steps,
            norm_mode="minmax",
            action_min=action_stats["min"],
            action_max=action_stats["max"],
            proprio_min=proprio_stats["min"],
            proprio_max=proprio_stats["max"],
            exec_horizon=exec_horizon,
            obs_horizon=obs_horizon,
            rot6d=use_rot6d,
        )
        env.close()
        return ep_results[0]

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(run_single_episode, i): i for i in range(num_episodes)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                log.warning("Episode %d failed: %s", futures[future], e)
                results.append({"success": False, "steps": 0, "reward": 0.0})

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / max(len(results), 1)
    log.info("V3 parallel eval: %d/%d (%.1f%%) with %d workers",
             n_success, len(results), success_rate * 100, num_workers)

    return success_rate, results
