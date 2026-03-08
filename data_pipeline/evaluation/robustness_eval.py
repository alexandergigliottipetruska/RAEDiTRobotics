"""Camera dropout robustness evaluation.

Tests whether the policy degrades gracefully when cameras are unavailable.
For each configuration, dropped views are zeroed out and view_present
is set to False for those slots. The full evaluation suite runs per config.

Most meaningful on RLBench (4 real cameras). For robomimic (2 real cameras,
slots 1-2 already padded), additional dropout has limited value.
"""

import numpy as np

from data_pipeline.evaluation.rollout import evaluate_policy
from data_pipeline.evaluation.metrics import wilson_ci


# Camera slot semantics:
#   0: front (agentview for robomimic)
#   1: left_shoulder (padded for robomimic)
#   2: right_shoulder (padded for robomimic)
#   3: wrist (robot0_eye_in_hand for robomimic)

DROPOUT_CONFIGS = [
    {"drop": [],       "label": "all_4"},
    {"drop": [0],      "label": "no_front"},
    {"drop": [3],      "label": "no_wrist"},
    {"drop": [1],      "label": "no_left"},
    {"drop": [0, 3],   "label": "no_front_wrist"},
    {"drop": [1, 2],   "label": "no_shoulders"},
    {"drop": [0, 1, 3], "label": "only_right"},
]


class CameraDropoutEnvWrapper:
    """Wraps a BaseManipulationEnv to zero out dropped camera views.

    Modifies get_multiview_images() to zero dropped slots and
    get_view_present() to mark them as False.
    """

    def __init__(self, env, drop_slots: list[int]):
        self._env = env
        self._drop_slots = drop_slots

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)

    def get_multiview_images(self) -> np.ndarray:
        images = self._env.get_multiview_images()  # [1, K, 3, H, W]
        for slot in self._drop_slots:
            images[:, slot] = 0.0
        return images

    def get_proprio(self) -> np.ndarray:
        return self._env.get_proprio()

    def get_view_present(self) -> np.ndarray:
        vp = self._env.get_view_present().copy()
        for slot in self._drop_slots:
            vp[slot] = False
        return vp

    def close(self):
        self._env.close()


def evaluate_robustness(
    policy,
    env,
    configs: list[dict] | None = None,
    norm_mode: str = "zscore",
    action_mean: np.ndarray | None = None,
    action_std: np.ndarray | None = None,
    action_min: np.ndarray | None = None,
    action_max: np.ndarray | None = None,
    proprio_mean: np.ndarray | None = None,
    proprio_std: np.ndarray | None = None,
    proprio_min: np.ndarray | None = None,
    proprio_max: np.ndarray | None = None,
    **eval_kwargs,
) -> dict:
    """Run evaluation across all camera dropout configurations.

    Args:
        policy: Model with .predict().
        env: BaseManipulationEnv (unwrapped).
        configs: List of dropout configs. Defaults to DROPOUT_CONFIGS.
        norm_mode: "zscore" or "minmax".
        action_mean, action_std: For zscore denormalization.
        action_min, action_max: For minmax denormalization.
        proprio_mean, proprio_std: For zscore proprio normalization.
        proprio_min, proprio_max: For minmax proprio normalization.
        **eval_kwargs: Passed to evaluate_policy.

    Returns:
        Dict mapping config label -> {success_rate, ci_lower, ci_upper, episodes}.
    """
    if configs is None:
        configs = DROPOUT_CONFIGS

    results = {}

    for cfg in configs:
        drop_slots = cfg["drop"]
        label = cfg["label"]

        wrapped_env = CameraDropoutEnvWrapper(env, drop_slots)

        sr, episodes = evaluate_policy(
            policy,
            wrapped_env,
            norm_mode=norm_mode,
            action_mean=action_mean,
            action_std=action_std,
            action_min=action_min,
            action_max=action_max,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            proprio_min=proprio_min,
            proprio_max=proprio_max,
            **eval_kwargs,
        )

        n_total = len(episodes)
        n_successes = sum(1 for e in episodes if e["success"])
        ci_lo, ci_hi = wilson_ci(n_successes, n_total)

        results[label] = {
            "success_rate": sr,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "dropped_slots": drop_slots,
            "episodes": episodes,
        }

    return results
