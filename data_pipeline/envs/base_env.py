"""Abstract environment interface for evaluation.

All benchmark wrappers (robomimic, RLBench, ManiSkill3) implement this
interface so the rollout loop is benchmark-agnostic.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseManipulationEnv(ABC):
    """Common interface for all benchmark environment wrappers.

    Image format: [1, K, 3, H, W] float32, ImageNet-normalized.
    Proprio format: [1, D_prop] float32, raw (normalization done in rollout).
    """

    @abstractmethod
    def reset(self) -> dict:
        """Reset the environment and return initial observation dict."""
        ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute one action.

        Args:
            action: [7] float32 array — denormalized delta-EE action.

        Returns:
            obs: Observation dict.
            reward: Scalar reward.
            done: Whether episode is finished.
            info: Dict with at least 'success' key (bool).
        """
        ...

    @abstractmethod
    def get_multiview_images(self) -> np.ndarray:
        """Return current multi-view images.

        Returns:
            [1, K, 3, H, W] float32 array, ImageNet-normalized.
            K = NUM_CAMERA_SLOTS (4). Unused slots are zero-filled.
        """
        ...

    @abstractmethod
    def get_proprio(self) -> np.ndarray:
        """Return current proprioceptive state.

        Returns:
            [1, D_prop] float32 array, raw values (not normalized).
        """
        ...

    @abstractmethod
    def get_view_present(self) -> np.ndarray:
        """Return camera presence flags.

        Returns:
            [K] bool array. True for slots with real camera data.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Release environment resources."""
        ...

    @property
    @abstractmethod
    def proprio_dim(self) -> int:
        """Dimension of the proprioceptive state vector."""
        ...

    @property
    @abstractmethod
    def num_cameras(self) -> int:
        """Number of camera slots (always 4)."""
        ...
