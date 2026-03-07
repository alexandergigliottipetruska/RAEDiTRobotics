"""Robosuite environment wrapper for evaluation.

Wraps a robosuite environment to implement the BaseManipulationEnv interface.
Extracts agentview + wrist images, resizes to 224x224, applies ImageNet
normalization, and pads to 4 camera slots.

Gripper action is continuous [-1, 1] and passed through as-is.
"""

import numpy as np
from PIL import Image

from data_pipeline.envs.base_env import BaseManipulationEnv

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Camera mapping: robomimic uses 2 of 4 slots
# slot 0: agentview, slot 1: empty, slot 2: empty, slot 3: wrist
_CAMERA_NAMES = {
    0: "agentview_image",
    3: "robot0_eye_in_hand_image",
}

# Task name -> robosuite env name
TASK_TO_ENV = {
    "lift": "Lift",
    "can": "PickPlaceCan",
    "square": "NutAssemblySquare",
    "tool_hang": "ToolHang",
}


def _process_image(img_hwc: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize + ImageNet normalize + HWC->CHW.

    Args:
        img_hwc: uint8 [H, W, 3] image from robosuite.

    Returns:
        float32 [3, H, W] ImageNet-normalized.
    """
    if img_hwc.shape[0] != target_size or img_hwc.shape[1] != target_size:
        img_hwc = np.array(
            Image.fromarray(img_hwc).resize(
                (target_size, target_size), Image.LANCZOS
            )
        )
    img_float = img_hwc.astype(np.float32) / 255.0
    normalized = (img_float - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.moveaxis(normalized, -1, -3)  # [3, H, W]


class RobomimicWrapper(BaseManipulationEnv):
    """Wraps a robosuite environment for policy evaluation.

    Args:
        task: Task name (lift, can, square, tool_hang).
        image_size: Target image size (default 224).
        seed: Random seed for env reset.
    """

    def __init__(
        self,
        task: str,
        image_size: int = 224,
        seed: int | None = None,
    ):
        import logging
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        # Suppress noisy robosuite warnings about unused controller components
        logging.getLogger("robosuite").setLevel(logging.ERROR)

        self._task = task
        self._image_size = image_size
        self._seed = seed

        controller_config = load_composite_controller_config(controller="BASIC")
        self._env = suite.make(
            env_name=TASK_TO_ENV[task],
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=84,
            camera_widths=84,
            use_object_obs=True,
            controller_configs=controller_config,
            ignore_done=True,
        )

        self._last_obs = None

    def reset(self) -> dict:
        self._last_obs = self._env.reset()
        return self._last_obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        info["success"] = bool(self._env._check_success())
        return obs, reward, done, info

    def get_multiview_images(self) -> np.ndarray:
        """Return [1, 4, 3, 224, 224] with slots 0,3 filled, 1,2 zeroed."""
        result = np.zeros(
            (1, 4, 3, self._image_size, self._image_size), dtype=np.float32
        )
        for slot, cam_key in _CAMERA_NAMES.items():
            img = self._last_obs[cam_key]  # [H, W, 3] uint8
            # robosuite returns images flipped vertically
            img = img[::-1].copy()
            result[0, slot] = _process_image(img, self._image_size)
        return result

    def get_proprio(self) -> np.ndarray:
        """Return [1, 9] proprio: eef_pos(3) + eef_quat(4) + gripper_qpos(2)."""
        obs = self._last_obs
        proprio = np.concatenate([
            obs["robot0_eef_pos"],       # [3]
            obs["robot0_eef_quat"],      # [4]
            obs["robot0_gripper_qpos"],  # [2]
        ]).astype(np.float32)
        return proprio.reshape(1, -1)

    def get_view_present(self) -> np.ndarray:
        return np.array([True, False, False, True])

    def close(self) -> None:
        self._env.close()

    @property
    def proprio_dim(self) -> int:
        return 9

    @property
    def num_cameras(self) -> int:
        return 4
