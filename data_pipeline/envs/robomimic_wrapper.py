"""Robosuite environment wrapper for evaluation.

Wraps a robosuite environment to implement the BaseManipulationEnv interface.
Extracts agentview + wrist images, resizes to 224x224, and pads to 4 camera slots.

Gripper action is continuous [-1, 1] and passed through as-is.
"""

import numpy as np
from PIL import Image

from data_pipeline.envs.base_env import BaseManipulationEnv

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
    """Resize + HWC->CHW. Returns float32 [0,1] (no ImageNet normalization).

    Args:
        img_hwc: uint8 [H, W, 3] image from robosuite.

    Returns:
        float32 [3, H, W] in [0, 1] range.
    """
    if img_hwc.shape[0] != target_size or img_hwc.shape[1] != target_size:
        img_hwc = np.array(
            Image.fromarray(img_hwc).resize(
                (target_size, target_size), Image.LANCZOS
            )
        )
    img_float = img_hwc.astype(np.float32) / 255.0
    return np.moveaxis(img_float, -1, -3)  # [3, H, W]


class RobomimicWrapper(BaseManipulationEnv):
    """Wraps a robosuite environment for policy evaluation.

    Args:
        task: Task name (lift, can, square, tool_hang).
        image_size: Target image size (default 224).
        seed: Random seed for env reset.
        abs_action: If True, use absolute EE pose actions instead of deltas.
    """

    def __init__(
        self,
        task: str,
        image_size: int = 224,
        seed: int | None = None,
        abs_action: bool = False,
    ):
        import logging
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        # Suppress noisy robosuite warnings about unused controller components
        logging.getLogger("robosuite").setLevel(logging.ERROR)

        self._task = task
        self._image_size = image_size
        self._seed = seed
        self._abs_action = abs_action

        controller_config = load_composite_controller_config(controller="BASIC")

        if abs_action:
            # For absolute actions: use the composite config but set
            # input_type to "absolute" instead of "delta".
            # Also remove input/output scaling to pass raw EE poses through.
            for part in controller_config.get("body_parts", {}).values():
                if part.get("type") == "OSC_POSE":
                    part["input_type"] = "absolute"
                    # In absolute mode, input is raw EE pose, not [-1,1] scaled.
                    # Remove the clipping that would limit to ±0.05m.
                    part["input_max"] = 1
                    part["input_min"] = -1
                    part["output_max"] = [1.0, 1.0, 1.0, 3.15, 3.15, 3.15]
                    part["output_min"] = [-1.0, -1.0, -1.0, -3.15, -3.15, -3.15]

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
        """Return [1, 4, 3, 224, 224] float32 [0,1] with slots 0,3 filled, 1,2 zeroed."""
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
