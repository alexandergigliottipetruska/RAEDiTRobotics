"""Robosuite environment wrapper for joint-space evaluation.

Uses robosuite's JOINT_POSITION controller instead of OSC_POSE.
Actions are 8D: [joint_positions(7), gripper_command(1)].
No IK or motion planning needed — joint targets are set directly.

Proprio is 9D: [joint_positions(7), gripper_qpos(2)].
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from data_pipeline.envs.base_env import BaseManipulationEnv

_CAMERA_NAMES = {
    0: "agentview_image",
    3: "robot0_eye_in_hand_image",
}

TASK_TO_ENV = {
    "lift": "Lift",
    "can": "PickPlaceCan",
    "square": "NutAssemblySquare",
    "tool_hang": "ToolHang",
}


def _process_image(img_hwc: np.ndarray, target_size: int = 224) -> np.ndarray:
    if img_hwc.shape[0] != target_size or img_hwc.shape[1] != target_size:
        img_hwc = np.array(
            Image.fromarray(img_hwc).resize(
                (target_size, target_size), Image.LANCZOS
            )
        )
    img_float = img_hwc.astype(np.float32) / 255.0
    return np.moveaxis(img_float, -1, -3)


class RobomimicJointWrapper(BaseManipulationEnv):
    """Wraps robosuite with JOINT_POSITION controller for joint-space eval.

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
        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        logging.getLogger("robosuite").setLevel(logging.ERROR)

        self._task = task
        self._image_size = image_size
        self._seed = seed

        controller_config = load_composite_controller_config(robot="panda")

        # Switch arm controller to JOINT_POSITION (absolute)
        right_cfg = controller_config["body_parts"]["right"]
        right_cfg["type"] = "JOINT_POSITION"
        right_cfg["input_type"] = "absolute"
        right_cfg["input_min"] = -10
        right_cfg["input_max"] = 10
        right_cfg["output_min"] = -10
        right_cfg["output_max"] = 10
        # No impedance/damping scaling needed — joint position control is direct
        right_cfg["kp"] = 150
        right_cfg["damping_ratio"] = 1.0

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

    def seed(self, seed: int) -> None:
        self._seed = seed

    def reset(self) -> dict:
        if self._seed is not None:
            np.random.seed(self._seed)
            self._env.seed = self._seed
            self._env.rng = np.random.default_rng(self._seed)
        self._last_obs = self._env.reset()
        return self._last_obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Step with 8D action: [joint_positions(7), gripper(1)]."""
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        info["success"] = bool(self._env._check_success())
        return obs, reward, done, info

    def get_multiview_images(self) -> np.ndarray:
        result = np.zeros(
            (1, 4, 3, self._image_size, self._image_size), dtype=np.float32
        )
        for slot, cam_key in _CAMERA_NAMES.items():
            img = self._last_obs[cam_key][::-1].copy()
            result[0, slot] = _process_image(img, self._image_size)
        return result

    def get_proprio(self) -> np.ndarray:
        """Return [1, 9] proprio: joint_pos(7) + gripper_qpos(2)."""
        obs = self._last_obs
        proprio = np.concatenate([
            obs["robot0_joint_pos"],       # [7]
            obs["robot0_gripper_qpos"],    # [2]
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
