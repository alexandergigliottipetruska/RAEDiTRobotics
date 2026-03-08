"""RLBench environment wrapper for evaluation.

Wraps an RLBench Environment to implement the BaseManipulationEnv interface.
Extracts 4 camera views (front, left_shoulder, right_shoulder, wrist),
resizes to 224x224, applies ImageNet normalization.

The policy outputs 8D absolute EE poses [position(3), quaternion_xyzw(4), gripper(1)].
This wrapper uses OMPL motion planning (EndEffectorPoseViaPlanning) to reach
each target pose, matching the approach used by RVT-2 and Chain-of-Action.

OMPL plans a collision-free path from the current pose to the target and
executes it step-by-step through the sim. This handles large pose jumps
and is the standard action mode used by all major RLBench papers
(PerAct, RVT, RVT-2, CoA, Act3D).

Gripper is thresholded at 0.5 to binary {0.0, 1.0}.

Requires CoppeliaSim (WSL2 or remote Linux).
"""

import numpy as np
from PIL import Image

from data_pipeline.envs.base_env import BaseManipulationEnv

# ImageNet normalization constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Camera mapping: RLBench uses all 4 slots
# slot 0: front, slot 1: left_shoulder, slot 2: right_shoulder, slot 3: wrist
_CAMERA_ATTRS = {
    0: "front_rgb",
    1: "left_shoulder_rgb",
    2: "right_shoulder_rgb",
    3: "wrist_rgb",
}

# Task name -> RLBench task class name
TASK_CLASS_MAP = {
    "close_jar": "CloseJar",
    "open_drawer": "OpenDrawer",
    "slide_block_to_color_target": "SlideBlockToColorTarget",
    "put_item_in_drawer": "PutItemInDrawer",
    "stack_cups": "StackCups",
    "place_shape_in_shape_sorter": "PlaceShapeInShapeSorter",
    "meat_off_grill": "MeatOffGrill",
    "turn_tap": "TurnTap",
    "push_buttons": "PushButtons",
    "reach_and_drag": "ReachAndDrag",
}


def _process_image(img_hwc: np.ndarray, target_size: int = 224) -> np.ndarray:
    """Resize + ImageNet normalize + HWC->CHW.

    Args:
        img_hwc: uint8 [H, W, 3] image from RLBench.

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


class RLBenchWrapper(BaseManipulationEnv):
    """Wraps an RLBench environment for policy evaluation.

    Passes absolute EE pose actions to the sim via OMPL motion planning,
    matching the RVT-2 / CoA evaluation approach.

    Args:
        task_name: Task name (must be in TASK_CLASS_MAP).
        image_size: Target image size (default 224).
        headless: Run CoppeliaSim headless (default True).
        cameras: Enable vision sensors (default True). Set False for
                 headless WSL2 to avoid OpenGL segfault.
    """

    def __init__(
        self,
        task_name: str,
        image_size: int = 224,
        headless: bool = True,
        cameras: bool = True,
    ):
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig

        self._task_name = task_name
        self._image_size = image_size
        self._cameras = cameras

        if task_name not in TASK_CLASS_MAP:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Supported: {list(TASK_CLASS_MAP.keys())}"
            )

        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(cameras)

        # Stock OMPL motion planning — matches replay_rlbench.py and RVT-2.
        # EndEffectorPoseViaPlanning defaults: absolute_mode=True, ignore_collisions=True
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )

        self._env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=headless,
        )
        self._env.launch()

        import rlbench.tasks
        task_cls = getattr(rlbench.tasks, TASK_CLASS_MAP[task_name])
        self._task = self._env.get_task(task_cls)

        self._last_obs = None

    def reset(self) -> dict:
        descriptions, obs = self._task.reset()
        self._last_obs = obs
        return {"descriptions": descriptions}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """Execute one absolute EE pose action via OMPL motion planning.

        Args:
            action: [8] float32 — [position(3), quaternion_xyzw(4), gripper(1)]
        """
        position = action[:3].astype(np.float64)
        quat_xyzw = action[3:7].astype(np.float64)
        gripper = 1.0 if float(action[7]) > 0.5 else 0.0

        # Build action: [x, y, z, qx, qy, qz, qw, gripper]
        abs_action = np.concatenate([position, quat_xyzw, [gripper]])

        try:
            obs, reward, terminate = self._task.step(abs_action)
        except Exception:
            # Planning can fail for unreachable poses.
            # Treat as episode termination with no success.
            return {}, 0.0, True, {"success": False}
        self._last_obs = obs

        success = reward == 1.0
        return {}, float(reward), bool(terminate), {"success": success}

    def get_multiview_images(self) -> np.ndarray:
        """Return [1, 4, 3, 224, 224] with all 4 camera slots filled."""
        result = np.zeros(
            (1, 4, 3, self._image_size, self._image_size), dtype=np.float32
        )
        for slot, cam_attr in _CAMERA_ATTRS.items():
            img = getattr(self._last_obs, cam_attr, None)
            if img is not None:
                result[0, slot] = _process_image(img, self._image_size)
        return result

    def get_proprio(self) -> np.ndarray:
        """Return [1, 8] proprio: joint_positions(7) + gripper_open(1)."""
        obs = self._last_obs
        proprio = np.concatenate([
            np.array(obs.joint_positions, dtype=np.float32),  # [7]
            [float(obs.gripper_open)],                         # [1]
        ])
        return proprio.reshape(1, -1)

    def get_view_present(self) -> np.ndarray:
        return np.array([True, True, True, True])

    def close(self) -> None:
        self._env.shutdown()

    @property
    def proprio_dim(self) -> int:
        return 8

    @property
    def num_cameras(self) -> int:
        return 4
