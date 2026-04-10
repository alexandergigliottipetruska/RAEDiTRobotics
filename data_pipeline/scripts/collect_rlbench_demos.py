"""Collect RLBench demos with joint_position_action (stepjam RLBench).

Generates demos in PerAct's all_variations layout:
  {save_path}/{split}/{task}/all_variations/episodes/episode{N}/
    low_dim_obs.pkl          (Demo with joint_position_action in obs.misc)
    variation_number.pkl     (int)
    variation_descriptions.pkl
    front_rgb/0.png ...
    left_shoulder_rgb/0.png ...
    right_shoulder_rgb/0.png ...
    wrist_rgb/0.png ...

Requires:
  - stepjam RLBench (stores joint_position_action + variation_index in obs.misc)
  - CoppeliaSim + Xvfb for camera rendering (DISPLAY=:99)

Usage:
    # Start Xvfb first:  Xvfb :99 -screen 0 1024x768x24 &
    DISPLAY=:99 PYTHONPATH=. python scripts/collect_rlbench_demos.py \
        --tasks close_jar open_drawer \
        --train_episodes 100 --valid_episodes 25 \
        --save_path data/raw/rlbench
"""

import argparse
import logging
import os
import pickle

import numpy as np
from PIL import Image
from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import (
    LEFT_SHOULDER_RGB_FOLDER, LEFT_SHOULDER_DEPTH_FOLDER, LEFT_SHOULDER_MASK_FOLDER,
    RIGHT_SHOULDER_RGB_FOLDER, RIGHT_SHOULDER_DEPTH_FOLDER, RIGHT_SHOULDER_MASK_FOLDER,
    OVERHEAD_RGB_FOLDER, OVERHEAD_DEPTH_FOLDER, OVERHEAD_MASK_FOLDER,
    WRIST_RGB_FOLDER, WRIST_DEPTH_FOLDER, WRIST_MASK_FOLDER,
    FRONT_RGB_FOLDER, FRONT_DEPTH_FOLDER, FRONT_MASK_FOLDER,
    IMAGE_FORMAT, DEPTH_SCALE, LOW_DIM_PICKLE,
)
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

VARIATION_NUMBER_FILE = "variation_number.pkl"
VARIATION_DESCRIPTIONS_FILE = "variation_descriptions.pkl"

# Camera folders for saving images
CAMERA_FOLDERS = {
    "left_shoulder_rgb": LEFT_SHOULDER_RGB_FOLDER,
    "left_shoulder_depth": LEFT_SHOULDER_DEPTH_FOLDER,
    "left_shoulder_mask": LEFT_SHOULDER_MASK_FOLDER,
    "right_shoulder_rgb": RIGHT_SHOULDER_RGB_FOLDER,
    "right_shoulder_depth": RIGHT_SHOULDER_DEPTH_FOLDER,
    "right_shoulder_mask": RIGHT_SHOULDER_MASK_FOLDER,
    "overhead_rgb": OVERHEAD_RGB_FOLDER,
    "overhead_depth": OVERHEAD_DEPTH_FOLDER,
    "overhead_mask": OVERHEAD_MASK_FOLDER,
    "wrist_rgb": WRIST_RGB_FOLDER,
    "wrist_depth": WRIST_DEPTH_FOLDER,
    "wrist_mask": WRIST_MASK_FOLDER,
    "front_rgb": FRONT_RGB_FOLDER,
    "front_depth": FRONT_DEPTH_FOLDER,
    "front_mask": FRONT_MASK_FOLDER,
}


def save_demo(demo, episode_path, variation, descriptions):
    """Save demo in PerAct format (images + pickle)."""
    os.makedirs(episode_path, exist_ok=True)

    # Create camera subdirectories
    for folder in CAMERA_FOLDERS.values():
        os.makedirs(os.path.join(episode_path, folder), exist_ok=True)

    for i, obs in enumerate(demo):
        # Save RGB images
        for attr, folder in [
            ("left_shoulder_rgb", LEFT_SHOULDER_RGB_FOLDER),
            ("right_shoulder_rgb", RIGHT_SHOULDER_RGB_FOLDER),
            ("overhead_rgb", OVERHEAD_RGB_FOLDER),
            ("wrist_rgb", WRIST_RGB_FOLDER),
            ("front_rgb", FRONT_RGB_FOLDER),
        ]:
            img_data = getattr(obs, attr, None)
            if img_data is not None:
                Image.fromarray(img_data).save(
                    os.path.join(episode_path, folder, IMAGE_FORMAT % i))

        # Save depth images
        for attr, folder in [
            ("left_shoulder_depth", LEFT_SHOULDER_DEPTH_FOLDER),
            ("right_shoulder_depth", RIGHT_SHOULDER_DEPTH_FOLDER),
            ("overhead_depth", OVERHEAD_DEPTH_FOLDER),
            ("wrist_depth", WRIST_DEPTH_FOLDER),
            ("front_depth", FRONT_DEPTH_FOLDER),
        ]:
            depth_data = getattr(obs, attr, None)
            if depth_data is not None:
                depth_img = utils.float_array_to_rgb_image(
                    depth_data, scale_factor=DEPTH_SCALE)
                depth_img.save(
                    os.path.join(episode_path, folder, IMAGE_FORMAT % i))

        # Save mask images
        for attr, folder in [
            ("left_shoulder_mask", LEFT_SHOULDER_MASK_FOLDER),
            ("right_shoulder_mask", RIGHT_SHOULDER_MASK_FOLDER),
            ("overhead_mask", OVERHEAD_MASK_FOLDER),
            ("wrist_mask", WRIST_MASK_FOLDER),
            ("front_mask", FRONT_MASK_FOLDER),
        ]:
            mask_data = getattr(obs, attr, None)
            if mask_data is not None:
                Image.fromarray((mask_data * 255).astype(np.uint8)).save(
                    os.path.join(episode_path, folder, IMAGE_FORMAT % i))

        # Null out image data for pickle (saves disk space)
        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    # Save low-dim pickle (Demo object with joint_position_action in misc)
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)

    # Save variation number
    with open(os.path.join(episode_path, VARIATION_NUMBER_FILE), "wb") as f:
        pickle.dump(variation, f)

    # Save descriptions
    with open(os.path.join(episode_path, VARIATION_DESCRIPTIONS_FILE), "wb") as f:
        pickle.dump(descriptions, f)


def collect_split(task_names, save_root, num_episodes, img_size, renderer):
    """Collect demos for one split (train or valid)."""
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    for cam in [obs_config.left_shoulder_camera, obs_config.right_shoulder_camera,
                obs_config.overhead_camera, obs_config.wrist_camera,
                obs_config.front_camera]:
        cam.image_size = img_size
        cam.depth_in_meters = False
        cam.masks_as_one_channel = False
        if renderer == "opengl":
            cam.render_mode = RenderMode.OPENGL
        elif renderer == "opengl3":
            cam.render_mode = RenderMode.OPENGL3

    env = Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=obs_config,
        headless=True,
    )
    env.launch()

    for task_name in task_names:
        task_cls = task_file_to_task_class(task_name)
        task_env = env.get_task(task_cls)
        num_variations = task_env.variation_count()

        episodes_path = os.path.join(
            save_root, task_name, "all_variations", "episodes")
        os.makedirs(episodes_path, exist_ok=True)

        log.info("Task: %s (%d variations, collecting %d episodes)",
                 task_name, num_variations, num_episodes)

        collected = 0
        while collected < num_episodes:
            # Random variation per episode (PerAct style)
            variation = np.random.randint(num_variations)
            task_env.set_variation(variation)
            descriptions, obs = task_env.reset()

            attempts = 10
            demo = None
            while attempts > 0:
                try:
                    demo, = task_env.get_demos(amount=1, live_demos=True)
                    break
                except Exception as e:
                    attempts -= 1
                    if attempts == 0:
                        log.warning("  Failed episode %d after 10 attempts: %s",
                                    collected, e)

            if demo is None:
                continue

            episode_path = os.path.join(episodes_path, f"episode{collected}")
            save_demo(demo, episode_path, variation, descriptions)

            # Verify joint_position_action is present
            if collected == 0:
                obs0 = demo._observations[1] if len(demo) > 1 else demo._observations[0]
                has_jpa = "joint_position_action" in obs0.misc
                has_vi = "variation_index" in obs0.misc
                log.info("  Verification: joint_position_action=%s variation_index=%s",
                         has_jpa, has_vi)
                assert has_jpa, "joint_position_action missing! Wrong RLBench version?"

            collected += 1
            if collected % 10 == 0 or collected == num_episodes:
                log.info("  %s: %d/%d episodes collected",
                         task_name, collected, num_episodes)

    env.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Collect RLBench demos with joint_position_action")
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="Task names to collect")
    parser.add_argument("--train_episodes", type=int, default=100)
    parser.add_argument("--valid_episodes", type=int, default=25)
    parser.add_argument("--save_path", type=str, default="data/raw/rlbench")
    parser.add_argument("--image_size", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--renderer", choices=["opengl", "opengl3"],
                        default="opengl",
                        help="opengl works with Xvfb; opengl3 needs hardware GL")
    args = parser.parse_args()

    img_size = args.image_size

    # Collect train split
    if args.train_episodes > 0:
        log.info("=== Collecting TRAIN split (%d episodes per task) ===",
                 args.train_episodes)
        collect_split(
            args.tasks,
            os.path.join(args.save_path, "train"),
            args.train_episodes,
            img_size, args.renderer,
        )

    # Collect valid split
    if args.valid_episodes > 0:
        log.info("=== Collecting VALID split (%d episodes per task) ===",
                 args.valid_episodes)
        collect_split(
            args.tasks,
            os.path.join(args.save_path, "valid"),
            args.valid_episodes,
            img_size, args.renderer,
        )

    log.info("Done! Demos saved to %s", args.save_path)


if __name__ == "__main__":
    main()
