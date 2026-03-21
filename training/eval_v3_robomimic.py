"""V3 evaluation using robomimic's infrastructure (matching Chi's eval pipeline).

Uses EnvUtils.create_env_from_metadata() to create environments with the exact
controller config stored in the HDF5, plus Chi's RobomimicImageWrapper and
MultiStepWrapper for obs stacking and multi-step action execution.

Key differences from eval_v3.py:
  - Environment created from HDF5 env_meta (not fresh suite.make())
  - Chi's RobomimicImageWrapper for image observations
  - Chi's MultiStepWrapper for obs stacking + action execution
  - Seeding via np.random.seed (Chi's approach)
  - Success measured via max reward per episode (Chi's metric)

Usage:
    from training.eval_v3_robomimic import evaluate_v3_robomimic

    success_rate, results = evaluate_v3_robomimic(
        policy=policy, ema_model=ema,
        hdf5_path="data/robomimic/lift/ph_abs_v15.hdf5",
        num_episodes=50, device="cuda",
    )
"""

import collections
import logging
from collections import defaultdict

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from data_pipeline.envs.robomimic_image_wrapper import RobomimicImageWrapper
from data_pipeline.envs.multistep_wrapper import MultiStepWrapper
from data_pipeline.utils.rotation import convert_actions_from_rot6d

log = logging.getLogger(__name__)

# Shape meta matching Chi's lift_image_abs.yaml
LIFT_SHAPE_META = {
    'obs': {
        'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
        'robot0_eef_pos': {'shape': [3]},
        'robot0_eef_quat': {'shape': [4]},
        'robot0_gripper_qpos': {'shape': [2]},
    },
    'action': {'shape': [10]},
}


def create_robomimic_env(hdf5_path, shape_meta=None, abs_action=True,
                         n_obs_steps=2, n_action_steps=8, max_steps=400):
    """Create environment using robomimic infrastructure (matching Chi).

    Args:
        hdf5_path: Path to HDF5 dataset (contains env_args in metadata).
        shape_meta: Observation/action shape metadata dict. Defaults to Lift.
        abs_action: If True, set control_delta=False for absolute actions.
        n_obs_steps: Number of observation frames to stack.
        n_action_steps: Number of actions to execute per step.
        max_steps: Maximum episode length.

    Returns:
        MultiStepWrapper wrapping RobomimicImageWrapper wrapping EnvRobosuite.
    """
    if shape_meta is None:
        shape_meta = LIFT_SHAPE_META

    # Initialize obs modality mapping (required by robomimic)
    modality_mapping = defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    # Read env_meta from HDF5 (exact controller config used during data collection)
    env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
    env_meta['env_kwargs']['use_object_obs'] = False

    if abs_action:
        # The env_args from data collection has delta-mode controller config
        # (input_min/max=[-1,1], control_delta=True). For absolute actions we
        # need to override these — matching our RobomimicWrapper's abs config.
        ctrl = env_meta['env_kwargs']['controller_configs']
        if 'body_parts' in ctrl:
            # robosuite 1.5 composite controller format
            for part in ctrl['body_parts'].values():
                part['control_delta'] = False
                part['input_type'] = 'absolute'
                part['input_ref_frame'] = 'world'
                part['input_min'] = -10
                part['input_max'] = 10
        else:
            # robosuite 1.2 flat controller format
            ctrl['control_delta'] = False
            ctrl['input_min'] = -10
            ctrl['input_max'] = 10

    # Create environment via robomimic (uses env_meta's controller config)
    robomimic_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    # NOTE: Chi uses hard_reset=False with AsyncVectorEnv (one episode per worker).
    # For sequential eval, keep hard_reset=True to prevent renderer corruption.
    # robomimic_env.env.hard_reset = False

    # Wrap with Chi's wrappers
    env = MultiStepWrapper(
        RobomimicImageWrapper(
            env=robomimic_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key='agentview_image',
        ),
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps,
    )

    return env


def evaluate_v3_robomimic(
    policy,
    ema_model=None,
    hdf5_path: str = "",
    norm_stats: dict = None,
    num_episodes: int = 50,
    seed_start: int = 100000,
    max_steps: int = 400,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    shape_meta: dict = None,
    use_rot6d: bool = True,
    device: str = "cuda",
    save_video: bool = False,
    video_dir: str = "",
) -> tuple:
    """Run V3 evaluation using robomimic's infrastructure.

    This matches Chi's eval pipeline exactly:
    1. Environment created from HDF5 env_meta
    2. RobomimicImageWrapper for image obs (float32 CHW [0,1])
    3. MultiStepWrapper for obs stacking + multi-step action execution
    4. np.random.seed seeding (Chi's approach)
    5. Success = max reward per episode

    Args:
        policy:        PolicyDiTv3 instance.
        ema_model:     diffusers EMAModel (optional).
        hdf5_path:     Path to HDF5 dataset with env_args.
        norm_stats:    Dict with 'actions' and 'proprio' stats.
        num_episodes:  Number of test episodes.
        seed_start:    First episode seed (Chi uses 100000).
        max_steps:     Max steps per episode.
        n_obs_steps:   Observation stacking window.
        n_action_steps: Actions executed per policy call.
        shape_meta:    Obs/action shape metadata. Defaults to Lift.
        use_rot6d:     If True, convert 10D rot6d → 7D axis-angle.
        device:        CUDA device.
        save_video:    If True, save MP4 videos of rollouts.
        video_dir:     Directory for video files. Required if save_video=True.

    Returns:
        (success_rate, per_episode_results)
    """
    if shape_meta is None:
        shape_meta = LIFT_SHAPE_META

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # Create env
    env = create_robomimic_env(
        hdf5_path=hdf5_path,
        shape_meta=shape_meta,
        abs_action=True,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
    )

    # Video setup
    if save_video:
        import os
        os.makedirs(video_dir, exist_ok=True)
        try:
            import imageio
        except ImportError:
            log.warning("imageio not installed, disabling video recording")
            save_video = False

    # Norm stats for action denormalization
    action_min = norm_stats["actions"]["min"]
    action_max = norm_stats["actions"]["max"]

    all_rewards = []
    results = {}

    for ep_idx in range(num_episodes):
        seed = seed_start + ep_idx

        # Chi's seeding: set seed on inner wrapper, then reset
        env.env.seed(seed)
        obs = env.reset()

        done = False
        episode_rewards = []
        frames = []  # for video recording

        while not done:
            # obs from MultiStepWrapper + RobomimicImageWrapper:
            #   'agentview_image': (n_obs_steps, H, W, C) uint8 [0,255] HWC
            #   'robot0_eye_in_hand_image': (n_obs_steps, H, W, C) uint8 HWC
            #   'robot0_eef_pos': (n_obs_steps, 3)
            #   'robot0_eef_quat': (n_obs_steps, 4)
            #   'robot0_gripper_qpos': (n_obs_steps, 2)

            # Convert HWC uint8 → CHW float32 [0,1]
            agentview_hwc = obs['agentview_image']          # (T, H, W, C) uint8
            eye_in_hand_hwc = obs['robot0_eye_in_hand_image']  # (T, H, W, C) uint8
            agentview = np.transpose(agentview_hwc, (0, 3, 1, 2)).astype(np.float32) / 255.0
            eye_in_hand = np.transpose(eye_in_hand_hwc, (0, 3, 1, 2)).astype(np.float32) / 255.0

            T_obs_actual = agentview.shape[0]

            # Our format: (T_obs, K=4, 3, H, W) with slots 0=agentview, 3=eye_in_hand
            K = 4
            H, W = agentview.shape[-2], agentview.shape[-1]
            images = np.zeros((T_obs_actual, K, 3, H, W), dtype=np.float32)
            images[:, 0] = agentview
            images[:, 3] = eye_in_hand

            # Proprio: concat eef_pos(3) + eef_quat(4) + gripper_qpos(2) = 9D
            proprio = np.concatenate([
                obs['robot0_eef_pos'],
                obs['robot0_eef_quat'],
                obs['robot0_gripper_qpos'],
            ], axis=-1)  # (T_obs, 9)

            # Normalize proprio (minmax)
            proprio_min = norm_stats["proprio"]["min"]
            proprio_max = norm_stats["proprio"]["max"]
            p_range = proprio_max - proprio_min
            p_range = np.where(p_range < 1e-6, 1.0, p_range)
            proprio_norm = 2.0 * (proprio - proprio_min) / p_range - 1.0

            view_present = np.array([True, False, False, True])  # agentview + eye_in_hand

            # To tensors, add batch dim
            images_t = torch.from_numpy(images).unsqueeze(0).to(device)     # (1, T, K, 3, H, W)
            proprio_t = torch.from_numpy(proprio_norm).float().unsqueeze(0).to(device)  # (1, T, 9)
            view_present_t = torch.from_numpy(view_present).unsqueeze(0).to(device)     # (1, K)

            obs_dict = {
                "images_enc": images_t,
                "proprio": proprio_t,
                "view_present": view_present_t,
            }

            # Run policy with EMA
            with torch.no_grad():
                if ema_model is not None:
                    ema_model.store(policy.parameters())
                    ema_model.copy_to(policy.parameters())
                    try:
                        actions_norm = policy.predict_action(obs_dict)
                    finally:
                        ema_model.restore(policy.parameters())
                else:
                    actions_norm = policy.predict_action(obs_dict)

            # actions_norm: (1, T_pred, 10) normalized [-1, 1]
            actions_norm = actions_norm[0].cpu().numpy()  # (T_pred, 10)

            # Denormalize actions (minmax)
            a_range = action_max - action_min
            a_range = np.where(np.abs(a_range) < 1e-6, 1.0, a_range)
            actions_raw = (actions_norm + 1.0) / 2.0 * a_range + action_min  # (T_pred, 10)

            # Convert rot6d → axis_angle if needed
            if use_rot6d:
                actions_7d = convert_actions_from_rot6d(actions_raw)  # (T_pred, 7)
            else:
                actions_7d = actions_raw

            # Extract n_action_steps actions for MultiStepWrapper
            # MultiStepWrapper.step() expects (n_action_steps, action_dim)
            exec_actions = actions_7d[:n_action_steps]  # (8, 7)

            obs, reward, done, info = env.step(exec_actions)
            episode_rewards.append(reward)

            # Capture frame for video (last agentview from obs stack)
            if save_video:
                frame = obs['agentview_image'][-1]  # (H, W, C) uint8
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                frames.append(frame)

        max_reward = np.max(episode_rewards) if episode_rewards else 0.0
        success = max_reward > 0.5  # robomimic Lift: reward=1.0 on success

        # Save video
        if save_video and frames:
            tag = "success" if success else "fail"
            video_path = os.path.join(video_dir, f"ep{ep_idx:03d}_seed{seed}_{tag}.mp4")
            imageio.mimwrite(video_path, frames, fps=10)

        results[ep_idx] = {
            "seed": seed,
            "success": bool(success),
            "max_reward": float(max_reward),
            "steps": len(episode_rewards) * n_action_steps,
        }

        if (ep_idx + 1) % 10 == 0 or ep_idx == 0:
            running_sr = sum(r["success"] for r in results.values()) / len(results)
            log.info("Eval ep %d/%d: seed=%d success=%s (running %.0f%%)",
                     ep_idx + 1, num_episodes, seed, success, running_sr * 100)

    env.close()

    success_rate = sum(r["success"] for r in results.values()) / num_episodes
    log.info("Robomimic eval: %.1f%% success (%d/%d episodes)",
             success_rate * 100, int(success_rate * num_episodes), num_episodes)

    return success_rate, results
