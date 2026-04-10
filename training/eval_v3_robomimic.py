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
    norm_mode: str = "minmax",
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

    # Flush GPU state before creating env (prevents CUDA/EGL context interference)
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()

    # Create env
    env = create_robomimic_env(
        hdf5_path=hdf5_path,
        shape_meta=shape_meta,
        abs_action=True,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
    )

    # Warmup render to initialize OpenGL context before real eval
    obs = env.reset()
    env.step(np.zeros((n_action_steps, 7)))
    env.reset()

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
        ep_crashed = False

        while not done:
          try:
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

            # Normalize proprio
            proprio_min = norm_stats["proprio"]["min"]
            proprio_max = norm_stats["proprio"]["max"]
            p_range = proprio_max - proprio_min
            p_range = np.where(p_range < 1e-6, 1.0, p_range)
            if norm_mode == "chi":
                # Chi: pos=minmax, quat=identity, grip=minmax
                proprio_norm = proprio.copy()
                proprio_norm[..., :3] = 2.0 * (proprio[..., :3] - proprio_min[:3]) / np.clip(p_range[:3], 1e-6, None) - 1.0
                # quat [3:7] — identity
                proprio_norm[..., 7:9] = 2.0 * (proprio[..., 7:9] - proprio_min[7:9]) / np.clip(p_range[7:9], 1e-6, None) - 1.0
            else:
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

            # Denormalize actions
            if norm_mode == "chi":
                # Chi: only position [0:3] was minmax-normalized, rest is identity
                actions_raw = actions_norm.copy()
                pos_min = action_min[:3]
                pos_max = action_max[:3]
                pos_range = np.clip(pos_max - pos_min, 1e-6, None)
                actions_raw[..., :3] = (actions_norm[..., :3] + 1.0) / 2.0 * pos_range + pos_min
            else:
                # Standard minmax denorm on all dims
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
          except Exception as e:
            log.warning("Eval episode %d (seed %d) crashed at step: %s", ep_idx, seed, e)
            ep_crashed = True
            break

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


# ── Parallel eval with AsyncVectorEnv ──────────────────────────────────


def _create_env_fn(hdf5_path, shape_meta, abs_action, n_obs_steps,
                   n_action_steps, max_steps, render_offscreen=True):
    """Factory that returns a callable creating one env (for worker process)."""
    def _fn():
        # Each import happens inside the worker process
        from collections import defaultdict
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from data_pipeline.envs.robomimic_image_wrapper import RobomimicImageWrapper
        from data_pipeline.envs.multistep_wrapper import MultiStepWrapper

        modality_mapping = defaultdict(list)
        for key, attr in shape_meta['obs'].items():
            modality_mapping[attr.get('type', 'low_dim')].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        env_meta = FileUtils.get_env_metadata_from_dataset(hdf5_path)
        env_meta['env_kwargs']['use_object_obs'] = False

        if abs_action:
            ctrl = env_meta['env_kwargs']['controller_configs']
            if 'body_parts' in ctrl:
                for part in ctrl['body_parts'].values():
                    part['control_delta'] = False
                    part['input_type'] = 'absolute'
                    part['input_ref_frame'] = 'world'
                    part['input_min'] = -10
                    part['input_max'] = 10
            else:
                ctrl['control_delta'] = False
                ctrl['input_min'] = -10
                ctrl['input_max'] = 10

        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=render_offscreen,
            use_image_obs=render_offscreen,
        )
        # Disable hard_reset: we use reset_to() with cached states, so
        # hard_reset is bypassed anyway. Setting False avoids robosuite
        # allocating new sims without freeing old ones (VRAM leak).
        robomimic_env.env.hard_reset = False

        return MultiStepWrapper(
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
    return _fn


def evaluate_v3_robomimic_parallel(
    policy,
    ema_model=None,
    hdf5_path: str = "",
    norm_stats: dict = None,
    num_episodes: int = 50,
    n_envs: int = None,
    seed_start: int = 100000,
    max_steps: int = 400,
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    shape_meta: dict = None,
    use_rot6d: bool = True,
    device: str = "cuda",
    norm_mode: str = "minmax",
    save_video: bool = False,
    video_dir: str = "",
) -> tuple:
    """Parallel V3 evaluation using AsyncVectorEnv (matching Chi's runner).

    Each environment runs in a separate process with its own OpenGL context,
    fixing the framebuffer swap bug and enabling batched policy inference.

    Args:
        n_envs: Number of parallel envs. Defaults to num_episodes.
    """
    import math
    import dill
    from data_pipeline.envs.async_vector_env import AsyncVectorEnv

    if shape_meta is None:
        shape_meta = LIFT_SHAPE_META
    if n_envs is None:
        n_envs = num_episodes

    # Flush GPU before creating envs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # env_fn: full rendering (runs in worker process)
    env_fn = _create_env_fn(
        hdf5_path, shape_meta, abs_action=True,
        n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
        max_steps=max_steps, render_offscreen=True,
    )
    # dummy_env_fn: no rendering (space detection in main process)
    dummy_env_fn = _create_env_fn(
        hdf5_path, shape_meta, abs_action=True,
        n_obs_steps=n_obs_steps, n_action_steps=n_action_steps,
        max_steps=max_steps, render_offscreen=False,
    )

    log.info("Creating AsyncVectorEnv with %d workers...", n_envs)
    env = AsyncVectorEnv(
        [env_fn] * n_envs,
        dummy_env_fn=dummy_env_fn,
        shared_memory=False,
    )

    # Video setup
    imageio = None
    if save_video:
        import os
        os.makedirs(video_dir, exist_ok=True)
        try:
            import imageio as _imageio
            imageio = _imageio
        except ImportError:
            log.warning("imageio not installed, disabling video recording")
            save_video = False

    # Norm stats
    action_min = norm_stats["actions"]["min"]
    action_max = norm_stats["actions"]["max"]
    proprio_min = norm_stats["proprio"]["min"]
    proprio_max = norm_stats["proprio"]["max"]
    p_range = proprio_max - proprio_min
    p_range = np.where(p_range < 1e-6, 1.0, p_range)

    # Process episodes in chunks of n_envs
    n_chunks = math.ceil(num_episodes / n_envs)
    all_results = {}

    # Apply EMA weights for the entire eval
    if ema_model is not None:
        ema_model.store(policy.parameters())
        ema_model.copy_to(policy.parameters())

    try:
        for chunk_idx in range(n_chunks):
            start_ep = chunk_idx * n_envs
            end_ep = min(start_ep + n_envs, num_episodes)
            n_active = end_ep - start_ep

            # Seed each env via init function
            seeds = [seed_start + i for i in range(start_ep, end_ep)]
            # Pad if n_active < n_envs
            while len(seeds) < n_envs:
                seeds.append(seeds[0])

            def _make_init_fn(seed):
                def _init(env):
                    from data_pipeline.envs.robomimic_image_wrapper import RobomimicImageWrapper
                    assert isinstance(env.env, RobomimicImageWrapper)
                    env.env.init_state = None
                    env.seed(seed)
                return _init

            init_fns = [dill.dumps(_make_init_fn(s)) for s in seeds]
            env.call_each('run_dill_function', args_list=[(fn,) for fn in init_fns])

            # Reset all envs
            obs = env.reset()

            # Rollout loop
            all_rewards = [[] for _ in range(n_envs)]
            all_frames = [[] for _ in range(n_envs)] if save_video else None
            done_all = False

            while not done_all:
                # Convert obs dict to our format (batched)
                agentview = obs['agentview_image']  # (n_envs, T_obs, C, H, W) or (n_envs, T_obs, H, W, C)
                eye_in_hand = obs['robot0_eye_in_hand_image']

                # Handle HWC vs CHW (robomimic returns CHW float [0,1])
                if agentview.shape[-1] == 3 and agentview.ndim == 5:
                    # HWC → CHW
                    agentview = np.transpose(agentview, (0, 1, 4, 2, 3))
                    eye_in_hand = np.transpose(eye_in_hand, (0, 1, 4, 2, 3))

                # Ensure float32 [0,1]
                if agentview.dtype == np.uint8:
                    agentview = agentview.astype(np.float32) / 255.0
                    eye_in_hand = eye_in_hand.astype(np.float32) / 255.0

                T_obs_actual = agentview.shape[1]
                B = n_envs
                K = 4
                H, W = agentview.shape[-2], agentview.shape[-1]

                images = np.zeros((B, T_obs_actual, K, 3, H, W), dtype=np.float32)
                images[:, :, 0] = agentview
                images[:, :, 3] = eye_in_hand

                proprio = np.concatenate([
                    obs['robot0_eef_pos'],
                    obs['robot0_eef_quat'],
                    obs['robot0_gripper_qpos'],
                ], axis=-1)  # (B, T_obs, 9)

                # Normalize proprio
                if norm_mode == "chi":
                    # Chi: pos=minmax, quat=identity, grip=minmax
                    proprio_norm = proprio.copy()
                    proprio_norm[..., :3] = 2.0 * (proprio[..., :3] - proprio_min[:3]) / np.clip(p_range[:3], 1e-6, None) - 1.0
                    # quat [3:7] — identity
                    proprio_norm[..., 7:9] = 2.0 * (proprio[..., 7:9] - proprio_min[7:9]) / np.clip(p_range[7:9], 1e-6, None) - 1.0
                else:
                    proprio_norm = 2.0 * (proprio - proprio_min) / p_range - 1.0

                view_present = np.array([True, False, False, True])
                view_present_batch = np.tile(view_present, (B, 1))

                # To tensors
                images_t = torch.from_numpy(images).to(device)
                proprio_t = torch.from_numpy(proprio_norm).float().to(device)
                view_present_t = torch.from_numpy(view_present_batch).to(device)

                obs_dict = {
                    "images_enc": images_t,
                    "proprio": proprio_t,
                    "view_present": view_present_t,
                }

                # Batched policy inference
                with torch.no_grad():
                    actions_norm = policy.predict_action(obs_dict)

                # (B, T_pred, 10) → numpy
                actions_norm = actions_norm.cpu().numpy()

                # Denormalize
                if norm_mode == "chi":
                    actions_raw = actions_norm.copy()
                    pos_min = action_min[:3]
                    pos_max = action_max[:3]
                    pos_range = np.clip(pos_max - pos_min, 1e-6, None)
                    actions_raw[..., :3] = (actions_norm[..., :3] + 1.0) / 2.0 * pos_range + pos_min
                else:
                    a_range = action_max - action_min
                    a_range = np.where(np.abs(a_range) < 1e-6, 1.0, a_range)
                    actions_raw = (actions_norm + 1.0) / 2.0 * a_range + action_min

                # Convert rot6d → axis_angle per env
                if use_rot6d:
                    actions_7d = np.stack([
                        convert_actions_from_rot6d(actions_raw[i])
                        for i in range(B)
                    ])  # (B, T_pred, 7)
                else:
                    actions_7d = actions_raw

                # Extract exec actions
                exec_actions = actions_7d[:, :n_action_steps]  # (B, 8, 7)

                obs, reward, done, info = env.step(exec_actions)

                for i in range(n_active):
                    all_rewards[i].append(reward[i])
                    if save_video:
                        frame = obs['agentview_image'][i, -1]  # (C,H,W) or (H,W,C)
                        if frame.shape[0] == 3 and frame.ndim == 3:
                            frame = np.transpose(frame, (1, 2, 0))  # CHW → HWC
                        if frame.dtype != np.uint8:
                            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                        all_frames[i].append(frame)

                done_all = np.all(done[:n_active])

            # Collect results and save videos for this chunk
            for i in range(n_active):
                ep_idx = start_ep + i
                max_reward = np.max(all_rewards[i]) if all_rewards[i] else 0.0
                success = max_reward > 0.5
                all_results[ep_idx] = {
                    "seed": seeds[i],
                    "success": bool(success),
                    "max_reward": float(max_reward),
                }
                if save_video and all_frames[i]:
                    tag = "success" if success else "fail"
                    video_path = os.path.join(
                        video_dir, f"ep{ep_idx:03d}_seed{seeds[i]}_{tag}.mp4")
                    imageio.mimwrite(video_path, all_frames[i], fps=10)

            chunk_sr = sum(all_results[start_ep + i]["success"] for i in range(n_active)) / n_active
            log.info("Eval chunk %d/%d: %d/%d success (%.0f%%)",
                     chunk_idx + 1, n_chunks, int(chunk_sr * n_active), n_active, chunk_sr * 100)

    finally:
        # Restore non-EMA weights
        if ema_model is not None:
            ema_model.restore(policy.parameters())
        env.close()

    success_rate = sum(r["success"] for r in all_results.values()) / num_episodes
    log.info("Parallel eval: %.1f%% success (%d/%d episodes)",
             success_rate * 100, int(success_rate * num_episodes), num_episodes)

    return success_rate, all_results


# ── CLI entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="V3 robomimic evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to V3 checkpoint")
    parser.add_argument("--hdf5", required=True, help="Path to unified HDF5 with env_args")
    parser.add_argument("--stage1_checkpoint", type=str, default="",
                        help="Path to Stage 1 checkpoint (optional if using cached tokens)")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--n_envs", type=int, default=None, help="Parallel envs (default: num_episodes)")
    parser.add_argument("--norm_mode", default=None, choices=["minmax", "chi"],
                        help="Override norm mode (default: read from checkpoint config)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequential", action="store_true", help="Use sequential eval (no AsyncVectorEnv)")
    parser.add_argument("--save_video", action="store_true", help="Save MP4 videos of rollouts")
    parser.add_argument("--video_dir", default="checkpoints/eval_videos", help="Directory for video files")

    # Architecture overrides (auto-detected from checkpoint config if available)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--n_cond_layers", type=int, default=None)
    parser.add_argument("--T_pred", type=int, default=None)
    parser.add_argument("--n_active_cams", type=int, default=None)
    parser.add_argument("--spatial_pool_size", type=int, default=None)
    parser.add_argument("--use_flow_matching", action="store_true", default=None)
    parser.add_argument("--denoiser_type", type=str, default=None)
    args = parser.parse_args()

    from data_pipeline.conversion.compute_norm_stats import load_norm_stats
    from models.policy_v3 import PolicyDiTv3
    from models.stage1_bridge import Stage1Bridge

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint and extract config
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    if cfg:
        log.info("Found config in checkpoint")
    else:
        log.info("No config in checkpoint — using CLI args / defaults")

    # Helper: CLI arg > checkpoint config > default
    def _get(name, default):
        cli_val = getattr(args, name, None)
        if cli_val is not None:
            return cli_val
        return cfg.get(name, default)

    d_model = _get("d_model", 256)
    n_head = _get("n_head", 4)
    n_layers = _get("n_layers", 8)
    n_cond_layers = _get("n_cond_layers", 0)
    T_pred = _get("T_pred", 10)
    n_active_cams = _get("n_active_cams", 2)
    spatial_pool_size = _get("spatial_pool_size", 1)
    use_flow_matching = _get("use_flow_matching", False)
    denoiser_type = _get("denoiser_type", "transformer")
    prediction_type = _get("prediction_type", "epsilon")
    cfg_drop_rate = _get("cfg_drop_rate", 0.0)
    cfg_scale = args.cfg_scale if hasattr(args, "cfg_scale") and args.cfg_scale is not None else _get("cfg_scale", 1.0)
    use_rope = _get("use_rope", False)
    norm_mode = args.norm_mode or cfg.get("norm_mode", "chi")

    log.info("Arch: d=%d, heads=%d, layers=%d, ncond=%d, spatial=%d, fm=%s, denoiser=%s",
             d_model, n_head, n_layers, n_cond_layers, spatial_pool_size,
             use_flow_matching, denoiser_type)

    # Build model
    bridge = Stage1Bridge(checkpoint_path=args.stage1_checkpoint)
    policy = PolicyDiTv3(
        bridge=bridge,
        ac_dim=_get("ac_dim", 10),
        proprio_dim=_get("proprio_dim", 9),
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        T_obs=_get("T_obs", 2),
        T_pred=T_pred,
        n_active_cams=n_active_cams,
        spatial_pool_size=spatial_pool_size,
        n_cond_layers=n_cond_layers,
        use_flow_matching=use_flow_matching,
        denoiser_type=denoiser_type,
        prediction_type=prediction_type,
        cfg_drop_rate=cfg_drop_rate,
        cfg_scale=cfg_scale,
        use_rope=use_rope,
    ).to(device)

    # Load policy weights
    if "denoiser" in ckpt:
        def _strip(sd):
            prefix = "_orig_mod."
            if any(k.startswith(prefix) for k in sd):
                return {k.removeprefix(prefix): v for k, v in sd.items()}
            return sd
        policy.denoiser.load_state_dict(_strip(ckpt["denoiser"]))
        policy.obs_encoder.load_state_dict(_strip(ckpt["obs_encoder"]))
        policy.bridge.adapter.load_state_dict(_strip(ckpt["adapter"]))
    else:
        policy.load_state_dict(ckpt.get("policy", ckpt.get("model", {})), strict=False)

    # Load EMA weights directly into policy (Chi's approach)
    if "ema" in ckpt:
        ema = ckpt["ema"]
        if "averaged_model" in ema:
            # Old format: full model state_dict
            policy.load_state_dict(ema["averaged_model"], strict=False)
        elif "denoiser" in ema:
            # New format: component-wise (no frozen encoder)
            policy.denoiser.load_state_dict(_strip(ema["denoiser"]))
            policy.obs_encoder.load_state_dict(_strip(ema["obs_encoder"]))
            policy.bridge.adapter.load_state_dict(_strip(ema["adapter"]))
        log.info("Loaded EMA weights into policy for eval")
    policy.eval()

    norm_stats = load_norm_stats(args.hdf5)

    if args.sequential:
        success_rate, results = evaluate_v3_robomimic(
            policy=policy, ema_model=None,
            hdf5_path=args.hdf5, norm_stats=norm_stats,
            num_episodes=args.num_episodes,
            use_rot6d=True, device=str(device),
            norm_mode=norm_mode,
            save_video=args.save_video,
            video_dir=args.video_dir,
        )
    else:
        success_rate, results = evaluate_v3_robomimic_parallel(
            policy=policy, ema_model=None,
            hdf5_path=args.hdf5, norm_stats=norm_stats,
            num_episodes=args.num_episodes, n_envs=args.n_envs,
            use_rot6d=True, device=str(device),
            norm_mode=norm_mode,
            save_video=args.save_video,
            video_dir=args.video_dir,
        )

    print(f"\nFinal: {success_rate*100:.1f}% ({int(success_rate*args.num_episodes)}/{args.num_episodes})")
    for ep_idx, r in sorted(results.items()):
        print(f"  ep{ep_idx:03d} seed={r['seed']} {'SUCCESS' if r['success'] else 'FAIL'} reward={r['max_reward']:.2f}")
