"""V3 evaluation for RLBench tasks.

Standalone eval loop (does NOT use rollout.py) because:
  - T_a=1: predict chunk, execute only first action, re-observe
  - Action conversion: 10D rot6d → 8D quaternion (different from robomimic's 10D → 7D aa)
  - Keeping rollout.py clean for robomimic avoids cross-benchmark bugs

Eval protocol (PerAct/CoA standard):
  - 25 episodes, variation 0, task.reset() between episodes
  - Success from RLBench's built-in success conditions
  - Per-episode timeout to prevent OMPL planning hangs

Usage:
    from training.eval_v3_rlbench import evaluate_v3_rlbench
    sr, results, env = evaluate_v3_rlbench(wrapper, norm_stats, task="close_jar")
"""

import logging
import os
import signal
from collections import deque

import numpy as np
import torch

log = logging.getLogger(__name__)

# Default max steps per RLBench task (episode length + margin)
TASK_MAX_STEPS = {
    "close_jar": 250,
    "open_drawer": 150,
    "slide_block_to_color_target": 200,
    "meat_off_grill": 200,
    "place_wine_at_rack_location": 250,
    "push_buttons": 250,
    "sweep_to_dustpan_of_size": 250,
    "sweep_to_dustpan": 250,
    "turn_tap": 200,
    "reach_target": 100,
}


def _denorm_and_convert(pred: np.ndarray, action_stats: dict,
                        norm_mode: str = "chi") -> np.ndarray:
    """Denormalize predicted actions and convert to 8D for OMPL.

    Supports both 10D rot6d (chi mode) and 8D canonical quat (minmax_margin).

    Args:
        pred: (T_p, D) normalized actions from policy. D=10 (rot6d) or D=8 (quat).
        action_stats: dict with 'min' and 'max'.
        norm_mode: normalization mode used during training.

    Returns:
        (T_p, 8) denormalized actions [pos(3), quat_xyzw(4), grip(1)].
    """
    raw = pred.copy()
    D = raw.shape[-1]

    if norm_mode == "minmax_margin":
        # Robobase-style: inverse minmax with 0.2 margin on ALL dims
        margin = 0.2
        a_min = action_stats["min"] - np.abs(action_stats["min"]) * margin
        a_max = action_stats["max"] + np.abs(action_stats["max"]) * margin
        raw = (raw + 1.0) / 2.0 * (a_max - a_min) + a_min

        if D == 8:
            # Renormalize quaternion to unit length (critical for OMPL)
            quat = raw[:, 3:7]
            quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
            quat_norm = np.where(quat_norm < 1e-6, 1.0, quat_norm)
            raw[:, 3:7] = quat / quat_norm
            return raw
        # Fall through for 10D rot6d case

    elif norm_mode == "chi":
        # Chi: only position [0:3] was minmax'd
        pos_min = action_stats["min"][:3]
        pos_max = action_stats["max"][:3]
        raw[:, :3] = (raw[:, :3] + 1.0) / 2.0 * (pos_max - pos_min) + pos_min
    else:
        # Plain minmax: all dims
        a_range = action_stats["max"] - action_stats["min"]
        raw = (raw + 1.0) / 2.0 * a_range + action_stats["min"]

    if D == 10:
        # Convert 10D rot6d → 8D quaternion
        from data_pipeline.utils.rotation import convert_actions_rot6d_to_quat
        return convert_actions_rot6d_to_quat(raw)
    elif D == 8:
        # Already 8D quat — renormalize
        quat = raw[:, 3:7]
        quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
        quat_norm = np.where(quat_norm < 1e-6, 1.0, quat_norm)
        raw[:, 3:7] = quat / quat_norm
        return raw
    else:
        raise ValueError(f"Unexpected action dim {D}")


def _temporal_ensemble(action_history, step, T_pred, gain=0.01):
    """Compute temporally-ensembled action for the current step.

    Averages overlapping predictions from previous steps with exponential
    weighting (most recent prediction gets highest weight).

    Args:
        action_history: list of (T_pred, 8) action chunks, one per past step.
        step: current step index.
        T_pred: prediction horizon.
        gain: exponential decay rate (0.01 = nearly uniform, higher = more recent).

    Returns:
        (8,) ensembled action for the current step.
    """
    # action_history is a list of (step_when_predicted, chunk) tuples
    predictions = []
    for pred_step, chunk in action_history:
        offset = step - pred_step  # which index in that chunk corresponds to now
        if 0 <= offset < chunk.shape[0]:
            predictions.append(chunk[offset])

    if len(predictions) == 0:
        raise ValueError(f"No predictions cover step {step}")
    if len(predictions) == 1:
        return predictions[0]

    preds = np.stack(predictions, axis=0)  # (N, 8)
    # Weights: most recent prediction (last) gets highest weight
    weights = np.exp(-gain * np.arange(len(preds))[::-1])
    weights /= weights.sum()
    action = (preds * weights[:, None]).sum(axis=0)

    # Re-normalize quaternion to unit length (weighted avg of unit quats isn't unit)
    quat = action[3:7]
    quat_norm = np.linalg.norm(quat)
    if quat_norm > 1e-6:
        action[3:7] = quat / quat_norm

    return action


def _run_episode(
    policy,
    env,
    action_stats: dict,
    proprio_stats: dict,
    max_steps: int,
    obs_horizon: int,
    save_frames: bool = False,
    temporal_ensemble: bool = True,
    ensemble_gain: float = 0.01,
    exec_horizon: int = 1,
    demo=None,
    norm_mode: str = "chi",
) -> dict:
    """Run a single RLBench episode.

    Args:
        exec_horizon: Number of actions to execute per prediction (T_a).
            T_a=1: re-predict every step (default, most reactive).
            T_a>1: execute multiple actions from each chunk before re-predicting
                   (reduces gripper flip-flopping, less reactive to errors).

    Returns dict with 'success', 'steps', 'reward', and optionally 'frames'.
    """
    if demo is not None:
        try:
            descriptions, obs = env._task.reset_to_demo(demo)
        except (KeyError, AttributeError):
            env._task.set_variation(demo.variation_number)
            descriptions, obs = env._task.reset()
        env._last_obs = obs
    else:
        env.reset()

    # Initialize observation buffer (duplicate first frame for start padding)
    init_images = env.get_multiview_images()  # [1, K, 3, H, W]
    init_proprio = env.get_proprio()            # [1, D_prop]
    img_buffer = deque([init_images] * obs_horizon, maxlen=obs_horizon)
    proprio_buffer = deque([init_proprio] * obs_horizon, maxlen=obs_horizon)
    view_present = env.get_view_present()       # [K] bool

    # Proprio normalization stats
    p_min, p_max = proprio_stats["min"], proprio_stats["max"]

    # Temporal ensemble: keep recent action chunks for averaging
    T_pred = None  # inferred from first prediction
    action_history = deque(maxlen=50)  # keep last 50 chunks (more than enough)

    frames = []
    step_count = 0
    total_reward = 0.0
    info = {}
    pending_actions = []  # buffered actions for exec_horizon > 1

    while step_count < max_steps:
        # If we have pending actions from a previous chunk, execute next one
        if pending_actions:
            action = pending_actions.pop(0)
        else:
            # Predict a new chunk
            images_seq = np.concatenate(list(img_buffer), axis=0)
            proprio_seq = np.concatenate(list(proprio_buffer), axis=0)

            # Normalize proprio (must match training norm_mode)
            if norm_mode == "minmax_margin":
                margin = 0.2
                p_min_m = p_min - np.abs(p_min) * margin
                p_max_m = p_max + np.abs(p_max) * margin
                p_range = np.clip(p_max_m - p_min_m, 1e-6, None)
                proprio_norm = 2.0 * (proprio_seq - p_min_m) / p_range - 1.0
            elif norm_mode == "chi":
                proprio_norm = proprio_seq.copy()
                pos_range = np.clip(p_max[:3] - p_min[:3], 1e-6, None)
                proprio_norm[..., :3] = 2.0 * (proprio_seq[..., :3] - p_min[:3]) / pos_range - 1.0
                g_min, g_max = p_min[7:], p_max[7:]
                g_range = np.clip(g_max - g_min, 1e-6, None)
                proprio_norm[..., 7:] = 2.0 * (proprio_seq[..., 7:] - g_min) / g_range - 1.0
            else:
                # plain minmax
                p_range = np.clip(p_max - p_min, 1e-6, None)
                proprio_norm = 2.0 * (proprio_seq - p_min) / p_range - 1.0

            with torch.no_grad():
                pred = policy.predict(
                    torch.from_numpy(images_seq).unsqueeze(0),
                    torch.from_numpy(proprio_norm).unsqueeze(0),
                    torch.from_numpy(view_present).unsqueeze(0),
                )

            raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
            actions_8d = _denorm_and_convert(raw, action_stats, norm_mode=norm_mode)

            if T_pred is None:
                T_pred = actions_8d.shape[0]

            if temporal_ensemble:
                action_history.append((step_count, actions_8d))
                action = _temporal_ensemble(
                    list(action_history), step_count, T_pred, gain=ensemble_gain,
                )
                # Buffer remaining actions for exec_horizon > 1
                if exec_horizon > 1:
                    for j in range(1, min(exec_horizon, T_pred)):
                        a = _temporal_ensemble(
                            list(action_history), step_count + j, T_pred, gain=ensemble_gain,
                        )
                        pending_actions.append(a)
            else:
                action = actions_8d[0]
                if exec_horizon > 1:
                    for j in range(1, min(exec_horizon, len(actions_8d))):
                        pending_actions.append(actions_8d[j])

        if step_count < 5 or step_count % 20 == 0:
            log.info("  step=%d predicted action: pos=%s quat=%s grip=%.2f",
                     step_count,
                     np.array2string(action[:3], precision=4, suppress_small=True),
                     np.array2string(action[3:7], precision=4, suppress_small=True),
                     action[7])
            if demo is not None and step_count < len(demo) - 1:
                gt_obs = demo[step_count + 1]
                gt_pos = gt_obs.gripper_pose[:3]
                gt_quat = gt_obs.gripper_pose[3:]
                gt_grip = gt_obs.misc.get("joint_position_action", [0]*8)[-1]
                pos_err = np.linalg.norm(action[:3] - gt_pos)
                log.info("  step=%d GT action:        pos=%s quat=%s grip=%.2f  |  pos_err=%.4f",
                         step_count,
                         np.array2string(gt_pos, precision=4, suppress_small=True),
                         np.array2string(gt_quat, precision=4, suppress_small=True),
                         gt_grip, pos_err)

        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            log.warning("  OMPL step failed at step %d: %s", step_count, e)
            return {"success": False, "steps": step_count, "reward": total_reward}

        step_count += 1
        total_reward += reward

        # Update observation buffer
        img_buffer.append(env.get_multiview_images())
        proprio_buffer.append(env.get_proprio())

        if save_frames:
            img = img_buffer[-1][0]  # [K, 3, H, W] float32 [0,1]
            views = [img[i].transpose(1, 2, 0) for i in range(img.shape[0])]
            frame = (np.concatenate(views, axis=1) * 255).astype(np.uint8)
            frames.append(frame)

        if done:
            break

    success = info.get("success", False) if isinstance(info, dict) else False
    result = {"success": success, "steps": step_count, "reward": total_reward}
    if save_frames:
        result["frames"] = frames
    return result


def _run_episode_keyframe(
    policy,
    env,
    action_stats: dict,
    proprio_stats: dict,
    obs_horizon: int,
    save_frames: bool = False,
    demo=None,
    norm_mode: str = "chi",
) -> dict:
    """Run a single RLBench episode in keyframe mode.

    Predicts the full keyframe sequence ONCE, then executes each keyframe
    action sequentially through OMPL. Re-observes between actions for video
    frames but does NOT re-predict. This matches how PerAct/CoA evaluate.

    Returns dict with 'success', 'steps', 'reward', and optionally 'frames'.
    """
    if demo is not None:
        try:
            descriptions, obs = env._task.reset_to_demo(demo)
        except (KeyError, AttributeError):
            env._task.set_variation(demo.variation_number)
            descriptions, obs = env._task.reset()
        env._last_obs = obs
    else:
        env.reset()

    # Proprio normalization stats (chi mode)
    p_min, p_max = proprio_stats["min"], proprio_stats["max"]

    # Get initial observation
    init_images = env.get_multiview_images()  # [1, K, 3, H, W]
    init_proprio = env.get_proprio()           # [1, D_prop]
    view_present = env.get_view_present()      # [K] bool

    # Build observation sequence (duplicate for obs_horizon padding)
    images_seq = np.concatenate([init_images] * obs_horizon, axis=0)
    proprio_seq = np.concatenate([init_proprio] * obs_horizon, axis=0)

    # Normalize proprio (must match training norm_mode)
    if norm_mode == "minmax_margin":
        margin = 0.2
        p_min_m = p_min - np.abs(p_min) * margin
        p_max_m = p_max + np.abs(p_max) * margin
        p_range = np.clip(p_max_m - p_min_m, 1e-6, None)
        proprio_norm = 2.0 * (proprio_seq - p_min_m) / p_range - 1.0
    elif norm_mode == "chi":
        proprio_norm = proprio_seq.copy()
        pos_range = np.clip(p_max[:3] - p_min[:3], 1e-6, None)
        proprio_norm[..., :3] = 2.0 * (proprio_seq[..., :3] - p_min[:3]) / pos_range - 1.0
        g_min, g_max = p_min[7:], p_max[7:]
        g_range = np.clip(g_max - g_min, 1e-6, None)
        proprio_norm[..., 7:] = 2.0 * (proprio_seq[..., 7:] - g_min) / g_range - 1.0
    else:
        p_range = np.clip(p_max - p_min, 1e-6, None)
        proprio_norm = 2.0 * (proprio_seq - p_min) / p_range - 1.0

    # Predict FULL keyframe sequence in one shot
    with torch.no_grad():
        pred = policy.predict(
            torch.from_numpy(images_seq).unsqueeze(0),
            torch.from_numpy(proprio_norm).unsqueeze(0),
            torch.from_numpy(view_present).unsqueeze(0),
        )

    raw = pred.cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)
    actions_8d = _denorm_and_convert(raw, action_stats)  # (T_pred, 8)

    log.debug("  Predicted %d keyframe actions", actions_8d.shape[0])

    # Execute each keyframe action through OMPL
    frames = []
    step_count = 0
    total_reward = 0.0
    info = {}

    # Frames per keyframe: hold each keyframe for ~1s at 10fps
    HOLD_FRAMES = 10

    # Capture initial frame for video
    if save_frames:
        img = init_images[0]  # [K, 3, H, W]
        views = [img[i].transpose(1, 2, 0) for i in range(img.shape[0])]
        frame = (np.concatenate(views, axis=1) * 255).astype(np.uint8)
        frames.extend([frame] * HOLD_FRAMES)

    for i, action in enumerate(actions_8d):
        try:
            obs, reward, done, info = env.step(action)
        except Exception as e:
            log.warning("  Keyframe %d OMPL failed: %s", i, e)
            continue  # skip failed plans, try next keyframe

        step_count += 1
        total_reward += reward

        if save_frames:
            img = env.get_multiview_images()[0]  # [K, 3, H, W]
            views = [img[j].transpose(1, 2, 0) for j in range(img.shape[0])]
            frame = (np.concatenate(views, axis=1) * 255).astype(np.uint8)
            frames.extend([frame] * HOLD_FRAMES)

        if info.get("success", False):
            log.debug("  SUCCESS at keyframe %d", i)
            break
        if done:
            break

    success = info.get("success", False) if isinstance(info, dict) else False
    result = {"success": success, "steps": step_count, "reward": total_reward}
    if save_frames:
        result["frames"] = frames
    return result


def evaluate_v3_rlbench(
    wrapper,  # V3PolicyWrapper
    norm_stats: dict,
    task: str = "close_jar",
    num_episodes: int = 25,
    max_steps: int = 0,
    seed_start: int = 0,
    obs_horizon: int = 2,
    image_size: int = 224,
    headless: bool = True,
    episode_timeout: int = 180,
    save_video: bool = False,
    video_dir: str = "",
    demos: list = None,
    exec_horizon: int = 1,
    keyframe_eval: bool = False,
    use_ik: bool = False,
    norm_mode: str = "chi",
    _cached_env=None,
) -> tuple:
    """Run V3 evaluation on an RLBench task.

    Args:
        wrapper:         V3PolicyWrapper wrapping PolicyDiTv3.
        norm_stats:      dict with 'actions' and 'proprio' stats (from HDF5).
        task:            RLBench task name.
        num_episodes:    Number of eval episodes (PerAct standard: 25).
        max_steps:       Max steps per episode (0 = auto from TASK_MAX_STEPS).
        seed_start:      First episode seed (0 = no explicit seeding).
        obs_horizon:     T_o = 2 (past frames to condition on).
        image_size:      RLBench render size (224).
        headless:        Run CoppeliaSim headless.
        episode_timeout: Per-episode timeout in seconds (prevents OMPL hangs).
        keyframe_eval:   Predict full keyframe sequence once, execute all
                         through OMPL (no temporal ensemble, no re-prediction).
        _cached_env:     Pre-created RLBenchWrapper to reuse across eval calls.

    Returns:
        (success_rate, per_episode_results, env)
        env is returned so callers can cache it for reuse.
    """
    from data_pipeline.envs.rlbench_wrapper import RLBenchWrapper

    if max_steps == 0:
        max_steps = TASK_MAX_STEPS.get(task, 200)

    # Reuse cached env or create new one
    env = _cached_env
    if env is None:
        log.info("Creating RLBenchWrapper for %s (headless=%s, ik=%s)...", task, headless, use_ik)
        env = RLBenchWrapper(
            task_name=task,
            image_size=image_size,
            headless=headless,
            use_ik=use_ik,
        )

    action_stats = norm_stats["actions"]
    proprio_stats = norm_stats["proprio"]

    if save_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)

    class _EpisodeTimeout(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise _EpisodeTimeout()

    results = []
    for ep in range(num_episodes):
        if seed_start > 0:
            env.seed(seed_start + ep)

        # Per-episode timeout via signal.alarm (main thread, Linux only)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(episode_timeout)

        try:
            demo = demos[ep] if demos is not None and ep < len(demos) else None
            if keyframe_eval:
                ep_result = _run_episode_keyframe(
                    wrapper, env, action_stats, proprio_stats,
                    obs_horizon,
                    save_frames=save_video,
                    demo=demo,
                    norm_mode=norm_mode,
                )
            else:
                ep_result = _run_episode(
                    wrapper, env, action_stats, proprio_stats,
                    max_steps, obs_horizon,
                    save_frames=save_video,
                    exec_horizon=exec_horizon,
                    demo=demo,
                    norm_mode=norm_mode,
                )
            signal.alarm(0)  # cancel alarm

            if save_video and ep_result.get("frames"):
                import imageio
                tag = "success" if ep_result["success"] else "fail"
                path = os.path.join(video_dir, f"ep{ep:03d}_{tag}.mp4")
                imageio.mimwrite(path, ep_result.pop("frames"), fps=10)

            results.append(ep_result)
        except _EpisodeTimeout:
            log.warning("Episode %d timed out after %ds — recreating env", ep, episode_timeout)
            results.append({"success": False, "steps": 0, "reward": 0.0})
            try:
                env.close()
            except Exception:
                pass
            env = RLBenchWrapper(
                task_name=task, image_size=image_size, headless=headless,
                use_ik=use_ik,
            )
        except Exception as e:
            signal.alarm(0)
            log.warning("Episode %d failed: %s", ep, e)
            results.append({"success": False, "steps": 0, "reward": 0.0})
        finally:
            signal.signal(signal.SIGALRM, old_handler)

        n_success = sum(1 for r in results if r["success"])
        log.info("Episode %d/%d: %s (%d steps) | Running: %d/%d (%.0f%%)",
                 ep + 1, num_episodes,
                 "SUCCESS" if results[-1]["success"] else "FAIL",
                 results[-1]["steps"],
                 n_success, len(results),
                 100 * n_success / len(results))

    n_success = sum(1 for r in results if r["success"])
    success_rate = n_success / num_episodes  # denominator is always num_episodes
    log.info("RLBench eval (%s): %.1f%% success (%d/%d episodes)",
             task, success_rate * 100, n_success, num_episodes)

    return success_rate, results, env


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="V3 RLBench evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to V3 checkpoint")
    parser.add_argument("--stage1_checkpoint", required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--eval_hdf5", required=True, help="HDF5 for norm stats")
    parser.add_argument("--task", required=True, help="RLBench task name")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--ac_dim", type=int, default=10)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--n_active_cams", type=int, default=4)
    parser.add_argument("--T_pred", type=int, default=10)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_dir", default="checkpoints/eval_videos")
    parser.add_argument("--demo_pickles", default="",
                        help="Path to episodes dir for reset_to_demo (eval on known scenes)")
    parser.add_argument("--exec_horizon", type=int, default=1,
                        help="Actions to execute per prediction (T_a). Higher = less gripper flip-flop")
    parser.add_argument("--keyframe_eval", action="store_true",
                        help="Keyframe mode: predict full sequence once, execute all through OMPL")
    parser.add_argument("--use_ik", action="store_true",
                        help="Use EndEffectorPoseViaIK instead of OMPL planning (faster, closer to Robomimic)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import pickle
    from pathlib import Path
    from data_pipeline.conversion.compute_norm_stats import load_norm_stats
    from models.policy_v3 import PolicyDiTv3
    from models.stage1_bridge import Stage1Bridge
    from training.eval_v3 import V3PolicyWrapper

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Auto-detect architecture from checkpoint config
    ckpt_peek = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt_peek.get("config", {})
    def _get(key, default):
        return cfg.get(key, default)

    spatial_pool_size = _get("spatial_pool_size", 1)
    n_cond_layers = _get("n_cond_layers", 0)
    use_flow_matching = _get("use_flow_matching", False)
    denoiser_type = _get("denoiser_type", "transformer")
    d_model = _get("d_model", 256)
    n_head = _get("n_head", 4)
    n_layers = _get("n_layers", 8)
    norm_mode = _get("norm_mode", "chi")
    T_obs = _get("T_obs", 1)

    log.info("Auto-detected: spatial=%d, ncond=%d, fm=%s, d=%d, norm=%s",
             spatial_pool_size, n_cond_layers, use_flow_matching, d_model, norm_mode)
    del ckpt_peek

    # Build model
    bridge = Stage1Bridge(checkpoint_path=args.stage1_checkpoint)
    policy = PolicyDiTv3(
        bridge=bridge, ac_dim=args.ac_dim,
        proprio_dim=args.proprio_dim, n_active_cams=args.n_active_cams,
        T_pred=args.T_pred,
        d_model=d_model, n_head=n_head, n_layers=n_layers,
        spatial_pool_size=spatial_pool_size,
        n_cond_layers=n_cond_layers,
        use_flow_matching=use_flow_matching,
        denoiser_type=denoiser_type,
        T_obs=T_obs,
    ).to(device)

    # Load checkpoint + EMA weights
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    def _strip(sd):
        prefix = "_orig_mod."
        return {k.removeprefix(prefix): v for k, v in sd.items()} if any(
            k.startswith(prefix) for k in sd) else sd

    if "ema" in ckpt:
        ema = ckpt["ema"]
        if "averaged_model" in ema:
            policy.load_state_dict(ema["averaged_model"], strict=False)
        elif "denoiser" in ema:
            policy.denoiser.load_state_dict(_strip(ema["denoiser"]))
            policy.obs_encoder.load_state_dict(_strip(ema["obs_encoder"]))
            policy.bridge.adapter.load_state_dict(_strip(ema["adapter"]))
        log.info("Loaded EMA weights for eval")
    elif "denoiser" in ckpt:
        policy.denoiser.load_state_dict(_strip(ckpt["denoiser"]))
        policy.obs_encoder.load_state_dict(_strip(ckpt["obs_encoder"]))
        policy.bridge.adapter.load_state_dict(_strip(ckpt["adapter"]))
    policy.eval()

    wrapper = V3PolicyWrapper(policy, device=str(device))
    norm_stats = load_norm_stats(args.eval_hdf5)

    # Load demo pickles for scene restoration (optional)
    demo_list = None
    if args.demo_pickles:
        ep_root = Path(args.demo_pickles)
        ep_dirs = sorted(
            [d for d in ep_root.iterdir()
             if d.is_dir() and (d / "low_dim_obs.pkl").exists()],
            key=lambda d: int(d.name.replace("episode", "")),
        )
        demo_list = []
        for d in ep_dirs[:args.num_episodes]:
            with open(d / "low_dim_obs.pkl", "rb") as fh:
                demo_list.append(pickle.load(fh))
        log.info("Loaded %d demo pickles for scene restoration", len(demo_list))

    sr, results, env = evaluate_v3_rlbench(
        wrapper, norm_stats,
        task=args.task,
        num_episodes=args.num_episodes,
        save_video=args.save_video,
        video_dir=args.video_dir,
        demos=demo_list,
        exec_horizon=args.exec_horizon,
        keyframe_eval=args.keyframe_eval,
        norm_mode=norm_mode,
        use_ik=args.use_ik,
    )
    env.close()

    print(f"\nFinal: {sr*100:.1f}% ({int(sr*args.num_episodes)}/{args.num_episodes})")
    for i, r in enumerate(results):
        print(f"  ep{i:03d} {'SUCCESS' if r['success'] else 'FAIL'} steps={r['steps']}")
