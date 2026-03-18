"""Bridge test: Chi's trained policy + our RobomimicWrapper.

Validates that our env creation + action pipeline produces
the same success rate as Chi's own eval infrastructure.

Usage (on pc03):
  python training/eval_bridge.py \
    --checkpoint /student/alinaqee/CSC415/diffusion_policy/data/outputs/lift_image_transformer/checkpoints/latest.ckpt \
    --n_episodes 25
"""
import os, sys, warnings, logging
import numpy as np
import torch

warnings.filterwarnings("ignore", module="robosuite")
logging.getLogger("robosuite").setLevel(logging.ERROR)

# Add both repos to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.expanduser("~/CSC415/diffusion_policy"))

from collections import deque


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # --- 1. Load Chi's checkpoint ---
    import dill
    print(f"Loading checkpoint: {args.checkpoint}")
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill, map_location='cpu')
    cfg = payload['cfg']

    # Instantiate workspace and load weights
    from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import (
        TrainDiffusionTransformerHybridWorkspace,
    )
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg, output_dir="/tmp/eval_bridge")
    workspace.load_payload(payload, exclude_keys=['optimizer'], include_keys=['global_step', 'epoch'])

    # Get EMA policy
    if cfg.training.use_ema:
        policy = workspace.ema_model
        print("Using EMA model")
    else:
        policy = workspace.model
        print("Using raw model")
    policy.to(device)
    policy.eval()

    # --- 2. Set up rot6d -> axis_angle converter (our own, no pytorch3d needed) ---
    from data_pipeline.utils.rotation import convert_actions_from_rot6d

    def undo_rot6d(action_np):
        """Convert 10D (pos3+rot6d6+grip1) -> 7D (pos3+aa3+grip1)."""
        return convert_actions_from_rot6d(action_np)

    # --- 3. Create OUR RobomimicWrapper ---
    from data_pipeline.envs.robomimic_wrapper import RobomimicWrapper
    print("Creating RobomimicWrapper with abs_action=True")

    # --- 4. Run episodes ---
    n_obs_steps = cfg.n_obs_steps  # 2
    n_action_steps = cfg.n_action_steps  # 8
    successes = 0

    for ep in range(args.n_episodes):
        seed = 100000 + ep  # Match Chi's test_start_seed
        env = RobomimicWrapper(task="lift", seed=seed, abs_action=True)
        obs_raw = env.reset()

        # Build obs buffers (raw robosuite obs, NOT our processed images)
        obs_history = deque(maxlen=n_obs_steps)
        obs_history.append(obs_raw)
        while len(obs_history) < n_obs_steps:
            obs_history.append(obs_raw)

        action_queue = []
        step = 0
        success = False

        while step < args.max_steps:
            if len(action_queue) == 0:
                # Format obs_dict for Chi's policy (raw 84x84 images)
                obs_list = list(obs_history)
                obs_dict = {}
                for key in ['agentview_image', 'robot0_eye_in_hand_image']:
                    imgs = []
                    for o in obs_list:
                        # Flip vertically: robosuite returns bottom-up,
                        # robomimic's EnvRobosuite flips to top-up.
                        img = o[key][::-1].copy()
                        # Convert to float [0,1] — normalizer was fitted on [0,1] data
                        img = img.astype(np.float32) / 255.0
                        # HWC→CHW (obs_encoder expects CHW)
                        img = np.moveaxis(img, -1, 0)  # (3, H, W) float [0,1]
                        imgs.append(img)
                    obs_dict[key] = torch.from_numpy(
                        np.stack(imgs)  # (T_o, 3, H, W) uint8
                    ).unsqueeze(0).to(device)  # (1, T_o, 3, H, W)

                for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']:
                    vals = [o[key].astype(np.float32) for o in obs_list]
                    obs_dict[key] = torch.from_numpy(
                        np.stack(vals)
                    ).unsqueeze(0).to(device)

                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                actions_10d = result['action'][0].cpu().numpy()

                actions_7d = undo_rot6d(actions_10d)
                action_queue = list(actions_7d)

            action = action_queue.pop(0)
            obs_raw, reward, done, info = env.step(action)
            obs_history.append(obs_raw)
            step += 1

            if info.get("success", False):
                success = True
                break

        successes += int(success)
        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep+1}/{args.n_episodes}: {status} ({step} steps)")
        env.close()

    rate = successes / args.n_episodes
    print(f"\n{'='*50}")
    print(f"Bridge test: {successes}/{args.n_episodes} ({rate*100:.1f}%)")
    print(f"{'='*50}")
    if rate > 0.8:
        print("PASSED — our eval harness is validated")
    elif rate > 0:
        print("PARTIAL — harness works but performance gap vs Chi's eval")
    else:
        print("FAILED — env/action pipeline has a remaining bug")


if __name__ == "__main__":
    main()
W