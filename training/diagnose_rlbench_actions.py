"""Diagnose RLBench policy actions: compare predicted vs ground-truth.

Loads a checkpoint and runs inference on training demo observations,
comparing the predicted 8D actions to the stored GT actions.

Usage:
    PYTHONPATH=. python training/diagnose_rlbench_actions.py \
        --checkpoint checkpoints/v3_rlbench_open_drawer/epoch_139.pt \
        --stage1_checkpoint checkpoints/stage1_full_rtx5090/epoch_024.pt \
        --hdf5 /virtual/csc415user/data/rlbench/open_drawer.hdf5 \
        --ac_dim 10 --proprio_dim 8 --n_active_cams 4 --T_pred 10
"""

import argparse
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import h5py
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--ac_dim", type=int, default=10)
    parser.add_argument("--proprio_dim", type=int, default=8)
    parser.add_argument("--n_active_cams", type=int, default=4)
    parser.add_argument("--T_pred", type=int, default=10)
    parser.add_argument("--num_demos", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from data_pipeline.conversion.compute_norm_stats import load_norm_stats
    from data_pipeline.utils.rotation import convert_actions_quat_to_rot6d, convert_actions_rot6d_to_quat
    from models.policy_v3 import PolicyDiTv3
    from models.stage1_bridge import Stage1Bridge

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build model
    bridge = Stage1Bridge(checkpoint_path=args.stage1_checkpoint)
    policy = PolicyDiTv3(
        bridge=bridge, ac_dim=args.ac_dim,
        proprio_dim=args.proprio_dim, n_active_cams=args.n_active_cams,
        T_pred=args.T_pred,
    ).to(device)

    # Load EMA weights
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "ema" in ckpt and "averaged_model" in ckpt["ema"]:
        policy.load_state_dict(ckpt["ema"]["averaged_model"], strict=False)
        log.info("Loaded EMA weights")
    policy.eval()

    # Load norm stats
    norm_stats = load_norm_stats(args.hdf5)
    a_stats = norm_stats["actions"]
    p_stats = norm_stats["proprio"]

    log.info("Action stats: min=%s, max=%s",
             np.array2string(a_stats["min"], precision=3),
             np.array2string(a_stats["max"], precision=3))
    log.info("Proprio stats: min=%s, max=%s",
             np.array2string(p_stats["min"], precision=3),
             np.array2string(p_stats["max"], precision=3))

    # Load training demos
    with h5py.File(args.hdf5, "r") as f:
        demo_keys = [k.decode() for k in f["mask/train"][:args.num_demos]]

        for di, dk in enumerate(demo_keys):
            grp = f[f"data/{dk}"]
            gt_actions = grp["actions"][:]    # (T, 8) raw 8D
            proprio = grp["proprio"][:]       # (T, 8)
            images = grp["images"][:]         # (T, K, H, W, 3) uint8

            T = gt_actions.shape[0]
            log.info("\n=== Demo %d (%s): %d steps ===", di, dk, T)

            # Pick step 0 for diagnosis
            for step_idx in [0, T // 4, T // 2]:
                if step_idx >= T:
                    continue

                # Prepare images: uint8 HWC -> float32 CHW [0,1]
                imgs_t = images[step_idx].astype(np.float32) / 255.0  # (K, H, W, 3)
                imgs_t = np.moveaxis(imgs_t, -1, -3)  # (K, 3, H, W)

                # Observation horizon = 2: duplicate for padding
                imgs_seq = np.stack([imgs_t, imgs_t], axis=0)  # (2, K, 3, H, W)

                # Prepare proprio
                prop_t = proprio[step_idx]  # (8,)
                prop_norm = prop_t.copy()

                # Chi normalization for proprio
                p_min, p_max = p_stats["min"], p_stats["max"]
                pos_range = np.clip(p_max[:3] - p_min[:3], 1e-6, None)
                prop_norm[:3] = 2.0 * (prop_t[:3] - p_min[:3]) / pos_range - 1.0
                g_min, g_max = p_min[7:], p_max[7:]
                g_range = np.clip(g_max - g_min, 1e-6, None)
                prop_norm[7:] = 2.0 * (prop_t[7:] - g_min) / g_range - 1.0

                prop_seq = np.stack([prop_norm, prop_norm], axis=0)  # (2, 8)

                view_present = np.array([True, True, True, True])

                # Run policy inference
                with torch.no_grad():
                    obs = {
                        "images_enc": torch.from_numpy(imgs_seq).unsqueeze(0).to(device),
                        "proprio": torch.from_numpy(prop_seq).unsqueeze(0).float().to(device),
                        "view_present": torch.from_numpy(view_present).unsqueeze(0).to(device),
                    }
                    pred_norm = policy.predict_action(obs)  # (1, T_pred, 10)
                    pred_norm = pred_norm[0].cpu().numpy()  # (T_pred, 10)

                # Denormalize position
                pred_raw = pred_norm.copy()
                pos_min = a_stats["min"][:3]
                pos_max = a_stats["max"][:3]
                pred_raw[:, :3] = (pred_norm[:, :3] + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

                # Convert rot6d -> quat
                pred_8d = convert_actions_rot6d_to_quat(pred_raw)  # (T_pred, 8)

                # GT action at this step (and next few)
                gt_end = min(step_idx + args.T_pred, T)
                gt_chunk = gt_actions[step_idx:gt_end]  # (<=T_pred, 8)

                print(f"\n  Step {step_idx}:")
                print(f"  Proprio raw: pos={prop_t[:3]}, quat={prop_t[3:7]}, grip={prop_t[7:]}")
                print(f"  Proprio norm: pos={prop_norm[:3]}, grip={prop_norm[7:]}")

                # Compare first predicted action vs GT
                n_compare = min(3, len(gt_chunk))
                for j in range(n_compare):
                    p = pred_8d[j]
                    g = gt_chunk[j]
                    pos_err = np.linalg.norm(p[:3] - g[:3])
                    quat_dot = abs(np.dot(p[3:7], g[3:7]))
                    print(f"  t+{j+1}:")
                    print(f"    PRED: pos={p[:3]}, quat={p[3:7]}, grip={p[7]:.2f}")
                    print(f"    GT:   pos={g[:3]}, quat={g[3:7]}, grip={g[7]:.2f}")
                    print(f"    Err:  pos={pos_err:.4f}m, quat_align={quat_dot:.4f} (1.0=perfect)")

                # Check if predicted positions are in workspace
                pred_pos = pred_8d[:, :3]
                in_ws = (pred_pos[:, 2] > 0.5) & (pred_pos[:, 2] < 2.0)
                print(f"  Workspace check: {in_ws.sum()}/{len(in_ws)} in z=[0.5, 2.0]")
                print(f"  Pred pos range: x=[{pred_pos[:,0].min():.3f},{pred_pos[:,0].max():.3f}]"
                      f" z=[{pred_pos[:,2].min():.3f},{pred_pos[:,2].max():.3f}]")

                # Check quaternion norms
                quat_norms = np.linalg.norm(pred_8d[:, 3:7], axis=1)
                print(f"  Quat norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}")


if __name__ == "__main__":
    main()
