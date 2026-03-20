"""Tests for V3 training loop.

Validates optimizer config, EMA setup, checkpoint save/load,
and single training step with PolicyDiTv3.
"""

import os
import tempfile

import torch
import pytest

from models.policy_v3 import PolicyDiTv3
from models.stage1_bridge import Stage1Bridge
from training.train_v3 import V3Config, save_v3_checkpoint, load_v3_checkpoint
from training.train_stage3 import train_step

# Constants
B = 2
T_O = 2
T_P = 10  # Chi: horizon=10
K = 4
AC_DIM = 10
PROPRIO_DIM = 9
D_MODEL = 256


def _make_policy():
    bridge = Stage1Bridge(pretrained_encoder=False)
    return PolicyDiTv3(
        bridge=bridge, ac_dim=AC_DIM, proprio_dim=PROPRIO_DIM,
        d_model=D_MODEL, n_head=4, n_layers=2,
        T_obs=T_O, T_pred=T_P, num_views=K, n_active_cams=2,
        train_diffusion_steps=100, eval_diffusion_steps=10,
        p_drop_emb=0.0, p_drop_attn=0.3,
    )


def _make_batch(b=B):
    vp = torch.zeros(b, K, dtype=torch.bool)
    vp[:, 0] = True
    vp[:, 3] = True
    return {
        "cached_tokens": torch.randn(b, T_O, K, 196, 1024),
        "actions": torch.randn(b, T_P, AC_DIM),
        "proprio": torch.randn(b, T_O, PROPRIO_DIM),
        "view_present": vp,
    }


class TestV3OptimizerConfig:
    def test_optimizer_betas(self):
        """Optimizer uses Chi's transformer betas (0.9, 0.95)."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.lr, betas=config.betas,
        )
        assert optimizer.defaults["betas"] == (0.9, 0.95)

    def test_optimizer_param_groups(self):
        """Three param groups with correct weight decay."""
        policy = _make_policy()
        config = V3Config()
        param_groups = [
            {"params": list(policy.denoiser.parameters()),
             "weight_decay": config.weight_decay_denoiser},
            {"params": list(policy.obs_encoder.parameters()),
             "weight_decay": config.weight_decay_encoder},
            {"params": list(policy.bridge.adapter.parameters()),
             "weight_decay": config.weight_decay_encoder},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, betas=config.betas)

        assert len(optimizer.param_groups) == 3
        assert optimizer.param_groups[0]["weight_decay"] == 1e-3
        assert optimizer.param_groups[1]["weight_decay"] == 1e-6
        assert optimizer.param_groups[2]["weight_decay"] == 1e-6

    def test_config_defaults(self):
        """V3Config has correct defaults matching Chi's recipe."""
        config = V3Config()
        assert config.betas == (0.9, 0.95)
        assert config.weight_decay_denoiser == 1e-3
        assert config.weight_decay_encoder == 1e-6
        assert config.ema_power == 0.75
        assert config.use_rot6d is True
        assert config.ac_dim == 10
        assert config.d_model == 256
        assert config.n_layers == 8
        assert config.n_head == 4
        assert config.T_pred == 10
        assert config.pad_after == 7
        assert config.p_drop_attn == 0.3
        assert config.p_drop_emb == 0.0


class TestV3TrainStep:
    def test_single_step_finite_loss(self):
        """One training step produces finite loss."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

        batch = _make_batch()
        losses = train_step(batch, policy, optimizer, config, global_step=0)

        assert "total" in losses
        assert losses["total"] > 0
        assert not (losses["total"] != losses["total"])  # not NaN

    def test_parameters_update(self):
        """Denoiser parameters change after one step."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        # Snapshot weights
        w_before = policy.denoiser.head.weight.clone()

        batch = _make_batch()
        train_step(batch, policy, optimizer, config, global_step=0)

        w_after = policy.denoiser.head.weight
        assert not torch.equal(w_before, w_after), "Weights should change after step"


class TestV3EMA:
    def test_ema_adaptive_decay(self):
        """EMA decay increases with step count (power=0.75 schedule)."""
        from diffusers.training_utils import EMAModel

        policy = _make_policy()
        ema = EMAModel(policy.parameters(), power=0.75, max_value=0.9999)

        # Step 0: decay should be very low
        ema.step(policy.parameters())
        decay_early = ema.cur_decay_value

        # Step many times
        for _ in range(100):
            ema.step(policy.parameters())
        decay_late = ema.cur_decay_value

        assert decay_late > decay_early, \
            f"EMA decay should increase: early={decay_early:.4f}, late={decay_late:.4f}"
        assert decay_early < 0.5, f"Early decay {decay_early:.4f} should be < 0.5"


class TestV3TrainStepDetailed:
    def test_loss_is_mse_range(self):
        """Policy loss at init should be ~1.0 (unit Gaussian noise target)."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

        losses_list = []
        for _ in range(5):
            batch = _make_batch()
            losses = train_step(batch, policy, optimizer, config, global_step=0)
            losses_list.append(losses["policy"])
        avg = sum(losses_list) / len(losses_list)
        assert 0.1 < avg < 10.0, f"Unexpected loss range: {avg}"

    def test_loss_decreasing_overfit(self):
        """Loss decreases when overfitting on a single batch."""
        torch.manual_seed(42)
        policy = _make_policy()
        config = V3Config(lr=5e-3, grad_clip=1.0, warmup_steps=0)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-3)

        batch = _make_batch(b=4)
        losses_list = []
        for step in range(150):
            losses = train_step(batch, policy, optimizer, config, global_step=step)
            losses_list.append(losses["policy"])

        n = len(losses_list)
        avg_first = sum(losses_list[:n // 3]) / (n // 3)
        avg_last = sum(losses_list[-n // 3:]) / (n // 3)
        assert avg_last < avg_first, \
            f"Loss did not decrease: first={avg_first:.4f} -> last={avg_last:.4f}"

    def test_grad_clip_applied(self):
        """Gradient clipping at tight value doesn't crash."""
        policy = _make_policy()
        config = V3Config(grad_clip=0.01)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        batch = _make_batch()
        losses = train_step(batch, policy, optimizer, config, global_step=0)
        assert losses["total"] > 0


class TestV3GradientFlow:
    def test_encoder_frozen(self):
        """Encoder receives no gradients."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=1e-3
        )

        batch = _make_batch()
        train_step(batch, policy, optimizer, config, global_step=0)

        for name, p in policy.bridge.encoder.named_parameters():
            assert p.grad is None or torch.all(p.grad == 0), \
                f"Encoder param {name} should not receive gradients"

    def test_adapter_weights_change(self):
        """Adapter weights change after a training step."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=1e-2
        )

        old_w = [p.data.clone() for p in policy.bridge.adapter.parameters()]
        batch = _make_batch()
        train_step(batch, policy, optimizer, config, global_step=0)

        changed = any(
            not torch.equal(old, p.data)
            for old, p in zip(old_w, policy.bridge.adapter.parameters())
        )
        assert changed, "Adapter weights should change"

    def test_denoiser_weights_change(self):
        """Denoiser weights change after a training step."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=1e-2
        )

        old_w = [p.data.clone() for p in policy.denoiser.parameters()
                 if p.requires_grad]
        batch = _make_batch()
        train_step(batch, policy, optimizer, config, global_step=0)

        n_changed = sum(
            1 for old, p in zip(
                old_w,
                (p for p in policy.denoiser.parameters() if p.requires_grad)
            )
            if not torch.equal(old, p.data)
        )
        assert n_changed > 0, "Some denoiser weights should change"

    def test_obs_encoder_weights_change(self):
        """ObservationEncoder weights change after a training step."""
        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=1e-2
        )

        old_w = [p.data.clone() for p in policy.obs_encoder.parameters()]
        batch = _make_batch()
        train_step(batch, policy, optimizer, config, global_step=0)

        changed = any(
            not torch.equal(old, p.data)
            for old, p in zip(old_w, policy.obs_encoder.parameters())
        )
        assert changed, "ObservationEncoder weights should change"


class TestV3DDIMInference:
    def test_predict_action_deterministic(self):
        """Same seed produces same actions from predict_action."""
        policy = _make_policy()
        policy.eval()
        obs = _make_batch()

        torch.manual_seed(42)
        a1 = policy.predict_action(obs)
        torch.manual_seed(42)
        a2 = policy.predict_action(obs)
        assert torch.allclose(a1, a2)

    def test_predict_action_different_eval_steps(self):
        """predict_action works with various DDIM step counts."""
        for steps in [5, 10, 50]:
            policy = _make_policy()
            policy.eval_diffusion_steps = steps
            obs = _make_batch()
            actions = policy.predict_action(obs)
            assert actions.shape == (B, T_P, AC_DIM)
            assert torch.isfinite(actions).all()


class TestV3LRScheduler:
    def test_warmup(self):
        """LR increases linearly during warmup."""
        from training.train_stage3 import _create_lr_scheduler
        import torch.nn as nn

        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = _create_lr_scheduler(optimizer, warmup_steps=100, total_steps=1000)

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            sched.step()
        assert lrs[-1] > lrs[0], "LR should increase during warmup"

    def test_cosine_decay(self):
        """LR decays after warmup."""
        from training.train_stage3 import _create_lr_scheduler
        import torch.nn as nn

        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = _create_lr_scheduler(optimizer, warmup_steps=10, total_steps=100)

        for _ in range(10):
            sched.step()
        lr_post_warmup = optimizer.param_groups[0]["lr"]

        for _ in range(80):
            sched.step()
        lr_end = optimizer.param_groups[0]["lr"]
        assert lr_end < lr_post_warmup


class TestV3NoiseScheduler:
    def test_add_noise_finite(self):
        """Adding noise produces finite result."""
        from diffusers import DDPMScheduler

        sched = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )
        actions = torch.randn(B, T_P, AC_DIM)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (B,))
        noisy = sched.add_noise(actions, noise, timesteps)
        assert noisy.shape == actions.shape
        assert torch.isfinite(noisy).all()

    def test_clip_sample_enabled(self):
        """V3 scheduler has clip_sample=True."""
        policy = _make_policy()
        assert policy.noise_scheduler.config.clip_sample is True


class TestV3FullPipeline:
    def _make_v3_hdf5(self, path):
        """Create synthetic HDF5 with 10D norm stats for V3."""
        import h5py
        import numpy as np

        with h5py.File(path, "w") as f:
            f.attrs["benchmark"] = "robomimic"
            f.attrs["task"] = "lift"
            f.attrs["action_dim"] = 7  # stored as 7D
            f.attrs["proprio_dim"] = PROPRIO_DIM
            f.attrs["image_size"] = 224
            f.attrs["num_cam_slots"] = K

            vp = np.array([True, False, False, True], dtype=bool)  # robomimic: slots 0,3
            keys = []
            for i in range(2):
                key = f"demo_{i}"
                keys.append(key)
                grp = f.create_group(f"data/{key}")
                grp.create_dataset("images", data=np.random.randint(
                    0, 256, (20, K, 224, 224, 3), dtype=np.uint8))
                # 7D abs actions with realistic values
                pos = np.random.uniform(-0.5, 0.5, (20, 3)).astype(np.float32)
                rot = np.random.uniform(-0.3, 0.3, (20, 3)).astype(np.float32)
                grip = np.random.choice([-1.0, 1.0], (20, 1)).astype(np.float32)
                grp.create_dataset("actions", data=np.concatenate([pos, rot, grip], axis=-1))
                grp.create_dataset("proprio", data=np.random.randn(
                    20, PROPRIO_DIM).astype(np.float32))
                grp.create_dataset("view_present", data=vp)

            mask = f.create_group("mask")
            dt = h5py.special_dtype(vlen=str)
            mask.create_dataset("train", data=keys, dtype=dt)
            mask.create_dataset("valid", data=keys[:1], dtype=dt)

            # 10D norm stats (rot6d-converted)
            from data_pipeline.utils.rotation import convert_actions_to_rot6d
            all_acts = []
            for key in keys:
                all_acts.append(convert_actions_to_rot6d(f[f"data/{key}/actions"][:]))
            all_acts = np.concatenate(all_acts, axis=0)

            ns = f.create_group("norm_stats")
            ag = ns.create_group("actions")
            ag.create_dataset("mean", data=all_acts.mean(0).astype(np.float32))
            ag.create_dataset("std", data=np.clip(all_acts.std(0), 1e-6, None).astype(np.float32))
            ag.create_dataset("min", data=all_acts.min(0).astype(np.float32))
            ag.create_dataset("max", data=all_acts.max(0).astype(np.float32))

            pg = ns.create_group("proprio")
            pg.create_dataset("mean", data=np.zeros(PROPRIO_DIM, dtype=np.float32))
            pg.create_dataset("std", data=np.ones(PROPRIO_DIM, dtype=np.float32))
            pg.create_dataset("min", data=-np.ones(PROPRIO_DIM, dtype=np.float32) * 2)
            pg.create_dataset("max", data=np.ones(PROPRIO_DIM, dtype=np.float32) * 2)

    def test_train_step_with_real_dataset(self, tmp_path):
        """Full pipeline: Stage3Dataset(use_rot6d=True) → batch → train_step."""
        from data_pipeline.datasets.stage3_dataset import Stage3Dataset

        path = str(tmp_path / "v3_test.hdf5")
        self._make_v3_hdf5(path)

        ds = Stage3Dataset(path, T_obs=T_O, T_pred=T_P, use_rot6d=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        batch = next(iter(loader))

        # Verify 10D actions from dataset
        assert batch["actions"].shape[-1] == 10

        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

        losses = train_step(batch, policy, optimizer, config, global_step=0)
        assert losses["total"] > 0
        assert not (losses["total"] != losses["total"])  # not NaN

    def test_multi_step_training(self, tmp_path):
        """Multiple training steps with EMA + LR scheduler."""
        from diffusers.training_utils import EMAModel
        from data_pipeline.datasets.stage3_dataset import Stage3Dataset
        from training.train_stage3 import _create_lr_scheduler

        path = str(tmp_path / "v3_multi.hdf5")
        self._make_v3_hdf5(path)

        ds = Stage3Dataset(path, T_obs=T_O, T_pred=T_P, use_rot6d=True)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)

        policy = _make_policy()
        config = V3Config()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        ema = EMAModel(policy.parameters(), power=0.75)
        sched = _create_lr_scheduler(optimizer, warmup_steps=2, total_steps=10)

        for i, batch in enumerate(loader):
            if i >= 3:
                break
            losses = train_step(batch, policy, optimizer, config, global_step=i)
            ema.step(policy.parameters())
            sched.step()
            assert losses["total"] > 0


class TestV3Checkpoint:
    def test_save_load_roundtrip(self):
        """Checkpoint save/load preserves policy state."""
        policy1 = _make_policy()
        optimizer1 = torch.optim.AdamW(policy1.parameters(), lr=1e-4)

        # Do one step so optimizer has state
        batch = _make_batch()
        from training.train_v3 import V3Config
        train_step(batch, policy1, optimizer1, V3Config(), global_step=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            save_v3_checkpoint(path, 5, 100, policy1, optimizer1, None, {"loss": 0.1})

            # Load into fresh policy
            policy2 = _make_policy()
            optimizer2 = torch.optim.AdamW(policy2.parameters(), lr=1e-4)
            start_epoch, global_step = load_v3_checkpoint(path, policy2, optimizer2)

            assert start_epoch == 6  # epoch + 1
            assert global_step == 100

            # Trainable weights should match (skip frozen encoder)
            for (n1, p1), (n2, p2) in zip(
                policy1.named_parameters(), policy2.named_parameters()
            ):
                if p1.requires_grad:
                    assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_save_load_with_ema(self):
        """EMA state survives checkpoint roundtrip."""
        from diffusers.training_utils import EMAModel

        policy1 = _make_policy()
        ema1 = EMAModel(policy1.parameters(), power=0.75)
        optimizer1 = torch.optim.AdamW(policy1.parameters(), lr=1e-4)

        # Step EMA a few times
        for _ in range(10):
            ema1.step(policy1.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_ema.pt")
            save_v3_checkpoint(path, 3, 50, policy1, optimizer1, ema1, {})

            policy2 = _make_policy()
            ema2 = EMAModel(policy2.parameters(), power=0.75)
            optimizer2 = torch.optim.AdamW(policy2.parameters(), lr=1e-4)
            load_v3_checkpoint(path, policy2, optimizer2, ema2)

            # EMA shadow params should match
            for s1, s2 in zip(ema1.shadow_params, ema2.shadow_params):
                assert torch.equal(s1, s2)
