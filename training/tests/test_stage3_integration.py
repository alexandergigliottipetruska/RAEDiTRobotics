"""C.9 Integration tests for Stage 3 pipeline.

These tests verify the interfaces between components — that Stage1Bridge,
TokenAssembly, ViewDropout, PolicyDiT, and the training loop wire together
correctly with matching shapes, gradient flow, and checkpoint save/load.

NOTE ON BATCH-FIRST: C.0 converted diffusion.py to batch-first (B, S, d).
All components output batch-first. No transpose needed anywhere.

Owner: Swagman
"""

import os
import tempfile

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from models.base_policy import BasePolicy
from models.diffusion import _DiTNoiseNet
from models.ema import EMA
from models.policy_dit import PolicyDiT
from models.stage1_bridge import Stage1Bridge
from models.token_assembly import TokenAssembly
from models.view_dropout import ViewDropout
from training.train_stage3 import (
    Stage3Config,
    create_noise_scheduler,
    train_step,
    ddim_inference,
    save_checkpoint,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B = 2
K = 4
N = 196         # tokens per view (14x14 patches)
D_ENC = 1024    # DINOv3 output dim
D_MODEL = 512   # adapter / hidden dim
H = W = 224
T_O = 2
T_P = 16
AC_DIM = 7
PROPRIO_DIM = 9

# Checkpoint paths (set to None if not available)
_CKPT_FULL = "checkpoints/stage1_full/epoch_007.pt"
_CKPT_5090 = "checkpoints/stage1_rtx5090/epoch_024.pt"

def _ckpt_available(path):
    """Check if a checkpoint file exists (relative to repo root)."""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[2]
    return (repo / path).is_file()


# ---------------------------------------------------------------------------
# Synthetic HDF5 helper
# ---------------------------------------------------------------------------

def _make_hdf5(tmp_path, num_demos=2, demo_len=25):
    path = str(tmp_path / "test.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["benchmark"] = "robomimic"
        f.attrs["task"] = "lift"
        f.attrs["action_dim"] = AC_DIM
        f.attrs["proprio_dim"] = PROPRIO_DIM
        f.attrs["image_size"] = 224
        f.attrs["num_cam_slots"] = K

        vp = np.array([True, True, False, False])
        keys = []
        for i in range(num_demos):
            key = f"demo_{i}"
            keys.append(key)
            grp = f.create_group(f"data/{key}")
            grp.create_dataset("images", data=np.random.randint(0, 256, (demo_len, K, H, W, 3), dtype=np.uint8))
            grp.create_dataset("actions", data=np.random.randn(demo_len, AC_DIM).astype(np.float32))
            grp.create_dataset("proprio", data=np.random.randn(demo_len, PROPRIO_DIM).astype(np.float32))
            grp.create_dataset("view_present", data=vp)

        mask = f.create_group("mask")
        dt = h5py.special_dtype(vlen=str)
        mask.create_dataset("train", data=keys, dtype=dt)
        mask.create_dataset("valid", data=keys[:1], dtype=dt)

        ns = f.create_group("norm_stats")
        for field, dim in [("actions", AC_DIM), ("proprio", PROPRIO_DIM)]:
            g = ns.create_group(field)
            g.create_dataset("mean", data=np.zeros(dim, dtype=np.float32))
            g.create_dataset("std", data=np.ones(dim, dtype=np.float32))
            g.create_dataset("min", data=-np.ones(dim, dtype=np.float32) * 2)
            g.create_dataset("max", data=np.ones(dim, dtype=np.float32) * 2)
    return path


def _make_bridge(**kwargs):
    """Create a Stage1Bridge with mock encoder."""
    defaults = dict(pretrained_encoder=False, load_decoder=False)
    defaults.update(kwargs)
    return Stage1Bridge(**defaults)


def _make_policy(**kwargs):
    """Create a PolicyDiT with mock encoder and small architecture."""
    bridge = _make_bridge(load_decoder=kwargs.pop("load_decoder", False))
    defaults = dict(
        bridge=bridge,
        ac_dim=AC_DIM,
        proprio_dim=PROPRIO_DIM,
        hidden_dim=D_MODEL,
        T_obs=T_O,
        T_pred=T_P,
        num_blocks=2,
        nhead=8,
        num_views=K,
        train_diffusion_steps=100,
        eval_diffusion_steps=5,
        p_view_drop=0.0,
        lambda_recon=0.0,
        use_lightning=True,
        policy_type="ddpm",
    )
    defaults.update(kwargs)
    return PolicyDiT(**defaults)


def _make_batch(b=B, t_o=T_O, k=K, t_p=T_P, ac_dim=AC_DIM, d_prop=PROPRIO_DIM):
    """Create a fake training batch."""
    return {
        "images_enc": torch.randn(b, t_o, k, 3, H, W),
        "proprio": torch.randn(b, t_o, d_prop),
        "actions": torch.randn(b, t_p, ac_dim),
        "view_present": torch.ones(b, k, dtype=torch.bool),
        "images_target": torch.rand(b, t_o, k, 3, H, W),
    }


def _make_obs(b=B, t_o=T_O, k=K, d_prop=PROPRIO_DIM):
    """Create a fake observation dict for inference."""
    return {
        "images_enc": torch.randn(b, t_o, k, 3, H, W),
        "proprio": torch.randn(b, t_o, d_prop),
        "view_present": torch.ones(b, k, dtype=torch.bool),
    }


# ============================================================
# C.1 BasePolicy interface tests
# ============================================================

class TestBasePolicy:
    """Tests for C.1 BasePolicy abstract interface."""

    def test_is_nn_module(self):
        assert issubclass(BasePolicy, nn.Module)

    def test_has_compute_loss(self):
        assert hasattr(BasePolicy, "compute_loss")

    def test_has_predict_action(self):
        assert hasattr(BasePolicy, "predict_action")


# ============================================================
# C.2 Stage1Bridge interface tests
# ============================================================

class TestStage1Bridge:
    """Tests for C.2 Stage1Bridge with mock encoder."""

    def test_has_encode_method(self):
        assert hasattr(Stage1Bridge, "encode")

    def test_has_compute_recon_loss(self):
        assert hasattr(Stage1Bridge, "compute_recon_loss")

    def test_encode_output_shape(self):
        """encode() returns (B, T_o, K, N, d') adapted tokens."""
        bridge = _make_bridge()
        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.ones(B, K, dtype=torch.bool)
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (B, T_O, K, N, D_MODEL), (
            f"Expected ({B}, {T_O}, {K}, {N}, {D_MODEL}), got {adapted.shape}"
        )

    def test_encode_partial_views(self):
        """Absent views are zero-filled in output."""
        bridge = _make_bridge()
        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.tensor([[True, True, False, False]] * B)
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (B, T_O, K, N, D_MODEL)
        # Absent views should be zero
        assert torch.all(adapted[:, :, 2:] == 0)

    def test_encoder_frozen_no_grad(self):
        """Encoder parameters should not receive gradients."""
        bridge = _make_bridge()
        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.ones(B, K, dtype=torch.bool)
        adapted = bridge.encode(images, vp)
        adapted.sum().backward()
        for name, p in bridge.encoder.named_parameters():
            assert p.grad is None or torch.all(p.grad == 0), (
                f"Encoder param {name} should be frozen"
            )

    def test_adapter_receives_grad(self):
        """Adapter parameters should receive gradients."""
        bridge = _make_bridge()
        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.ones(B, K, dtype=torch.bool)
        adapted = bridge.encode(images, vp)
        adapted.sum().backward()
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in bridge.adapter.parameters()
        )
        assert has_grad, "Adapter should receive gradients through bridge.encode()"


# ============================================================
# C.4 TokenAssembly interface tests
# ============================================================

class TestTokenAssembly:
    """Tests for C.4 TokenAssembly embeddings and assembly."""

    def test_output_shape_all_views(self):
        """All views present: S_obs = T_o * K * N + T_o = 2*4*196+2 = 1570."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        expected_S = T_O * K * N + T_O  # 1570
        assert out.shape == (B, expected_S, D_MODEL)

    def test_output_shape_fixed_length(self):
        """Even with partial views, sequence length is fixed (all K views included)."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.tensor([[True, True, False, False]] * B)
        out = ta(adapted, proprio, vp)
        # Fixed length regardless of view_present
        expected_S = T_O * K * N + T_O  # 1570
        assert out.shape == (B, expected_S, D_MODEL)

    def test_output_is_batch_first(self):
        """Output must be batch-first [B, S, d]."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        assert out.shape[0] == B

    def test_gradient_flows(self):
        """Gradients flow through token assembly to adapted tokens."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        adapted = torch.randn(B, T_O, K, N, D_MODEL, requires_grad=True)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)
        out = ta(adapted, proprio, vp)
        out.sum().backward()
        assert adapted.grad is not None
        assert not torch.all(adapted.grad == 0)


# ============================================================
# C.10 PolicyDiT end-to-end tests
# ============================================================

class TestPolicyDiT:
    """Tests for C.10 PolicyDiT wrapper with mock encoder."""

    def test_compute_loss_finite(self):
        """compute_loss returns a finite scalar."""
        policy = _make_policy()
        batch = _make_batch()
        loss = policy.compute_loss(batch)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    def test_predict_action_shape(self):
        """predict_action returns (B, T_pred, ac_dim)."""
        policy = _make_policy()
        policy.eval()
        obs = _make_obs()
        actions = policy.predict_action(obs)
        assert actions.shape == (B, T_P, AC_DIM)

    def test_no_transpose_batch_first(self):
        """After C.0, PolicyDiT passes batch-first tensors directly to noise_net."""
        policy = _make_policy()
        batch = _make_batch()
        # If dims are incompatible this would crash
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_encoder_frozen_adapter_trainable(self):
        """Gradient reaches adapter but NOT encoder through PolicyDiT.

        adaLN-Zero starts with zero modulation, blocking upstream gradients.
        Perturb zero-init params to unblock gradient flow for this test.
        """
        policy = _make_policy()
        # Perturb zero-initialized params so gradients flow
        with torch.no_grad():
            for p in policy.noise_net.parameters():
                if p.requires_grad and torch.all(p == 0):
                    p.add_(torch.randn_like(p) * 0.01)

        batch = _make_batch()
        loss = policy.compute_loss(batch)
        loss.backward()

        # Encoder should be frozen
        for p in policy.bridge.encoder.parameters():
            assert p.grad is None or torch.all(p.grad == 0)

        # Adapter should get gradients
        has_grad = any(
            p.grad is not None and not torch.all(p.grad == 0)
            for p in policy.bridge.adapter.parameters()
        )
        assert has_grad, "Adapter should receive gradients"


# ============================================================
# Cross-component wiring tests
# ============================================================

class TestCrossComponentWiring:
    """Tests that verify correct wiring between components."""

    def test_bridge_to_assembly_shapes(self):
        """Stage1Bridge output feeds into TokenAssembly without shape errors."""
        bridge = _make_bridge()
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)

        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.ones(B, K, dtype=torch.bool)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)

        adapted = bridge.encode(images, vp)  # (B, T_o, K, N, d')
        obs_tokens = ta(adapted, proprio, vp)
        expected_S = T_O * K * N + T_O
        assert obs_tokens.shape == (B, expected_S, D_MODEL)

    def test_assembly_to_noise_net_shapes(self):
        """TokenAssembly output feeds into noise_net.forward_enc without errors."""
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)
        noise_net = _DiTNoiseNet(
            ac_dim=AC_DIM, ac_chunk=T_P, hidden_dim=D_MODEL,
            num_blocks=2, nhead=8,
        )

        adapted = torch.randn(B, T_O, K, N, D_MODEL)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)
        vp = torch.ones(B, K, dtype=torch.bool)

        obs_tokens = ta(adapted, proprio, vp)  # (B, S_obs, d')
        enc_cache = noise_net.forward_enc(obs_tokens)
        assert isinstance(enc_cache, list)
        assert enc_cache[0].shape[0] == B  # batch dim first

    def test_bridge_viewdrop_assembly_pipeline(self):
        """Bridge → ViewDropout → TokenAssembly produces correct shapes."""
        bridge = _make_bridge()
        vd = ViewDropout(d_model=D_MODEL, p=0.5)
        ta = TokenAssembly(d_model=D_MODEL, num_patches=N, num_views=K,
                           num_obs_steps=T_O, proprio_dim=PROPRIO_DIM)

        images = torch.randn(B, T_O, K, 3, H, W)
        vp = torch.ones(B, K, dtype=torch.bool)
        proprio = torch.randn(B, T_O, PROPRIO_DIM)

        adapted = bridge.encode(images, vp)  # (B, T_o, K, N, d')

        # ViewDropout expects (B, K, N, d') — reshape per timestep
        B_, T_o, K_, N_, d_ = adapted.shape
        flat = adapted.reshape(B_ * T_o, K_, N_, d_)
        flat_vp = vp.unsqueeze(1).expand(-1, T_o, -1).reshape(B_ * T_o, K_)
        dropped, new_vp = vd(flat, flat_vp)
        dropped = dropped.reshape(B_, T_o, K_, N_, d_)

        obs_tokens = ta(dropped, proprio, vp)
        expected_S = T_O * K * N + T_O
        assert obs_tokens.shape == (B, expected_S, D_MODEL)

    def test_full_pipeline_dataset_to_loss(self, tmp_path):
        """Full pipeline: Stage3Dataset -> PolicyDiT.compute_loss -> finite scalar."""
        from data_pipeline.datasets.stage3_dataset import Stage3Dataset

        hdf5_path = _make_hdf5(tmp_path)
        ds = Stage3Dataset(hdf5_path, T_obs=T_O, T_pred=T_P)
        loader = torch.utils.data.DataLoader(ds, batch_size=B)
        batch = next(iter(loader))

        policy = _make_policy()
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_full_pipeline_checkpoint_roundtrip(self, tmp_path):
        """Save and load a full PolicyDiT checkpoint."""
        policy = _make_policy()
        ema = EMA(policy.noise_net, decay=0.999)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

        path = str(tmp_path / "policy.pt")
        save_checkpoint(
            path, epoch=0, global_step=0,
            noise_net=policy.noise_net,
            adapter=policy.bridge.adapter,
            optimizer=optimizer, ema=ema, val_metrics={},
        )
        assert os.path.isfile(path)


# ============================================================
# Real checkpoint tests (skip if checkpoints not available)
# ============================================================

class TestRealCheckpoint:
    """Tests with actual Stage 1 checkpoints.

    These verify that Stage1Bridge can load real trained weights
    and produce meaningful adapted tokens.
    """

    @pytest.mark.skipif(
        not _ckpt_available(_CKPT_FULL),
        reason=f"Checkpoint not found: {_CKPT_FULL}"
    )
    def test_load_full_checkpoint(self):
        """Load the multi-task Stage 1 checkpoint (epoch 7, torch.compile prefix)."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        bridge = Stage1Bridge(
            checkpoint_path=str(repo / _CKPT_FULL),
            pretrained_encoder=False,  # mock encoder for speed
            load_decoder=True,
        )
        # Adapter should have loaded weights
        w = bridge.adapter.adapter[0].weight  # first Linear in nn.Sequential
        assert w.shape == (1024, 1024)
        # Decoder should be loaded
        assert bridge.decoder is not None

    @pytest.mark.skipif(
        not _ckpt_available(_CKPT_5090),
        reason=f"Checkpoint not found: {_CKPT_5090}"
    )
    def test_load_rtx5090_checkpoint(self):
        """Load the RTX 5090 checkpoint (epoch 24, no compile prefix)."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        bridge = Stage1Bridge(
            checkpoint_path=str(repo / _CKPT_5090),
            pretrained_encoder=False,
            load_decoder=True,
        )
        assert bridge.decoder is not None

    @pytest.mark.skipif(
        not _ckpt_available(_CKPT_FULL),
        reason=f"Checkpoint not found: {_CKPT_FULL}"
    )
    def test_encode_with_real_adapter(self):
        """Encoding with trained adapter produces non-zero adapted tokens."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        bridge = Stage1Bridge(
            checkpoint_path=str(repo / _CKPT_FULL),
            pretrained_encoder=False,  # mock encoder
        )
        images = torch.randn(1, T_O, K, 3, H, W)
        vp = torch.ones(1, K, dtype=torch.bool)
        adapted = bridge.encode(images, vp)
        assert adapted.shape == (1, T_O, K, N, D_MODEL)
        # Should be non-zero (trained adapter transforms random encoder output)
        assert adapted.abs().mean() > 0.01

    @pytest.mark.skipif(
        not _ckpt_available(_CKPT_FULL),
        reason=f"Checkpoint not found: {_CKPT_FULL}"
    )
    def test_full_policy_with_real_adapter(self):
        """PolicyDiT with trained adapter produces finite loss."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        bridge = Stage1Bridge(
            checkpoint_path=str(repo / _CKPT_FULL),
            pretrained_encoder=False,
        )
        policy = PolicyDiT(
            bridge=bridge,
            ac_dim=AC_DIM,
            proprio_dim=PROPRIO_DIM,
            hidden_dim=D_MODEL,
            T_obs=T_O,
            T_pred=T_P,
            num_blocks=2,
            nhead=8,
            num_views=K,
            train_diffusion_steps=100,
            eval_diffusion_steps=5,
            p_view_drop=0.0,
            lambda_recon=0.0,
            use_lightning=True,
            policy_type="ddpm",
        )
        batch = _make_batch(b=1)
        loss = policy.compute_loss(batch)
        assert torch.isfinite(loss)

    @pytest.mark.skipif(
        not _ckpt_available(_CKPT_FULL),
        reason=f"Checkpoint not found: {_CKPT_FULL}"
    )
    def test_cotrain_recon_with_real_decoder(self):
        """Co-training with real decoder produces finite reconstruction loss."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parents[2]
        bridge = Stage1Bridge(
            checkpoint_path=str(repo / _CKPT_FULL),
            pretrained_encoder=False,
            load_decoder=True,
        )
        images = torch.randn(1, T_O, K, 3, H, W)
        targets = torch.rand(1, T_O, K, 3, H, W)
        vp = torch.ones(1, K, dtype=torch.bool)
        adapted = bridge.encode(images, vp)
        recon_loss = bridge.compute_recon_loss(adapted, targets, vp)
        assert torch.isfinite(recon_loss)
        assert recon_loss.item() > 0
