"""PolicyDiT — C.10.

Top-level policy module that composes Stage1Bridge + ViewDropout +
TokenAssembly + _DiTNoiseNetV2 into the BasePolicy interface.

Architecture (Dasari et al. 2024 + Chi et al. 2023 + Zheng et al. 2025):
  - DINOv3 adapted tokens (196 patches, 512-dim) from Stage 1
  - Spatial pooling: 14x14 → 7x7 = 49 patches per view
  - Linear projection: 512 → hidden_dim (default 256)
  - ViewDropout on pooled tokens
  - TokenAssembly: spatial + view + timestep embeddings + proprio
  - Self-attention encoder (4 layers)
  - adaLN-Zero decoder (4 layers, NO cross-attention, Dasari-style)
  - Conditioning: mean_pool(encoder_output[d]) + time_embedding

Supports two policy types:
  - "ddpm":          DDPM training + DDIM inference (Chi et al. 2023)
  - "flow_matching":  Flow matching training + Euler inference (Lipman et al. 2023, pi0)

Training:  policy.compute_loss(batch) -> scalar loss
Inference: policy.predict_action(obs) -> (B, T_p, D_act)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.base_policy import BasePolicy
from models.diffusion import _DiTNoiseNetV2
from models.stage1_bridge import Stage1Bridge
from models.token_assembly import TokenAssembly
from models.view_dropout import ViewDropout


# Adapter output dimension (fixed by Stage 1 TrainableAdapter)
_ADAPTER_DIM = 512
# DINOv3 spatial grid: 14x14 = 196 patches
_PATCH_GRID = 14
# Pooled spatial grid: 7x7 = 49 patches (matches Dasari's ResNet-26 output)
_POOL_GRID = 7
_POOL_PATCHES = _POOL_GRID * _POOL_GRID  # 49


class PolicyDiT(BasePolicy):
    """Full Policy DiT: Stage1Bridge + SpatialPool + TokenAssembly + DiTNoiseNetV2.

    Args:
        bridge: Stage1Bridge instance (encoder + adapter + optional decoder).
        ac_dim: Action dimension (7 for robomimic, 8 for RLBench).
        proprio_dim: Proprioceptive state dimension.
        hidden_dim: Transformer hidden dimension (default 256, matching Chi).
        T_obs: Observation horizon.
        T_pred: Prediction horizon (action chunk length).
        num_blocks: Number of transformer layers (encoder and decoder).
        nhead: Number of attention heads.
        num_views: Maximum camera slots.
        train_diffusion_steps: DDPM training steps.
        eval_diffusion_steps: DDIM inference steps (default 100, matching Chi).
        p_view_drop: View dropout probability.
        lambda_recon: Reconstruction co-training weight (0 = no co-training).
        policy_type: "ddpm" or "flow_matching".
        fm_timestep_dist: "uniform" or "beta" (pi0 distribution).
        fm_timestep_scale: Scale tau before time network (default 1000).
        fm_beta_a: Beta distribution alpha parameter.
        fm_beta_b: Beta distribution beta parameter.
        fm_cutoff: Maximum tau for pi0 distribution (s=0.999).
        num_flow_steps: Euler integration steps for flow matching inference.
    """

    def __init__(
        self,
        bridge: Stage1Bridge,
        ac_dim: int = 7,
        proprio_dim: int = 9,
        hidden_dim: int = 256,
        T_obs: int = 2,
        T_pred: int = 16,
        num_blocks: int = 4,
        nhead: int = 8,
        num_views: int = 4,
        train_diffusion_steps: int = 100,
        eval_diffusion_steps: int = 100,
        p_view_drop: float = 0.0,
        lambda_recon: float = 0.0,
        policy_type: str = "ddpm",
        fm_timestep_dist: str = "beta",
        fm_timestep_scale: float = 1000.0,
        fm_beta_a: float = 1.5,
        fm_beta_b: float = 1.0,
        fm_cutoff: float = 0.999,
        num_flow_steps: int = 10,
    ):
        super().__init__()

        assert policy_type in ("ddpm", "flow_matching"), (
            f"policy_type must be 'ddpm' or 'flow_matching', got '{policy_type}'"
        )

        self.bridge = bridge
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.ac_dim = ac_dim
        self.lambda_recon = lambda_recon
        self._train_steps = train_diffusion_steps
        self._eval_steps = eval_diffusion_steps
        self.policy_type = policy_type

        # Flow matching config
        self.fm_timestep_dist = fm_timestep_dist
        self.fm_timestep_scale = fm_timestep_scale
        self.fm_beta_a = fm_beta_a
        self.fm_beta_b = fm_beta_b
        self.fm_cutoff = fm_cutoff
        self.num_flow_steps = num_flow_steps

        # Spatial pooling: 14x14 → 7x7 (no params)
        self.spatial_pool = nn.AdaptiveAvgPool2d((_POOL_GRID, _POOL_GRID))

        # Project adapter output (512) to hidden_dim (256)
        self.obs_proj = nn.Linear(_ADAPTER_DIM, hidden_dim)

        # View dropout (at hidden_dim, on pooled tokens)
        self.view_dropout = ViewDropout(d_model=hidden_dim, p=p_view_drop)

        # Token assembly (49 patches per view, at hidden_dim)
        self.token_assembly = TokenAssembly(
            d_model=hidden_dim,
            num_patches=_POOL_PATCHES,
            num_views=num_views,
            num_obs_steps=T_obs,
            proprio_dim=proprio_dim,
        )

        # Dasari-style noise net: adaLN-Zero decoder, no cross-attention
        self.noise_net = _DiTNoiseNetV2(
            ac_dim=ac_dim,
            ac_chunk=T_pred,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            nhead=nhead,
            dropout=0.0,
            dim_feedforward=hidden_dim * 4,
        )

        # DDIM scheduler (only used in DDPM mode)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _encode_and_assemble(
        self, batch, proprio, view_present
    ):
        """Encode images (or use cached tokens), pool, project, assemble.

        Pipeline:
          adapter output (B, T_o, K, 196, 512)
          → spatial pool 14x14 → 7x7  (B, T_o, K, 49, 512)
          → linear projection         (B, T_o, K, 49, hidden_dim)
          → view dropout
          → token assembly + embeddings

        Args:
            batch: Full batch dict.
            proprio: (B, T_o, D_prop) normalized.
            view_present: (B, K) bool.

        Returns:
            obs_tokens: (B, S_obs, hidden_dim) for the encoder.
            adapted_clean: (B, T_o, K, 49, hidden_dim) for optional recon loss.
            view_present_after: (B, K) after dropout.
        """
        if "cached_tokens" in batch:
            adapted = self.bridge.adapt(batch["cached_tokens"], view_present)
        else:
            adapted = self.bridge.encode(batch["images_enc"], view_present)
        # adapted: (B, T_o, K, 196, 512)

        B, T_o, K, _, d = adapted.shape  # _=196 patches, d=512

        # Spatial pooling: reshape to (B*T_o*K, 512, 14, 14) for pool2d
        adapted = adapted.reshape(B * T_o * K, _PATCH_GRID, _PATCH_GRID, d)
        adapted = adapted.permute(0, 3, 1, 2)  # (*, 512, 14, 14)
        adapted = self.spatial_pool(adapted)     # (*, 512, 7, 7)
        adapted = adapted.permute(0, 2, 3, 1)   # (*, 7, 7, 512)
        adapted = adapted.reshape(B * T_o * K, _POOL_PATCHES, d)  # (*, 49, 512)

        # Project 512 → hidden_dim
        adapted = self.obs_proj(adapted)  # (*, 49, hidden_dim)
        adapted = adapted.reshape(B, T_o, K, _POOL_PATCHES, -1)

        adapted_clean = adapted  # save pre-dropout for recon loss

        # View dropout
        hd = adapted.shape[-1]
        adapted_flat = adapted.reshape(B * T_o, K, _POOL_PATCHES, hd)
        vp_flat = view_present.unsqueeze(1).expand(B, T_o, K).reshape(B * T_o, K)
        adapted_flat, vp_after_flat = self.view_dropout(adapted_flat, vp_flat)
        adapted = adapted_flat.reshape(B, T_o, K, _POOL_PATCHES, hd)
        vp_after = vp_after_flat.reshape(B, T_o, K)[:, 0]

        # Token assembly
        obs_tokens = self.token_assembly(adapted, proprio, vp_after)

        return obs_tokens, adapted_clean, vp_after

    # ------------------------------------------------------------------
    # Flow matching helpers
    # ------------------------------------------------------------------

    def _sample_flow_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample tau from pi0's shifted Beta distribution or uniform."""
        if self.fm_timestep_dist == "beta":
            u = torch.distributions.Beta(
                self.fm_beta_a, self.fm_beta_b
            ).sample((batch_size,)).to(device)
            tau = self.fm_cutoff * (1.0 - u)
        else:
            tau = torch.rand(batch_size, device=device)
        return tau

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass for DDP compatibility. Delegates to compute_loss."""
        return self.compute_loss(batch)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute policy loss (dispatches to DDPM or flow matching)."""
        if self.policy_type == "flow_matching":
            return self._compute_loss_flow_matching(batch)
        return self._compute_loss_ddpm(batch)

    def _compute_loss_ddpm(self, batch: dict) -> torch.Tensor:
        """DDPM noise prediction loss (Chi et al. 2023)."""
        proprio = batch["proprio"]
        actions = batch["actions"]
        view_present = batch["view_present"]

        obs_tokens, adapted_clean, vp_after = self._encode_and_assemble(
            batch, proprio, view_present
        )

        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self._train_steps, (actions.shape[0],), device=actions.device
        )
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)

        _, eps_pred = self.noise_net(noisy_actions, timesteps, obs_tokens)

        loss = F.mse_loss(eps_pred, noise)

        if self.lambda_recon > 0 and self.bridge.decoder is not None:
            images_target = batch["images_target"]
            loss_recon = self.bridge.compute_recon_loss(
                adapted_clean, images_target, view_present
            )
            loss = loss + self.lambda_recon * loss_recon

        return loss

    def _compute_loss_flow_matching(self, batch: dict) -> torch.Tensor:
        """Flow matching velocity prediction loss (Lipman et al. 2023, pi0)."""
        proprio = batch["proprio"]
        actions = batch["actions"]
        view_present = batch["view_present"]
        B = actions.shape[0]

        obs_tokens, adapted_clean, vp_after = self._encode_and_assemble(
            batch, proprio, view_present
        )

        eps = torch.randn_like(actions)
        tau = self._sample_flow_timestep(B, actions.device)

        tau_expand = tau[:, None, None]
        x_tau = tau_expand * actions + (1.0 - tau_expand) * eps
        target_velocity = actions - eps

        tau_scaled = tau * self.fm_timestep_scale

        enc_cache = self.noise_net.forward_enc(obs_tokens)
        v_pred = self.noise_net.forward_dec(x_tau, tau_scaled, enc_cache)

        loss = F.mse_loss(v_pred, target_velocity)

        if self.lambda_recon > 0 and self.bridge.decoder is not None:
            images_target = batch["images_target"]
            loss_recon = self.bridge.compute_recon_loss(
                adapted_clean, images_target, view_present
            )
            loss = loss + self.lambda_recon * loss_recon

        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_action(self, obs: dict) -> torch.Tensor:
        """Generate an action chunk (dispatches to DDIM or Euler)."""
        if self.policy_type == "flow_matching":
            return self._predict_action_flow_matching(obs)
        return self._predict_action_ddpm(obs)

    def _predict_action_ddpm(self, obs: dict) -> torch.Tensor:
        """DDIM inference (Chi et al. 2023)."""
        proprio = obs["proprio"]
        view_present = obs["view_present"]
        if "cached_tokens" in obs:
            B = obs["cached_tokens"].shape[0]
            device = obs["cached_tokens"].device
        else:
            B = obs["images_enc"].shape[0]
            device = obs["images_enc"].device

        obs_tokens, _, _ = self._encode_and_assemble(obs, proprio, view_present)
        enc_cache = self.noise_net.forward_enc(obs_tokens)

        noise_actions = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        self.scheduler.set_timesteps(self._eval_steps, device=device)
        for t in self.scheduler.timesteps:
            t_batch = t.expand(B)
            eps_pred = self.noise_net.forward_dec(noise_actions, t_batch, enc_cache)
            noise_actions = self.scheduler.step(
                eps_pred, t, noise_actions
            ).prev_sample

        return noise_actions

    def _predict_action_flow_matching(self, obs: dict) -> torch.Tensor:
        """Euler ODE integration for flow matching inference."""
        proprio = obs["proprio"]
        view_present = obs["view_present"]
        if "cached_tokens" in obs:
            B = obs["cached_tokens"].shape[0]
            device = obs["cached_tokens"].device
        else:
            B = obs["images_enc"].shape[0]
            device = obs["images_enc"].device

        obs_tokens, _, _ = self._encode_and_assemble(obs, proprio, view_present)
        enc_cache = self.noise_net.forward_enc(obs_tokens)

        x = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        N = self.num_flow_steps
        dt = 1.0 / N
        for i in range(N):
            tau = torch.full((B,), i * dt, device=device)
            tau_scaled = tau * self.fm_timestep_scale
            v_pred = self.noise_net.forward_dec(x, tau_scaled, enc_cache)
            x = x + dt * v_pred

        return x
