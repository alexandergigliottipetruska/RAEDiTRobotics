"""PolicyDiT — C.10.

Top-level policy module that composes Stage1Bridge + ViewDropout +
TokenAssembly + _DiTNoiseNet into the BasePolicy interface.

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
from models.diffusion import _DiTNoiseNet
from models.stage1_bridge import Stage1Bridge
from models.token_assembly import TokenAssembly
from models.view_dropout import ViewDropout


class PolicyDiT(BasePolicy):
    """Full Policy DiT: Stage1Bridge + TokenAssembly + ViewDropout + DiTNoiseNet.

    Args:
        bridge: Stage1Bridge instance (encoder + adapter + optional decoder).
        ac_dim: Action dimension (7 for robomimic, 8 for RLBench).
        proprio_dim: Proprioceptive state dimension.
        hidden_dim: Transformer hidden dimension.
        T_obs: Observation horizon.
        T_pred: Prediction horizon (action chunk length).
        num_blocks: Number of transformer layers.
        nhead: Number of attention heads.
        num_views: Maximum camera slots.
        train_diffusion_steps: DDPM training steps.
        eval_diffusion_steps: DDIM inference steps.
        p_view_drop: View dropout probability.
        lambda_recon: Reconstruction co-training weight (0 = no co-training).
        use_lightning: Use LightningDiTBlock (adaLN-Zero) vs standard.
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
        hidden_dim: int = 512,
        T_obs: int = 2,
        T_pred: int = 16,
        num_blocks: int = 6,
        nhead: int = 8,
        num_views: int = 4,
        train_diffusion_steps: int = 100,
        eval_diffusion_steps: int = 10,
        p_view_drop: float = 0.15,
        lambda_recon: float = 0.0,
        use_lightning: bool = True,
        policy_type: str = "flow_matching",
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

        # View dropout
        self.view_dropout = ViewDropout(d_model=hidden_dim, p=p_view_drop)

        # Token assembly
        self.token_assembly = TokenAssembly(
            d_model=hidden_dim,
            num_patches=196,
            num_views=num_views,
            num_obs_steps=T_obs,
            proprio_dim=proprio_dim,
        )

        # Noise prediction / velocity prediction network (same architecture)
        self.noise_net = _DiTNoiseNet(
            ac_dim=ac_dim,
            ac_chunk=T_pred,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            nhead=nhead,
            use_lightning=use_lightning,
            dropout=0.0,
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
        """Encode images (or use cached tokens), apply view dropout, assemble.

        Args:
            batch: Full batch dict. Uses 'cached_tokens' if present,
                   otherwise 'images_enc'.
            proprio: (B, T_o, D_prop) normalized proprio.
            view_present: (B, K) bool.

        Returns:
            obs_tokens: (B, S_obs, d') for the encoder.
            adapted_clean: (B, T_o, K, N, d') for optional recon loss.
            view_present_after: (B, K) after dropout.
        """
        if "cached_tokens" in batch:
            # Fast path: skip encoder+LN, just run adapter
            adapted = self.bridge.adapt(batch["cached_tokens"], view_present)
        else:
            # Standard path: frozen encoder -> LN -> adapter
            adapted = self.bridge.encode(batch["images_enc"], view_present)
        adapted_clean = adapted  # save pre-dropout for recon loss

        # View dropout: (B, T_o, K, N, d') -> per-timestep dropout
        # Apply to each timestep independently
        B, T_o, K, N, d = adapted.shape
        # Reshape to (B*T_o, K, N, d) for ViewDropout
        adapted_flat = adapted.reshape(B * T_o, K, N, d)
        vp_flat = view_present.unsqueeze(1).expand(B, T_o, K).reshape(B * T_o, K)
        adapted_flat, vp_after_flat = self.view_dropout(adapted_flat, vp_flat)
        adapted = adapted_flat.reshape(B, T_o, K, N, d)
        vp_after = vp_after_flat.reshape(B, T_o, K)[:, 0]  # same across T_o

        # Token assembly
        obs_tokens = self.token_assembly(adapted, proprio, vp_after)

        return obs_tokens, adapted_clean, vp_after

    # ------------------------------------------------------------------
    # Flow matching helpers
    # ------------------------------------------------------------------

    def _sample_flow_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample tau from pi0's shifted Beta distribution or uniform.

        Returns:
            tau: (B,) float in [0, s] where s = fm_cutoff.
        """
        if self.fm_timestep_dist == "beta":
            u = torch.distributions.Beta(
                self.fm_beta_a, self.fm_beta_b
            ).sample((batch_size,)).to(device)
            tau = self.fm_cutoff * (1.0 - u)
        else:  # uniform
            tau = torch.rand(batch_size, device=device)
        return tau

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass for DDP compatibility. Delegates to compute_loss."""
        return self.compute_loss(batch)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute policy loss (dispatches to DDPM or flow matching).

        Args:
            batch: Dict with keys:
                images_enc: (B, T_o, K, 3, H, W) ImageNet-normalized  [standard]
                cached_tokens: (B, T_o, K, 196, 1024) precomputed     [cached]
                proprio: (B, T_o, D_prop) normalized
                actions: (B, T_p, D_act) normalized
                view_present: (B, K) bool
                images_target: (B, T_o, K, 3, H, W) raw [0,1] (for recon)

        Returns:
            Scalar total loss.
        """
        if self.policy_type == "flow_matching":
            return self._compute_loss_flow_matching(batch)
        return self._compute_loss_ddpm(batch)

    def _compute_loss_ddpm(self, batch: dict) -> torch.Tensor:
        """DDPM noise prediction loss (Chi et al. 2023)."""
        proprio = batch["proprio"]
        actions = batch["actions"]
        view_present = batch["view_present"]

        # Encode + assemble
        obs_tokens, adapted_clean, vp_after = self._encode_and_assemble(
            batch, proprio, view_present
        )

        # DDPM forward
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0, self._train_steps, (actions.shape[0],), device=actions.device
        )
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)

        # Noise prediction
        _, eps_pred = self.noise_net(noisy_actions, timesteps, obs_tokens)

        # Policy loss
        loss = F.mse_loss(eps_pred, noise)

        # Optional reconstruction co-training
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

        # Encode + assemble (identical to DDPM)
        obs_tokens, adapted_clean, vp_after = self._encode_and_assemble(
            batch, proprio, view_present
        )

        # Sample noise
        eps = torch.randn_like(actions)

        # Sample continuous timestep tau in [0, s]
        tau = self._sample_flow_timestep(B, actions.device)

        # Linear interpolation: x_tau = tau * a_0 + (1 - tau) * eps
        tau_expand = tau[:, None, None]  # (B, 1, 1) for broadcasting
        x_tau = tau_expand * actions + (1.0 - tau_expand) * eps

        # Velocity target: v = a_0 - eps
        target_velocity = actions - eps

        # Scale tau for the time network (sinusoidal encoding)
        tau_scaled = tau * self.fm_timestep_scale

        # Forward through noise net (same network, velocity semantics)
        enc_cache = self.noise_net.forward_enc(obs_tokens)
        v_pred = self.noise_net.forward_dec(x_tau, tau_scaled, enc_cache)

        # MSE loss on velocity
        loss = F.mse_loss(v_pred, target_velocity)

        # Optional reconstruction co-training (identical to DDPM)
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
        """Generate an action chunk (dispatches to DDIM or Euler).

        Args:
            obs: Dict with keys:
                images_enc: (B, T_o, K, 3, H, W)
                proprio: (B, T_o, D_prop)
                view_present: (B, K) bool

        Returns:
            actions: (B, T_pred, ac_dim) predicted action chunk.
        """
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

        # Encode + assemble (no dropout at eval)
        obs_tokens, _, _ = self._encode_and_assemble(
            obs, proprio, view_present
        )

        # Cache encoder outputs
        enc_cache = self.noise_net.forward_enc(obs_tokens)

        # Start from pure noise
        noise_actions = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        # DDIM denoising loop
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

        # Encode + assemble (no dropout at eval)
        obs_tokens, _, _ = self._encode_and_assemble(
            obs, proprio, view_present
        )

        # Cache encoder outputs (reused across all integration steps)
        enc_cache = self.noise_net.forward_enc(obs_tokens)

        # Start from pure noise at tau=0
        x = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        # Euler integration from tau=0 to tau=1
        N = self.num_flow_steps
        dt = 1.0 / N
        for i in range(N):
            tau = torch.full((B,), i * dt, device=device)
            tau_scaled = tau * self.fm_timestep_scale
            v_pred = self.noise_net.forward_dec(x, tau_scaled, enc_cache)
            x = x + dt * v_pred

        return x
