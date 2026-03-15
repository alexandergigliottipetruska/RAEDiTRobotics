"""PolicyDiT — C.10.

Top-level policy module that composes Stage1Bridge + ViewDropout +
TokenAssembly + _DiTNoiseNet into the BasePolicy interface.

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
    ):
        super().__init__()

        self.bridge = bridge
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.ac_dim = ac_dim
        self.lambda_recon = lambda_recon
        self._train_steps = train_diffusion_steps
        self._eval_steps = eval_diffusion_steps

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

        # Noise prediction network (from existing diffusion.py)
        self.noise_net = _DiTNoiseNet(
            ac_dim=ac_dim,
            ac_chunk=T_pred,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            nhead=nhead,
            use_lightning=use_lightning,
        )

        # DDIM scheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def _encode_and_assemble(
        self, images_enc, proprio, view_present
    ):
        """Encode images, apply view dropout, assemble tokens.

        Returns:
            obs_tokens: (B, S_obs, d') for the encoder.
            adapted_clean: (B, T_o, K, N, d') for optional recon loss.
            view_present_after: (B, K) after dropout.
        """
        # Stage 1 encode: frozen encoder -> LN -> adapter
        adapted = self.bridge.encode(images_enc, view_present)
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

    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass for DDP compatibility. Delegates to compute_loss."""
        return self.compute_loss(batch)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute DDPM noise prediction loss.

        Args:
            batch: Dict with keys:
                images_enc: (B, T_o, K, 3, H, W) ImageNet-normalized
                proprio: (B, T_o, D_prop) normalized
                actions: (B, T_p, D_act) normalized
                view_present: (B, K) bool
                images_target: (B, T_o, K, 3, H, W) raw [0,1] (for recon)

        Returns:
            Scalar total loss.
        """
        images_enc = batch["images_enc"]
        proprio = batch["proprio"]
        actions = batch["actions"]
        view_present = batch["view_present"]

        # Encode + assemble
        obs_tokens, adapted_clean, vp_after = self._encode_and_assemble(
            images_enc, proprio, view_present
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

    @torch.no_grad()
    def predict_action(self, obs: dict) -> torch.Tensor:
        """Run DDIM inference to generate an action chunk.

        Args:
            obs: Dict with keys:
                images_enc: (B, T_o, K, 3, H, W)
                proprio: (B, T_o, D_prop)
                view_present: (B, K) bool

        Returns:
            actions: (B, T_pred, ac_dim) predicted action chunk.
        """
        images_enc = obs["images_enc"]
        proprio = obs["proprio"]
        view_present = obs["view_present"]
        B = images_enc.shape[0]
        device = images_enc.device

        # Encode + assemble (no dropout at eval)
        obs_tokens, _, _ = self._encode_and_assemble(
            images_enc, proprio, view_present
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
