"""PolicyDiTv3 — Chi's cross-attention transformer denoiser in our pipeline.

Composes: Stage1Bridge → ObservationEncoder → TransformerDenoiser
with DDPM training (epsilon prediction) and DDIM inference.

Key differences from PolicyDiT (V1/V2):
  - Cross-attention denoiser (actions attend to obs), NOT adaLN-Zero
  - Pluggable denoiser interface (transformer now, U-Net later)
  - ImageNet norm inside policy (Stage1Bridge handles it for online mode)
  - Adaptive EMA schedule (power=0.75, handled externally)
  - clip_sample=True during DDIM inference
  - No spatial pooling 196→49 — avg pool to single vector per view

Training:  policy.compute_loss(batch) → scalar loss
Inference: policy.predict_action(obs_dict) → (B, T_p, ac_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.obs_encoder_v3 import ObservationEncoder
from models.denoiser_transformer import TransformerDenoiser
from models.stage1_bridge import Stage1Bridge


class PolicyDiTv3(nn.Module):
    """V3 diffusion policy with cross-attention transformer denoiser.

    Args:
        bridge:               Stage1Bridge (frozen encoder + trainable adapter).
        ac_dim:               Action dimension (10 for robomimic rot6d, 8 for RLBench).
        proprio_dim:          Proprioceptive state dimension.
        d_model:              Transformer hidden dimension.
        n_head:               Attention heads.
        n_layers:             Decoder layers.
        T_obs:                Observation horizon.
        T_pred:               Prediction horizon (action chunk length).
        num_views:            Maximum camera slots.
        n_active_cams:        Number of ACTIVE cameras (2 for robomimic, 4 for RLBench).
        train_diffusion_steps: DDPM training timesteps.
        eval_diffusion_steps:  DDIM inference steps.
        p_drop_emb:           Embedding dropout.
        p_drop_attn:          Attention dropout.
    """

    def __init__(
        self,
        bridge: Stage1Bridge,
        ac_dim: int = 10,
        proprio_dim: int = 9,
        d_model: int = 256,
        n_head: int = 4,
        n_layers: int = 8,
        T_obs: int = 2,
        T_pred: int = 16,
        num_views: int = 4,
        n_active_cams: int = 2,
        train_diffusion_steps: int = 100,
        eval_diffusion_steps: int = 100,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
    ):
        super().__init__()

        self.bridge = bridge
        self.ac_dim = ac_dim
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.train_diffusion_steps = train_diffusion_steps
        self.eval_diffusion_steps = eval_diffusion_steps

        # Observation encoder: adapted tokens + proprio → conditioning sequence
        self.obs_encoder = ObservationEncoder(
            adapter_dim=512,
            d_model=d_model,
            proprio_dim=proprio_dim,
            T_obs=T_obs,
            n_active_cams=n_active_cams,
        )

        # Transformer denoiser: cross-attention to obs conditioning
        self.denoiser = TransformerDenoiser(
            ac_dim=ac_dim,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            T_pred=T_pred,
            cond_dim=d_model,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=True,
        )

        # DDPM noise scheduler for training
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )

    def _encode_obs(self, batch: dict, pre_normalized: bool = True) -> dict:
        """Encode observations into conditioning dict.

        Handles both precomputed tokens (cached_tokens key) and
        online encoding (images_enc key via Stage1Bridge).

        Args:
            batch: dict with images_enc or cached_tokens, proprio, view_present.
            pre_normalized: Whether images_enc is already ImageNet-normalized.
                True during training (Stage3Dataset pre-normalizes).
                False during eval (env returns float [0,1]).

        Returns:
            obs_cond: dict with 'tokens' (B, S_obs, d_model) and 'global' (B, d_model)
        """
        view_present = batch["view_present"]

        if "cached_tokens" in batch:
            # Precomputed: run adapter only (skip frozen encoder)
            adapted = self.bridge.adapt(batch["cached_tokens"], view_present)
        else:
            # Online: full encoder → LN → adapter
            adapted = self.bridge.encode(
                batch["images_enc"], view_present, pre_normalized=pre_normalized
            )

        # adapted: (B, T_o, K, 196, 512)
        obs_cond = self.obs_encoder(adapted, batch["proprio"], view_present)
        return obs_cond

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """DDPM training: sample timestep, add noise, predict epsilon, MSE loss.

        Args:
            batch: dict with keys from Stage3Dataset:
                - images_enc or cached_tokens
                - actions: (B, T_pred, ac_dim) normalized
                - proprio: (B, T_obs, proprio_dim) normalized
                - view_present: (B, K) bool

        Returns:
            Scalar MSE loss.
        """
        actions = batch["actions"]  # (B, T_pred, ac_dim) normalized [-1, 1]
        B = actions.shape[0]
        device = actions.device

        # 1. Encode observations (training: images are pre-normalized by dataset)
        obs_cond = self._encode_obs(batch, pre_normalized=True)

        # 2. Sample random timesteps
        timesteps = torch.randint(
            0, self.train_diffusion_steps, (B,), device=device, dtype=torch.long
        )

        # 3. Add noise to actions
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # 4. Predict noise
        noise_pred = self.denoiser(noisy_actions, timesteps, obs_cond)

        # 5. MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> torch.Tensor:
        """DDIM inference: denoise from random noise to action trajectory.

        Args:
            obs_dict: dict with:
                - images_enc or cached_tokens
                - proprio: (B, T_obs, proprio_dim) normalized
                - view_present: (B, K) bool

        Returns:
            actions: (B, T_pred, ac_dim) in normalized scale [-1, 1]
        """
        device = obs_dict["proprio"].device
        B = obs_dict["proprio"].shape[0]

        # 1. Encode observations (eval: images are raw [0,1], bridge normalizes)
        obs_cond = self._encode_obs(obs_dict, pre_normalized=False)

        # 2. Create DDIM scheduler for inference
        scheduler = DDIMScheduler(
            num_train_timesteps=self.train_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
        )
        scheduler.set_timesteps(self.eval_diffusion_steps, device=device)

        # 3. Start from random noise
        actions = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        # 4. Denoising loop
        for t in scheduler.timesteps:
            timestep = t.expand(B)
            noise_pred = self.denoiser(actions, timestep, obs_cond)
            actions = scheduler.step(noise_pred, t, actions).prev_sample

        return actions

    def forward(self, batch: dict) -> torch.Tensor:
        """Alias for compute_loss (used by training loop)."""
        return self.compute_loss(batch)
