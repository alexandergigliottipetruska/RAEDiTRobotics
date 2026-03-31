"""PolicyDiTv3 — pluggable diffusion policy with selectable denoiser backbone.

Composes: Stage1Bridge → ObservationEncoder → Denoiser
with DDPM training (epsilon prediction) and DDIM inference.

Supported denoiser_type values:
  - "transformer" (default): Chi's cross-attention decoder (TransformerDenoiser)
  - "dit": True DiT with adaLN-Zero blocks (DiTDenoiser, Peebles & Xie 2023)

Key features:
  - Pluggable denoiser interface (same forward signature for both)
  - ImageNet norm inside policy (Stage1Bridge handles it for online mode)
  - Adaptive EMA schedule (power=0.75, handled externally)
  - clip_sample=True during DDIM inference
  - Configurable spatial pooling: S=1 avg pool (default), S=4/7/14 spatial tokens

Training:  policy.compute_loss(batch) → scalar loss
Inference: policy.predict_action(obs_dict) → (B, T_p, ac_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.distributions import LogisticNormal

from models.obs_encoder_v3 import ObservationEncoder
from models.denoiser_transformer import TransformerDenoiser
from models.denoiser_dit import DiTDenoiser
from models.stage1_bridge import Stage1Bridge


class PolicyDiTv3(nn.Module):
    """V3 diffusion policy with pluggable denoiser backbone.

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
        denoiser_type:        "transformer" (Chi cross-attn) or "dit" (adaLN-Zero).
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
        spatial_pool_size: int = 1,
        use_spatial_softmax: bool = False,
        n_cond_layers: int = 0,
        denoiser_type: str = "transformer",
        use_flow_matching: bool = False,
    ):
        super().__init__()

        self.bridge = bridge
        self.ac_dim = ac_dim
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.train_diffusion_steps = train_diffusion_steps
        self.eval_diffusion_steps = eval_diffusion_steps
        self.use_flow_matching = use_flow_matching

        # Observation encoder: adapted tokens + proprio → conditioning sequence
        self.obs_encoder = ObservationEncoder(
            adapter_dim=512,
            d_model=d_model,
            proprio_dim=proprio_dim,
            T_obs=T_obs,
            n_active_cams=n_active_cams,
            spatial_pool_size=spatial_pool_size,
            use_spatial_softmax=use_spatial_softmax,
        )

        # Denoiser: selectable backbone
        denoiser_kwargs = dict(
            ac_dim=ac_dim,
            d_model=d_model,
            n_head=n_head,
            n_layers=n_layers,
            T_pred=T_pred,
            cond_dim=self.obs_encoder.output_dim,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=True,
            n_cond_layers=n_cond_layers,
        )

        if denoiser_type == "transformer":
            self.denoiser = TransformerDenoiser(**denoiser_kwargs)
        elif denoiser_type == "dit":
            self.denoiser = DiTDenoiser(**denoiser_kwargs)
        else:
            raise ValueError(f"Unknown denoiser_type: {denoiser_type!r}. "
                             f"Choose 'transformer' or 'dit'.")

        # Noise schedule / flow matching setup
        if use_flow_matching:
            self.logistic_normal = LogisticNormal(
                torch.tensor(0.0), torch.tensor(1.0)
            )
            self.noise_scheduler = None
        else:
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
        """Training loss: DDPM epsilon-prediction or L1 Flow sample-prediction.

        Args:
            batch: dict with keys from Stage3Dataset:
                - images_enc or cached_tokens
                - actions: (B, T_pred, ac_dim) normalized
                - proprio: (B, T_obs, proprio_dim) normalized
                - view_present: (B, K) bool

        Returns:
            Scalar loss.
        """
        actions = batch["actions"]  # (B, T_pred, ac_dim) normalized [-1, 1]
        B = actions.shape[0]
        device = actions.device

        # 1. Encode observations (training: images are pre-normalized by dataset)
        obs_cond = self._encode_obs(batch, pre_normalized=True)

        if self.use_flow_matching:
            # L1 Sample Flow (Song et al. 2025)
            noise = torch.randn_like(actions)

            # Logistic-normal + 1% uniform timestep sampling
            t = self.logistic_normal.sample((B,))[:, 0].to(device)
            uni_t = torch.rand_like(t)
            mask = torch.rand_like(t) < 0.01
            t[mask] = uni_t[mask]

            # Linear interpolation: x_t = t * x₁ + (1-t) * x₀
            t_expand = t[:, None, None]
            noisy_actions = t_expand * actions + (1 - t_expand) * noise

            # Model predicts clean sample x₁; scale t for sinusoidal embedding
            timesteps = (t * 1000).long()
            x_pred = self.denoiser(noisy_actions, timesteps, obs_cond)

            # L1 loss on sample prediction
            loss = F.l1_loss(x_pred, actions)
        else:
            # DDPM epsilon prediction (Chi's original)
            timesteps = torch.randint(
                0, self.train_diffusion_steps, (B,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            noise_pred = self.denoiser(noisy_actions, timesteps, obs_cond)
            loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> torch.Tensor:
        """Inference: DDIM denoising or L1 Flow 2-step prediction.

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

        # 2. Start from random noise
        actions = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        if self.use_flow_matching:
            # L1 Flow 2-step inference (Song et al. 2025)
            t = torch.zeros(B, device=device)

            # Step 1: One-step integration to midpoint (t=0 → t=0.5)
            t_int = (t * 1000).long()
            x_pred = self.denoiser(actions, t_int, obs_cond)
            v_t = (x_pred - actions) / (1 - t[:, None, None])
            actions = actions + 0.5 * v_t
            t = t + 0.5

            # Step 2: Direct prediction at midpoint (t=0.5 → x₁)
            t_int = (t * 1000).long()
            actions = self.denoiser(actions, t_int, obs_cond)
            actions = actions.clamp(-1, 1)
        else:
            # DDIM denoising loop (Chi's original)
            scheduler = DDIMScheduler(
                num_train_timesteps=self.train_diffusion_steps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=True,
            )
            scheduler.set_timesteps(self.eval_diffusion_steps, device=device)

            for t in scheduler.timesteps:
                timestep = t.expand(B)
                noise_pred = self.denoiser(actions, timestep, obs_cond)
                actions = scheduler.step(noise_pred, t, actions).prev_sample

        return actions

    def forward(self, batch: dict) -> torch.Tensor:
        """Alias for compute_loss (used by training loop)."""
        return self.compute_loss(batch)
