"""PolicyDiTv3 — pluggable diffusion policy with selectable denoiser backbone.

Composes: Stage1Bridge → ObservationEncoder → Denoiser
with DDPM training (epsilon/v-prediction) and DDIM inference.

Supported denoiser_type values:
  - "transformer" (default): Chi's cross-attention decoder (TransformerDenoiser)
  - "dit": True DiT with adaLN-Zero blocks (DiTDenoiser, Peebles & Xie 2023)

Key features:
  - Pluggable denoiser interface (same forward signature for both)
  - ImageNet norm inside policy (Stage1Bridge handles it for online mode)
  - Adaptive EMA schedule (power=0.75, handled externally)
  - clip_sample=True during DDIM inference
  - Configurable spatial pooling: S=1 avg pool (default), S=4/7/14 spatial tokens
  - Classifier-Free Guidance (CFG): drop obs conditioning during training, guide at inference
  - v-prediction: predict v = alpha_t * epsilon - sigma_t * x_0 (more stable at high noise)

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


def _gpu_zero_pad_compact_tokens(batch: dict) -> torch.Tensor:
    """Zero-pad compact tokens (K_active → K_full) on GPU if needed."""
    cached = batch["cached_tokens"]
    if "K_full" in batch:
        K_full = batch["K_full"]
        if isinstance(K_full, torch.Tensor):
            K_full = K_full[0].item()
        if cached.shape[2] < K_full:
            active_idx = batch["active_cam_indices"][0]  # same for all in batch
            full = cached.new_zeros(*cached.shape[:2], K_full, *cached.shape[3:])
            full[:, :, active_idx] = cached
            return full
    return cached


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
        prediction_type:      "epsilon" (default), "v_prediction", or "sample".
        cfg_drop_rate:        Classifier-free guidance dropout rate (0 = disabled).
        cfg_scale:            Guidance scale at inference (1.0 = no guidance).
        use_rope:             Use RoPE in DiT denoiser.
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
        prediction_type: str = "epsilon",
        cfg_drop_rate: float = 0.0,
        cfg_scale: float = 1.0,
        use_rope: bool = False,
    ):
        super().__init__()

        self.bridge = bridge
        self.ac_dim = ac_dim
        self.T_obs = T_obs
        self.T_pred = T_pred
        self.train_diffusion_steps = train_diffusion_steps
        self.eval_diffusion_steps = eval_diffusion_steps
        self.use_flow_matching = use_flow_matching
        self.prediction_type = prediction_type
        self.cfg_drop_rate = cfg_drop_rate
        self.cfg_scale = cfg_scale

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
            denoiser_kwargs["use_rope"] = use_rope
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
                prediction_type=prediction_type,
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
            cached = _gpu_zero_pad_compact_tokens(batch)
            # Precomputed: run adapter only (skip frozen encoder)
            adapted = self.bridge.adapt(cached, view_present)
        else:
            # Online: full encoder → LN → adapter
            adapted = self.bridge.encode(
                batch["images_enc"], view_present, pre_normalized=pre_normalized
            )

        # adapted: (B, T_o, K, 196, 512)
        obs_cond = self.obs_encoder(adapted, batch["proprio"], view_present)
        return obs_cond

    def _get_null_obs_cond(self, obs_cond: dict) -> dict:
        """Create null (zeros) conditioning for classifier-free guidance."""
        null_cond = {}
        for k, v in obs_cond.items():
            if isinstance(v, torch.Tensor):
                null_cond[k] = torch.zeros_like(v)
            else:
                null_cond[k] = v
        return null_cond

    def compute_loss(self, batch: dict, lambda_recon: float = 0.0) -> dict:
        """Training loss: DDPM epsilon/v-prediction or L1 Flow sample-prediction.

        With CFG: randomly drops observation conditioning with probability cfg_drop_rate.

        Args:
            batch: dict with keys from Stage3Dataset:
                - images_enc or cached_tokens
                - actions: (B, T_pred, ac_dim) normalized
                - proprio: (B, T_obs, proprio_dim) normalized
                - view_present: (B, K) bool
                - images_target: (B, T_obs, K, 3, H, W) optional, for co-training
            lambda_recon: weight for reconstruction loss (0=disabled).

        Returns:
            dict with 'loss' (scalar for backward), 'policy' (float),
            'recon' (float), for logging.
        """
        actions = batch["actions"]  # (B, T_pred, ac_dim) normalized [-1, 1]
        B = actions.shape[0]
        device = actions.device
        view_present = batch["view_present"]

        # 1. Encode observations — capture adapted tokens for reconstruction
        if "cached_tokens" in batch:
            cached = _gpu_zero_pad_compact_tokens(batch)
            adapted = self.bridge.adapt(cached, view_present)
        else:
            adapted = self.bridge.encode(
                batch["images_enc"], view_present, pre_normalized=True
            )

        # adapted: (B, T_o, K, 196, 512)
        obs_cond = self.obs_encoder(adapted, batch["proprio"], view_present)

        # 2. CFG: randomly drop conditioning
        if self.cfg_drop_rate > 0 and self.training:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_rate
            if drop_mask.any():
                null_cond = self._get_null_obs_cond(obs_cond)
                for k, v in obs_cond.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == B:
                        obs_cond[k] = torch.where(
                            drop_mask.view(B, *([1] * (v.dim() - 1))),
                            null_cond[k], v,
                        )

        # 3. Diffusion loss
        if self.use_flow_matching:
            noise = torch.randn_like(actions)
            t = self.logistic_normal.sample((B,))[:, 0].to(device)
            uni_t = torch.rand_like(t)
            mask = torch.rand_like(t) < 0.01
            t[mask] = uni_t[mask]
            t_expand = t[:, None, None]
            noisy_actions = t_expand * actions + (1 - t_expand) * noise
            timesteps = (t * 1000).long()
            x_pred = self.denoiser(noisy_actions, timesteps, obs_cond)
            loss_policy = F.l1_loss(x_pred, actions)
        else:
            # DDPM training
            timesteps = torch.randint(
                0, self.train_diffusion_steps, (B,), device=device, dtype=torch.long
            )
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            model_output = self.denoiser(noisy_actions, timesteps, obs_cond)

            if self.prediction_type == "epsilon":
                target = noise
            elif self.prediction_type == "v_prediction":
                # v = alpha_t * noise - sigma_t * sample
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps] ** 0.5
                sigma_t = (1 - self.noise_scheduler.alphas_cumprod[timesteps]) ** 0.5
                alpha_t = alpha_t[:, None, None]
                sigma_t = sigma_t[:, None, None]
                target = alpha_t * noise - sigma_t * actions
            elif self.prediction_type == "sample":
                target = actions
            else:
                raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
            loss_policy = F.mse_loss(model_output, target)

        # 3. Reconstruction loss (co-training)
        loss_recon = torch.tensor(0.0, device=device)
        if lambda_recon > 0 and "images_target" in batch and self.bridge.decoder is not None:
            loss_recon = self.bridge.compute_recon_loss(
                adapted, batch["images_target"], view_present,
            )

        loss = loss_policy + lambda_recon * loss_recon

        return {
            "loss": loss,
            "policy": loss_policy.item(),
            "recon": loss_recon.item(),
        }

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> torch.Tensor:
        """Inference: DDIM denoising or L1 Flow 2-step prediction.

        With CFG: runs conditional and unconditional passes, blends with cfg_scale.

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

        use_cfg = self.cfg_scale > 1.0 and self.cfg_drop_rate > 0
        if use_cfg:
            null_cond = self._get_null_obs_cond(obs_cond)

        # 2. Start from random noise
        actions = torch.randn(B, self.T_pred, self.ac_dim, device=device)

        if self.use_flow_matching:
            # L1 Flow 2-step inference (Song et al. 2025)
            t = torch.zeros(B, device=device)

            # Step 1: One-step integration to midpoint (t=0 → t=0.5)
            t_int = (t * 1000).long()
            x_pred = self.denoiser(actions, t_int, obs_cond)
            if use_cfg:
                x_pred_uncond = self.denoiser(actions, t_int, null_cond)
                x_pred = x_pred_uncond + self.cfg_scale * (x_pred - x_pred_uncond)
            v_t = (x_pred - actions) / (1 - t[:, None, None])
            actions = actions + 0.5 * v_t
            t = t + 0.5

            # Step 2: Direct prediction at midpoint (t=0.5 → x₁)
            t_int = (t * 1000).long()
            actions = self.denoiser(actions, t_int, obs_cond)
            if use_cfg:
                actions_uncond = self.denoiser(actions, t_int, null_cond)
                actions = actions_uncond + self.cfg_scale * (actions - actions_uncond)
            actions = actions.clamp(-1, 1)
        else:
            # DDIM denoising loop
            scheduler = DDIMScheduler(
                num_train_timesteps=self.train_diffusion_steps,
                beta_schedule="squaredcos_cap_v2",
                prediction_type=self.prediction_type,
                clip_sample=True,
            )
            scheduler.set_timesteps(self.eval_diffusion_steps, device=device)

            for t in scheduler.timesteps:
                timestep = t.expand(B)
                model_output = self.denoiser(actions, timestep, obs_cond)

                if use_cfg:
                    model_output_uncond = self.denoiser(actions, timestep, null_cond)
                    model_output = model_output_uncond + self.cfg_scale * (
                        model_output - model_output_uncond
                    )

                actions = scheduler.step(model_output, t, actions).prev_sample

        return actions

    @staticmethod
    def project_rot6d_via_quaternion(actions: torch.Tensor) -> torch.Tensor:
        """Project rot6d portion of 10D actions through quaternion space.

        10D = pos3 + rot6d6 + grip1.
        rot6d → Gram-Schmidt → rotation matrix → quaternion → normalize → rotation matrix → rot6d.
        Ensures the rotation component is a valid SO(3) element.
        """
        pos = actions[..., :3]
        rot6d = actions[..., 3:9]
        grip = actions[..., 9:]

        # Gram-Schmidt orthogonalization
        a1 = rot6d[..., :3]
        a2 = rot6d[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)

        # Rotation matrix (*, 3, 3)
        R = torch.stack([b1, b2, b3], dim=-1)

        # R → quaternion (wxyz) via Shepperd's method
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        qw = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) / 2
        qx = (R[..., 2, 1] - R[..., 1, 2]) / (4 * qw.clamp(min=1e-8))
        qy = (R[..., 0, 2] - R[..., 2, 0]) / (4 * qw.clamp(min=1e-8))
        qz = (R[..., 1, 0] - R[..., 0, 1]) / (4 * qw.clamp(min=1e-8))
        q = torch.stack([qw, qx, qy, qz], dim=-1)

        # Normalize quaternion (enforce unit constraint)
        q = F.normalize(q, dim=-1)

        # Quaternion → rotation matrix
        qw, qx, qy, qz = q.unbind(-1)
        R2 = torch.stack([
            1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw),
            2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw),
            2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy),
        ], dim=-1).reshape(*actions.shape[:-1], 3, 3)

        # First two columns → rot6d
        rot6d_clean = torch.cat([R2[..., :, 0], R2[..., :, 1]], dim=-1)
        return torch.cat([pos, rot6d_clean, grip], dim=-1)

    def forward(self, batch: dict) -> dict:
        """Alias for compute_loss (used by training loop)."""
        return self.compute_loss(batch, lambda_recon=getattr(self, '_lambda_recon', 0.0))
