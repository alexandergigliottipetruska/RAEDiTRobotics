"""DiT Denoiser — Diffusion Transformer with adaLN-Zero blocks (Peebles & Xie, 2023).

Adapted for robot policy: observation tokens are **prepended** to the action token
sequence so the model can discriminate between cameras, timesteps, and spatial
locations via full self-attention. The diffusion timestep is injected via adaLN-Zero
modulation (the defining feature of DiT).

Architecture:
  1. Obs tokens → project to d_model → prepend to sequence
  2. Noisy actions → linear embed + positional encoding → append to sequence
  3. Diffusion timestep → sinusoidal → MLP → c (adaLN-Zero conditioning vector)
  4. Full sequence [obs_tokens | action_tokens] through N × DiTBlock(x, c)
  5. Strip obs prefix → Final adaLN-Zero → linear head → predicted noise

Each DiTBlock:
  - LN(x) modulated by (γ₁, β₁) from c → self-attention → gated by α₁ → residual
  - LN(x) modulated by (γ₂, β₂) from c → FFN → gated by α₂ → residual
  - α gates are zero-initialized so blocks start as identity (key DiT insight)

This gives the model per-token access to each camera view and timestep (like
cross-attention in DP-Transformer) while keeping the clean DiT conditioning
mechanism for the diffusion timestep.

Interface: matches TransformerDenoiser.forward(noisy_actions, timestep, obs_cond)
"""

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


def modulate(x, shift, scale):
    """Apply adaLN modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Single DiT block with adaLN-Zero conditioning.

    Args:
        d_model:  Hidden dimension.
        n_head:   Number of attention heads.
        mlp_ratio: FFN hidden dim multiplier.
        dropout:  Dropout rate for attention and FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

        # adaLN-Zero: 6 modulation params per block (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        # Zero-init so blocks start as identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — token sequence (obs + action tokens)
            c: (B, d_model)    — diffusion timestep conditioning vector

        Returns:
            (B, T, d_model)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Self-attention with adaLN modulation
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + gate_msa.unsqueeze(1) * h

        # FFN with adaLN modulation
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h

        return x


class FinalLayer(nn.Module):
    """DiT final layer: adaLN-Zero → linear projection to action dim."""

    def __init__(self, d_model: int, ac_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, ac_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model),
        )
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)
        return x


class DiTDenoiser(nn.Module):
    """Diffusion Transformer denoiser with adaLN-Zero conditioning.

    Observation tokens are prepended to the action token sequence so the model
    can discriminate between individual camera views, timesteps, and spatial
    locations via self-attention. The diffusion timestep is injected via
    adaLN-Zero modulation on every block.

    Sequence layout: [obs_0, obs_1, ..., obs_N, act_0, act_1, ..., act_T]
      - obs tokens: projected from ObservationEncoder output
      - act tokens: embedded noisy actions with learned positional encoding
      - After all DiT blocks, obs prefix is stripped; only action tokens are decoded

    Args:
        ac_dim:      Action dimension.
        d_model:     Transformer hidden dimension.
        n_head:      Number of attention heads.
        n_layers:    Number of DiT blocks.
        T_pred:      Prediction horizon (action sequence length).
        cond_dim:    Dimension of obs conditioning tokens (from ObservationEncoder).
        p_drop_emb:  Dropout on input embeddings.
        p_drop_attn: Dropout on attention weights.
        causal_attn: Unused (kept for interface compat with TransformerDenoiser).
    """

    def __init__(
        self,
        ac_dim: int = 10,
        d_model: int = 256,
        n_head: int = 4,
        n_layers: int = 8,
        T_pred: int = 16,
        cond_dim: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
    ):
        super().__init__()
        self.ac_dim = ac_dim
        self.d_model = d_model
        self.T_pred = T_pred

        # --- Action input embedding ---
        self.input_emb = nn.Linear(ac_dim, d_model)
        self.act_pos_emb = nn.Parameter(torch.zeros(1, T_pred, d_model))

        # --- Obs token projection (cond_dim → d_model) for legacy S=1 path ---
        self.obs_proj = nn.Linear(cond_dim, d_model)
        # Positional embedding for obs tokens (generous max size)
        max_obs_len = 1024
        self.obs_pos_emb = nn.Parameter(torch.zeros(1, max_obs_len, d_model))

        # --- Timestep embedding: sinusoidal → MLP → adaLN-Zero conditioning ---
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

        # --- Input dropout ---
        self.drop = nn.Dropout(p_drop_emb)

        # --- DiT blocks ---
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_head, mlp_ratio=4.0, dropout=p_drop_attn)
            for _ in range(n_layers)
        ])

        # --- Final layer: adaLN-Zero → linear (applied only to action tokens) ---
        self.final_layer = FinalLayer(d_model, ac_dim)

        # --- Weight initialization ---
        self.apply(self._init_weights)
        nn.init.normal_(self.act_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.obs_pos_emb, mean=0.0, std=0.02)

    def _init_weights(self, module):
        """Initialize weights: normal(0, 0.02) for Linear/Embedding, standard for LN."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]:
                weight = getattr(module, name, None)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ["in_proj_bias", "bias_k", "bias_v"]:
                bias = getattr(module, name, None)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Split parameters into decay/no_decay groups.

        Decay: weight params of Linear and MultiheadAttention.
        No decay: biases, LayerNorm, positional embeddings.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.MultiheadAttention)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Positional embeddings should not be decayed
        no_decay.add("act_pos_emb")
        no_decay.add("obs_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            f"parameters {inter_params} in both decay/no_decay sets"
        assert len(param_dict.keys() - union_params) == 0, \
            f"parameters {param_dict.keys() - union_params} not in either set"

        return [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        obs_cond: dict,
    ) -> torch.Tensor:
        """
        Args:
            noisy_actions: (B, T_pred, ac_dim) — noisy action trajectory
            timestep:      (B,) int — diffusion step k (0–99)
            obs_cond:      dict with 'tokens' from ObservationEncoder

        Returns:
            prediction: (B, T_pred, ac_dim) — predicted noise (epsilon)
        """
        B = noisy_actions.shape[0]
        obs_tokens = obs_cond["tokens"]
        pre_projected = obs_cond.get("pre_projected", False)

        # 1. Diffusion timestep → adaLN-Zero conditioning vector
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_actions.device)
        elif timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B)
        c = self.time_emb(timestep)  # (B, d_model)

        # 2. Project obs tokens to d_model
        if pre_projected:
            # Spatial tokens: already (B, N_obs, d_model)
            obs_emb = obs_tokens
        else:
            # Legacy tokens: (B, T_obs, cond_dim) → project each token
            obs_emb = self.obs_proj(obs_tokens)  # (B, T_obs, d_model)

        # Add obs positional embeddings
        N_obs = obs_emb.shape[1]
        obs_emb = obs_emb + self.obs_pos_emb[:, :N_obs, :]

        # 3. Embed noisy actions + positional encoding
        act_emb = self.input_emb(noisy_actions)  # (B, T_pred, d_model)
        act_emb = act_emb + self.act_pos_emb[:, :act_emb.shape[1], :]

        # 4. Concatenate: [obs_tokens | action_tokens]
        x = self.drop(torch.cat([obs_emb, act_emb], dim=1))  # (B, N_obs + T_pred, d_model)

        # 5. DiT blocks — full self-attention over obs + action tokens,
        #    diffusion timestep modulates via adaLN-Zero
        for block in self.blocks:
            x = block(x, c)

        # 6. Strip obs prefix, keep only action tokens
        x = x[:, N_obs:, :]  # (B, T_pred, d_model)

        # 7. Final layer: adaLN-Zero → linear head
        x = self.final_layer(x, c)  # (B, T_pred, ac_dim)

        return x
