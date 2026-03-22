"""V3 Transformer Denoiser — faithful reimplementation of Chi's TransformerForDiffusion.

Cross-attention decoder: action tokens self-attend causally and cross-attend to
a memory sequence containing the diffusion timestep token + observation tokens.

Key architectural choices (from v3.2.tex and Chi's code):
  - n_cond_layers=0: MLP (not transformer) processes memory tokens
  - Causal self-attention on action tokens
  - Memory mask: action token i sees memory positions s where s <= i + 1
  - Pre-norm (norm_first=True) decoder layers
  - All weights initialized normal(0, 0.02), matching Chi exactly
  - Attention dropout 0.01, embedding dropout 0.01, no residual dropout

Interface: BaseDenoiser.forward(noisy_actions, timestep, obs_cond)
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
        """x: (B,) integer timesteps → (B, dim) embeddings."""
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TransformerDenoiser(nn.Module):
    """Chi's TransformerForDiffusion adapted for our V3 pipeline.

    Args:
        ac_dim:      Action dimension (10 for robomimic rot6d, 8 for RLBench).
        d_model:     Transformer hidden dimension.
        n_head:      Number of attention heads.
        n_layers:    Number of decoder layers.
        T_pred:      Prediction horizon (action sequence length).
        cond_dim:    Dimension of obs conditioning tokens (from ObservationEncoder).
        p_drop_emb:  Dropout on input/cond embeddings.
        p_drop_attn: Dropout on attention weights.
        causal_attn: Whether to apply causal masking on action self-attention.
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
        self.causal_attn = causal_attn

        # --- Action input embedding ---
        self.input_emb = nn.Linear(ac_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, T_pred, d_model))

        # --- Timestep embedding (Chi: raw sinusoidal, no MLP) ---
        self.time_emb = SinusoidalPosEmb(d_model)

        # --- Conditioning obs embedding ---
        self.cond_obs_emb = nn.Linear(cond_dim, d_model)

        # --- Condition positional embedding (timestep + obs tokens) ---
        # We don't know S_obs at init, so we allocate a generous max size
        # and slice at forward time (same pattern as Chi's code)
        max_cond_len = 128  # generous upper bound
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, d_model))

        # --- Dropout ---
        self.drop = nn.Dropout(p_drop_emb)

        # --- Condition encoder: MLP (n_cond_layers=0) ---
        self.encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.Mish(),
            nn.Linear(4 * d_model, d_model),
        )

        # --- Decoder: cross-attention transformer ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
        )

        # --- Output head ---
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, ac_dim)

        # --- Weight initialization: Chi's scheme, normal(0, 0.02) everywhere ---
        self.apply(self._init_weights)
        # Chi also initializes positional embeddings with normal(0, 0.02)
        # (not zeros — this affects early training convergence)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.cond_pos_emb, mean=0.0, std=0.02)

    def _init_weights(self, module):
        """Chi's weight initialization: normal(0, 0.02) for Linear/Embedding."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
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
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

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
            obs_cond:      dict with 'tokens': (B, S_obs, cond_dim) from ObservationEncoder

        Returns:
            prediction: (B, T_pred, ac_dim) — predicted noise (epsilon)
        """
        B = noisy_actions.shape[0]
        obs_tokens = obs_cond["tokens"]  # (B, S_obs, cond_dim)

        # 1. Timestep → token
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_actions.device)
        elif timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B)
        time_token = self.time_emb(timestep).unsqueeze(1)  # (B, 1, d_model)

        # 2. Obs conditioning → project to d_model
        cond_obs = self.cond_obs_emb(obs_tokens)  # (B, S_obs, d_model)

        # 3. Memory = [time_token, cond_obs] + positional embedding + dropout
        memory = torch.cat([time_token, cond_obs], dim=1)  # (B, 1+S_obs, d_model)
        tc = memory.shape[1]
        memory = self.drop(memory + self.cond_pos_emb[:, :tc, :])

        # 4. Process memory through MLP encoder (n_cond_layers=0)
        memory = self.encoder(memory)  # (B, 1+S_obs, d_model)

        # 5. Action tokens: embed + positional embedding + dropout
        action_emb = self.input_emb(noisy_actions)  # (B, T_pred, d_model)
        t = action_emb.shape[1]
        action_emb = self.drop(action_emb + self.pos_emb[:, :t, :])

        # 6. Build masks
        tgt_mask = None
        memory_mask = None
        if self.causal_attn:
            # Causal self-attention mask for action tokens
            # torch.nn.Transformer uses additive mask: -inf blocks, 0 allows
            tgt_mask = torch.triu(
                torch.ones(t, t, device=noisy_actions.device), diagonal=1
            ).float().masked_fill_(
                torch.triu(torch.ones(t, t, device=noisy_actions.device), diagonal=1).bool(),
                float("-inf"),
            )

            # Memory mask: action token i can see memory positions s where s <= i + 1
            # (timestep is always visible; obs tokens visible up to current time)
            i_idx = torch.arange(t, device=noisy_actions.device)
            s_idx = torch.arange(tc, device=noisy_actions.device)
            memory_mask = (i_idx[:, None] >= (s_idx[None, :] - 1)).float()
            memory_mask = memory_mask.masked_fill(memory_mask == 0, float("-inf"))
            memory_mask = memory_mask.masked_fill(memory_mask == 1, 0.0)

        # 7. Decoder: cross-attention to memory
        x = self.decoder(
            tgt=action_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )  # (B, T_pred, d_model)

        # 8. Output head
        x = self.ln_f(x)
        x = self.head(x)  # (B, T_pred, ac_dim)

        return x
