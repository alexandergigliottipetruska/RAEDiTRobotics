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
        n_cond_layers: int = 0,
    ):
        super().__init__()
        self.ac_dim = ac_dim
        self.d_model = d_model
        self.T_pred = T_pred
        self.causal_attn = causal_attn

        # --- Action input embedding ---
        self.input_emb = nn.Linear(ac_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, T_pred, d_model))  # action positions only (Chi: no timestep in tgt)

        # --- Timestep embedding (Chi: raw sinusoidal, no MLP) ---
        self.time_emb = SinusoidalPosEmb(d_model)

        # --- Conditioning obs embedding ---
        self.cond_obs_emb = nn.Linear(cond_dim, d_model)

        # --- Condition positional embedding (timestep + obs tokens) ---
        # We don't know S_obs at init, so we allocate a generous max size
        # and slice at forward time (same pattern as Chi's code)
        max_cond_len = 1024  # handles up to S=14 spatial tokens (786 memory)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, d_model))

        # --- Dropout ---
        self.drop = nn.Dropout(p_drop_emb)

        # --- Condition encoder ---
        self.n_cond_layers = n_cond_layers
        if n_cond_layers > 0:
            # Self-attention encoder for spatial tokens to interact
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_cond_layers,
            )
        else:
            # MLP (n_cond_layers=0, Chi's original design)
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

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """Split parameters into decay/no_decay groups (Chi's pattern).

        Decay: weight params of Linear and MultiheadAttention.
        No decay: biases, LayerNorm weights+biases, positional embeddings.
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
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Positional embeddings should not be decayed
        no_decay.add("pos_emb")
        no_decay.add("cond_pos_emb")

        # Validate every parameter is accounted for
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
            obs_cond:      dict with 'tokens': (B, S_obs, cond_dim) from ObservationEncoder

        Returns:
            prediction: (B, T_pred, ac_dim) — predicted noise (epsilon)
        """
        B = noisy_actions.shape[0]
        obs_tokens = obs_cond["tokens"]  # (B, S_obs, cond_dim) or (B, N_mem, d_model)
        pre_projected = obs_cond.get("pre_projected", False)

        # 1. Timestep → token
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_actions.device)
        elif timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(B)
        time_token = self.time_emb(timestep).unsqueeze(1)  # (B, 1, d_model)

        # 2. Obs conditioning
        if pre_projected:
            cond_obs = obs_tokens  # already (B, N_mem, d_model)
        else:
            cond_obs = self.cond_obs_emb(obs_tokens)  # (B, S_obs, d_model)

        # 3. Memory = [time_token, obs_tokens] + positional embedding + dropout
        #    Chi: timestep is the FIRST token in memory, not in action sequence
        cond_embeddings = torch.cat([time_token, cond_obs], dim=1)  # (B, 1+S_obs, d_model)
        tc = cond_embeddings.shape[1]
        memory = self.drop(cond_embeddings + self.cond_pos_emb[:, :tc, :])

        # 4. Process memory through MLP encoder (n_cond_layers=0)
        memory = self.encoder(memory)  # (B, 1+S_obs, d_model)

        # 5. Action sequence = action_embs only + positional embedding + dropout
        #    Chi: NO timestep token in the action (tgt) sequence
        action_emb = self.input_emb(noisy_actions)  # (B, T_pred, d_model)
        t = action_emb.shape[1]  # T_pred
        tgt = self.drop(action_emb + self.pos_emb[:, :t, :])

        # 6. Build masks
        tgt_mask = None
        memory_mask = None
        if self.causal_attn:
            # Causal self-attention on actions: token i attends to tokens 0..i
            tgt_mask = torch.triu(
                torch.ones(t, t, device=noisy_actions.device), diagonal=1
            ).float().masked_fill_(
                torch.triu(torch.ones(t, t, device=noisy_actions.device), diagonal=1).bool(),
                float("-inf"),
            )

            if pre_projected:
                # Spatial tokens: all memory visible to all action tokens
                memory_mask = None
            else:
                # Legacy temporal mask (Chi line 130): action t sees memory s where t >= (s-1)
                # memory[0] = timestep (always visible), memory[1..] = obs tokens
                i_idx = torch.arange(t, device=noisy_actions.device)
                s_idx = torch.arange(tc, device=noisy_actions.device)
                memory_mask = (i_idx[:, None] >= (s_idx[None, :] - 1)).float()
                memory_mask = memory_mask.masked_fill(memory_mask == 0, float("-inf"))
                memory_mask = memory_mask.masked_fill(memory_mask == 1, 0.0)

        # 7. Decoder: cross-attention to memory
        x = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )  # (B, T_pred, d_model)

        # 8. Output head (no timestep token to strip)
        x = self.ln_f(x)
        x = self.head(x)  # (B, T_pred, ac_dim)

        return x
