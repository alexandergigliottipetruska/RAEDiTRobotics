# Copyright (c) Sudeep Dasari, 2023
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Architecture merges two papers:
#
#   [iRDT] "Ingredients for Robotic Diffusion Transformers"
#       - Cross-attention from action tokens to obs tokens at every decoder block
#       - Multi-scale encoder-decoder coupling (each decoder depth conditions on its
#         paired encoder depth's output)
#       - AdaLN modulation of self-attention and MLP from the timestep embedding
#
#   [RAE-DiT] "Diffusion Transformers with Representation Autoencoders"
#       - Lightning DiT block: adaLN-Zero unified modulation (one SiLU→Linear produces
#         all 6 shift/scale/gate params; final Linear zero-initialized for identity
#         initialization) — faster and more stable than separate _ShiftScaleMod layers
#       - Cross-attention to externally-provided conditioning tokens (e.g. RAE latents)
#         allowing the noise net to decouple from a fixed internal encoder
#
# Both block variants (_DiTCrossAttnBlock, _LightningDiTBlock) share the same
# interface so they are drop-in replaceable via the `use_lightning` flag.


import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from models.base_policy import BasePolicy


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positional encodings of shape (batch_size, seq_len, d_model)
        """
        pe = self.pe[:, : x.shape[1]]      # (1, seq_len, d_model)
        pe = pe.repeat((x.shape[0], 1, 1)) # (batch_size, seq_len, d_model)
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c).unsqueeze(1) + self.shift(c).unsqueeze(1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c).unsqueeze(1)

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x, t, cond):
        # cond: (B, seq_len, d) — mean-pool to (B, d) then add timestep
        cond = torch.mean(cond, dim=1) + t
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x   # (B, ac_chunk, ac_dim)

    def reset_parameters(self):
        # Zero everything: adaLN modulation AND output linear projection.
        # At init, model predicts zero noise/velocity everywhere ("I don't know").
        # This matches DiT (Peebles & Xie 2023) original initialization.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )
        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs  # list of per-depth outputs for multi-scale conditioning


# ---------------------------------------------------------------------------
# Standard DiT cross-attention block  [iRDT + RAE-DiT]
# ---------------------------------------------------------------------------

class _DiTCrossAttnBlock(nn.Module):
    """Standard DiT decoder block combining iRDT and RAE-DiT.

    Three sequential operations, all with separate AdaLN from the timestep:

      1. Self-attention on action tokens
         Pre-norm shifted/scaled by t  →  attention  →  zero-gated residual

      2. Cross-attention from action tokens to obs/conditioning tokens  [iRDT]
         Plain LayerNorm (conditioning info comes through attention, not AdaLN)

      3. MLP
         Pre-norm shifted/scaled by t  →  FFN  →  zero-gated residual

    The conditioning tokens (`cond`) can be:
      - Per-depth encoder outputs  (iRDT multi-scale coupling)
      - External RAE latents       (RAE-DiT decoupled conditioning)
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop_self = nn.Dropout(dropout)
        self.drop_cross = nn.Dropout(dropout)
        self.drop_mlp = nn.Dropout(dropout)
        self.drop_gate = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # AdaLN modulation from timestep (iRDT style)
        self.attn_mod1 = _ShiftScaleMod(d_model)   # pre-norm shift/scale
        self.attn_mod2 = _ZeroScaleMod(d_model)    # post-attn gate
        self.mlp_mod1 = _ShiftScaleMod(d_model)    # pre-norm shift/scale
        self.mlp_mod2 = _ZeroScaleMod(d_model)     # post-MLP gate

    def forward(self, x, t, cond):
        # 1. Self-attention with AdaLN from timestep
        x2 = self.attn_mod1(self.norm1(x), t)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.drop_self(x2), t) + x

        # 2. Cross-attention to obs/RAE conditioning tokens  [iRDT]
        x2 = self.norm_cross(x)
        x2, _ = self.cross_attn(x2, cond, cond, need_weights=False)
        x = x + self.drop_cross(x2)

        # 3. MLP with AdaLN from timestep
        x2 = self.mlp_mod1(self.norm2(x), t)
        x2 = self.linear2(self.drop_mlp(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.drop_gate(x2), t)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for mod in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            mod.reset_parameters()


# ---------------------------------------------------------------------------
# Lightning DiT block  [RAE-DiT]
# ---------------------------------------------------------------------------

class _LightningDiTBlock(nn.Module):
    """Lightning DiT block from the RAE-DiT paper.

    Key difference from the standard block: adaLN-Zero parameterization.
    A single SiLU → Linear maps the timestep embedding to *all six* modulation
    parameters at once:

        (shift₁, scale₁, gate₁,  shift₂, scale₂, gate₂)
          ↑ self-attn pre-norm ↑   ↑   MLP pre-norm    ↑

    The final Linear is **zero-initialized**, so at the start of training every
    block is an identity transformation — this stabilizes early optimization
    and is the defining feature of the Lightning variant.

    Layer norms for self-attn and MLP have `elementwise_affine=False` since
    adaLN handles the affine transform. The cross-attention norm is a plain
    LayerNorm (cross-attn has no adaLN; conditioning comes through attention).

    Like _DiTCrossAttnBlock, `cond` can be per-depth encoder outputs (iRDT
    multi-scale) or external RAE latents (RAE-DiT).
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # No affine params — adaLN-Zero supplies them
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_cross = nn.LayerNorm(d_model)   # plain: no adaLN on cross-attn
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # adaLN-Zero: one MLP → 6 params; zero-init → identity at init  [RAE-DiT]
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        self.drop_self = nn.Dropout(dropout)
        self.drop_cross = nn.Dropout(dropout)
        self.drop_mlp = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, x, t, cond):
        # Produce all 6 modulation params from a single pass  [RAE-DiT]
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(t).chunk(6, dim=1)

        # 1. Self-attention with adaLN-Zero
        x2 = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = x + gate1.unsqueeze(1) * self.drop_self(x2)

        # 2. Cross-attention to obs/RAE conditioning tokens  [iRDT]
        x2 = self.norm_cross(x)
        x2, _ = self.cross_attn(x2, cond, cond, need_weights=False)
        x = x + self.drop_cross(x2)

        # 3. MLP with adaLN-Zero
        x2 = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x2 = self.linear2(self.drop_mlp(self.activation(self.linear1(x2))))
        x = x + gate2.unsqueeze(1) * x2
        return x

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Re-zero the adaLN output projection after xavier sweep
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)


# ---------------------------------------------------------------------------
# Dasari DiT block  [DiT-Block Policy — adaLN-Zero, NO cross-attention]
# ---------------------------------------------------------------------------

class _DasariDiTBlock(nn.Module):
    """DiT-Block Policy decoder block (Dasari et al. 2024).

    adaLN-Zero modulation from a conditioning VECTOR (not token sequence).
    No cross-attention — Dasari found cross-attention "catastrophically
    unstable" for diffusion policy training and removed it entirely.

    Conditioning enters only through LayerNorm shift/scale/gate:
        cond_vec = mean_pool(encoder_output_at_depth_d) + time_embedding

    Two operations with adaLN-Zero:
      1. Self-attention on action tokens
      2. MLP

    Both use zero-initialized gates for identity initialization.
    """

    def __init__(
        self, d_model, nhead, dim_feedforward=1024, dropout=0.0, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # No affine params — adaLN-Zero supplies them
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # adaLN-Zero: one MLP → 6 params; zero-init → identity at init
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

        self.activation = _get_activation_fn(activation)

    def forward(self, x, cond_vec):
        """
        Args:
            x: (B, T_pred, d_model) action tokens.
            cond_vec: (B, d_model) conditioning vector
                      (mean_pool(encoder_output) + time_embedding).
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(cond_vec).chunk(6, dim=1)

        # 1. Self-attention with adaLN-Zero
        x2 = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = x + gate1.unsqueeze(1) * x2

        # 2. MLP with adaLN-Zero (NO cross-attention)
        x2 = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x2 = self.linear2(self.activation(self.linear1(x2)))
        x = x + gate2.unsqueeze(1) * x2
        return x

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)


# ---------------------------------------------------------------------------
# Noise network V2  [Dasari-style: adaLN-Zero decoder, no cross-attention]
# ---------------------------------------------------------------------------

class _DiTNoiseNetV2(nn.Module):
    """Diffusion Transformer noise network V2 — Dasari-style decoder.

    Key difference from V1: decoder blocks use adaLN-Zero conditioning
    from a mean-pooled vector, NOT cross-attention to encoder tokens.

    Encoder: same self-attention encoder as V1, producing per-depth outputs.
    Decoder: _DasariDiTBlock layers, each conditioned on:
        cond_vec = mean_pool(enc_outputs[d]) + time_embedding

    This matches Dasari et al. (2024) "DiT-Block Policy" which proved that
    adaLN-Zero conditioning is far more stable than cross-attention for
    robotic diffusion policy training.
    """

    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=128,
        hidden_dim=256,
        num_blocks=4,
        dropout=0.0,
        dim_feedforward=1024,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()

        # Positional encodings
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(1, ac_chunk, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # Input projections
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # Obs self-attention encoder (same as V1)
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # Dasari-style decoder: adaLN-Zero only, no cross-attention
        decoder_module = _DasariDiTBlock(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_module) for _ in range(num_blocks)]
        )
        for layer in self.decoder:
            layer.reset_parameters()

        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)

    def forward_enc(self, obs_enc):
        """Encode obs tokens. Returns list of per-depth tensors."""
        pos = self.enc_pos(obs_enc)
        return self.encoder(obs_enc, pos)

    def forward_dec(self, noise_actions, time, enc_cache):
        """Denoise action tokens with Dasari-style adaLN-Zero conditioning.

        Each decoder layer d receives:
            cond_vec = mean_pool(enc_cache[d]) + time_embedding
        """
        time_enc = self.time_net(time)
        ac_tokens = self.ac_proj(noise_actions) + self.dec_pos
        x = ac_tokens

        if isinstance(enc_cache, list):
            for layer, cond_tokens in zip(self.decoder, enc_cache):
                cond_vec = torch.mean(cond_tokens, dim=1) + time_enc  # (B, d)
                x = layer(x, cond_vec)
            final_cond = enc_cache[-1]
        else:
            cond_vec = torch.mean(enc_cache, dim=1) + time_enc
            for layer in self.decoder:
                x = layer(x, cond_vec)
            final_cond = enc_cache

        return self.eps_out(x, time_enc, final_cond)


# ---------------------------------------------------------------------------
# Noise network
# ---------------------------------------------------------------------------

class _DiTNoiseNet(nn.Module):
    """Diffusion Transformer noise prediction network.

    Merges iRDT and RAE-DiT:

    Encoder  (iRDT multi-scale coupling)
    ─────────────────────────────────────────────────
    Obs tokens → self-attention encoder (N layers) → per-depth feature list.
    The feature at depth d feeds the decoder block at depth d via cross-attention.
    When using external conditioning (e.g. RAE latents), pass a pre-computed
    tensor as `enc_cache` to skip the internal encoder entirely.

    Decoder  (iRDT cross-attention  +  RAE-DiT Lightning option)
    ─────────────────────────────────────────────────────────────
    Noisy action tokens + timestep → N decoder blocks → ε prediction.

    Each decoder block is either:
      _DiTCrossAttnBlock  —  standard AdaLN modulation  (iRDT)
      _LightningDiTBlock  —  adaLN-Zero unified modulation  (RAE-DiT)

    Set `use_lightning=True` to use the Lightning variant.

    The `enc_cache` returned by `forward` / `forward_enc` is always a list of
    per-depth tensors when the internal encoder is used, or whatever tensor
    was supplied as external conditioning. `forward_dec` handles both.
    """

    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        use_lightning=True,
    ):
        super().__init__()

        # positional encodings
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(1, ac_chunk, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input projections
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # obs self-attention encoder  [iRDT multi-scale]
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks: standard (iRDT) or lightning (RAE-DiT)
        if use_lightning:
            decoder_module = _LightningDiTBlock(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
        else:
            decoder_module = _DiTCrossAttnBlock(
                hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
        self.decoder = nn.ModuleList(
            [copy.deepcopy(decoder_module) for _ in range(num_blocks)]
        )
        for layer in self.decoder:
            layer.reset_parameters()

        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        print(
            "number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)

    def forward_enc(self, obs_enc):
        """Encode obs tokens through the self-attention encoder.

        Returns a list of per-depth tensors for multi-scale conditioning  [iRDT].
        Pass the returned list (or any single tensor) as `enc_cache` to `forward_dec`.
        """
        pos = self.enc_pos(obs_enc)
        return self.encoder(obs_enc, pos)   # list[depth] of (B, seq_len, hidden_dim)

    def _forward_dec_inner(self, noise_actions, time, enc_cache):
        """Run the decoder and return pre-output state (x, time_enc, final_cond).

        Splitting here lets _DDTHead consume the backbone's intermediate token
        features zt before the final projection — the DDT head formula requires:
            zt = M(xt | t, y)   ← this method returns zt as `x`
            vt = H(xt | zt, t)  ← _DDTHead uses zt as cross-attention K/V
        """
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noise_actions) + self.dec_pos  # (B, ac_chunk, hidden_dim)
        x = ac_tokens

        if isinstance(enc_cache, list):
            # iRDT multi-scale: pair each decoder layer with its encoder counterpart
            for layer, cond in zip(self.decoder, enc_cache):
                x = layer(x, time_enc, cond)
            final_cond = enc_cache[-1]
        else:
            # RAE-DiT single conditioning: broadcast the same latents to every layer
            for layer in self.decoder:
                x = layer(x, time_enc, enc_cache)
            final_cond = enc_cache

        return x, time_enc, final_cond

    def forward_dec(self, noise_actions, time, enc_cache):
        """Denoise action tokens conditioned on enc_cache.

        enc_cache may be:
          list  — multi-scale encoder outputs  [iRDT]: decoder depth d gets enc_cache[d]
          tensor — single conditioning tensor (e.g. external RAE latents  [RAE-DiT]):
                   all decoder layers receive the same tensor
        """
        x, time_enc, final_cond = self._forward_dec_inner(noise_actions, time, enc_cache)
        return self.eps_out(x, time_enc, final_cond)


# ---------------------------------------------------------------------------
# Dimension-dependent noise schedule shift  [RAE-DiT]
# ---------------------------------------------------------------------------

def _shift_alphas_cumprod(alphas_cumprod, num_tokens, ref_tokens=1):
    """Shift the noise schedule based on latent token count.

    From RAE-DiT: higher-dimensional latents have higher-entropy marginals,
    so more noise is needed to destroy the signal completely. The shift is:

        log(SNR) → log(SNR) + 2·log(num_tokens / ref_tokens)

    which scales the terminal SNR by (num_tokens / ref_tokens)².
    ref_tokens=1 treats a single token as the baseline (no shift).

    Args:
        alphas_cumprod: (T,) tensor of ᾱ values from the scheduler
        num_tokens:     number of RAE latent tokens  (m in the paper)
        ref_tokens:     reference token count  (default 1)
    Returns:
        (T,) shifted ᾱ values — drop-in replacement for the scheduler's buffer
    """
    if num_tokens == ref_tokens:
        return alphas_cumprod
    log_snr = torch.log(alphas_cumprod / (1.0 - alphas_cumprod))
    log_snr = log_snr + 2.0 * np.log(num_tokens / ref_tokens)
    return torch.sigmoid(log_snr)


# ---------------------------------------------------------------------------
# DDT Head  [RAE-DiT Section 5]
# ---------------------------------------------------------------------------

class _DDTHead(nn.Module):
    """Wide, shallow DDT head from RAE-DiT Section 5.

    Implements:  vt = H(xt | zt, t)

    where xt are the noisy action tokens and zt are the backbone's
    intermediate token features (output of _DiTNoiseNet._forward_dec_inner).

    Design:
      - Wide hidden dim (head_dim >> backbone hidden_dim, default 2048)
      - Shallow depth (default 2 layers)
      - Lightning DiT blocks: xt tokens self-attend, then cross-attend to zt
      - zt is projected from backbone_dim → head_dim before K/V
      - Separate time_net so the head can operate at a different width
    """

    def __init__(
        self,
        ac_dim,
        ac_chunk,
        backbone_dim,
        time_dim=256,
        head_dim=2048,
        num_blocks=2,
        dropout=0.1,
        dim_feedforward=4096,
        nhead=16,
        activation="gelu",
    ):
        super().__init__()

        # Project noisy actions into head dimension
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, head_dim),
        )

        # Learnable positional embeddings for action tokens
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(1, ac_chunk, head_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # Project backbone token features zt into head_dim for cross-attention K/V
        self.zt_proj = nn.Linear(backbone_dim, head_dim)

        # Separate timestep embedding at head width
        self.time_net = _TimeNetwork(time_dim, head_dim)

        # Wide shallow Lightning DiT blocks
        block = _LightningDiTBlock(
            head_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(num_blocks)]
        )
        for b in self.blocks:
            b.reset_parameters()

        # Final output projection
        self.eps_out = _FinalLayer(head_dim, ac_dim)

        print(
            "number of DDT head parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(self, noise_actions, time, zt):
        """
        Args:
            noise_actions: (B, ac_chunk, ac_dim)
            time:          (B,) integer timesteps
            zt:            (B, ac_chunk, backbone_dim) — backbone intermediate tokens
        Returns:
            (B, ac_chunk, ac_dim) predicted noise / velocity
        """
        time_enc = self.time_net(time)

        # Project xt and zt into head dimension
        x = self.ac_proj(noise_actions) + self.dec_pos  # (B, ac_chunk, head_dim)
        cond = self.zt_proj(zt)                         # (B, ac_chunk, head_dim)

        for block in self.blocks:
            x = block(x, time_enc, cond)

        return self.eps_out(x, time_enc, cond)


# ---------------------------------------------------------------------------
# DiTDH: backbone + DDT head  [RAE-DiT Section 5]
# ---------------------------------------------------------------------------

class _DiTDHNoiseNet(nn.Module):
    """DiTDH — base DiT backbone augmented with a wide DDT head.

    Implements the two-stage prediction from RAE-DiT Section 5:
        zt = M(xt | t, y)   — backbone (any _DiTNoiseNet)
        vt = H(xt | zt, t)  — DDT head (_DDTHead)

    Exposes the same forward / forward_enc / forward_dec interface as
    _DiTNoiseNet so DiffusionTransformerAgent works unchanged.
    """

    def __init__(self, backbone: "_DiTNoiseNet", head: _DDTHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)

    def forward_enc(self, obs_enc):
        return self.backbone.forward_enc(obs_enc)

    def forward_dec(self, noise_actions, time, enc_cache):
        # Stage 1: backbone intermediate features
        zt, _, _ = self.backbone._forward_dec_inner(
            noise_actions, time, enc_cache
        )
        # Stage 2: DDT head  H(xt | zt, t)
        return self.head(noise_actions, time, zt)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DiffusionTransformerAgent(BasePolicy):
    def __init__(
        self,
        features,
        feat_dim,
        token_dim,
        n_cams,
        odim,
        ac_dim,
        ac_chunk,
        train_diffusion_steps,
        eval_diffusion_steps,
        use_obs=False,
        imgs_per_cam=1,
        dropout=0,
        share_cam_features=False,
        view_dropout_p=0.0,
        feat_norm=None,
        max_patches_per_view=None,
        noise_net_kwargs=dict(),
        ddt_head_kwargs=dict(
            head_dim=2048,
            num_blocks=2,
            nhead=16,
            dim_feedforward=4096,
        ),
        latent_num_tokens=None,
        latent_dim=None,
        width_mult=None,
    ):
        super().__init__(
            features=features,
            feat_dim=feat_dim,
            token_dim=token_dim,
            n_cams=n_cams,
            odim=odim,
            use_obs=use_obs,
            imgs_per_cam=imgs_per_cam,
            share_cam_features=share_cam_features,
            view_dropout_p=view_dropout_p,
            dropout=dropout,
            feat_norm=feat_norm,
            max_patches_per_view=max_patches_per_view,
        )

        # Width scaling: hidden_dim ∝ latent_dim  [RAE-DiT]
        # Ensures the backbone has sufficient capacity for the latent dimensionality.
        if latent_dim is not None and width_mult is not None:
            noise_net_kwargs = dict(noise_net_kwargs)
            hidden_dim = int(width_mult * latent_dim)
            hidden_dim = max(64, (hidden_dim + 63) // 64 * 64)  # round to multiple of 64
            noise_net_kwargs.setdefault("hidden_dim", hidden_dim)

        backbone = _DiTNoiseNet(
            ac_dim=ac_dim,
            ac_chunk=ac_chunk,
            **noise_net_kwargs,
        )

        if ddt_head_kwargs is not None:
            # DiTDH: attach a wide DDT head to the backbone  [RAE-DiT Section 5]
            head = _DDTHead(
                ac_dim=ac_dim,
                ac_chunk=ac_chunk,
                backbone_dim=noise_net_kwargs.get("hidden_dim", 512),
                **ddt_head_kwargs,
            )
            self.noise_net = _DiTDHNoiseNet(backbone, head)
        else:
            self.noise_net = backbone
        self._ac_dim, self._ac_chunk = ac_dim, ac_chunk

        assert (
            eval_diffusion_steps <= train_diffusion_steps
        ), "Can't eval with more steps!"
        self._train_diffusion_steps = train_diffusion_steps
        self._eval_diffusion_steps = eval_diffusion_steps
        self.diffusion_schedule = DDIMScheduler(
            num_train_timesteps=train_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

        # Dimension-dependent noise schedule shift  [RAE-DiT]
        # Shifts log(SNR) by 2·log(latent_num_tokens) to compensate for the
        # higher-entropy marginal distribution of multi-token latents.
        if latent_num_tokens is not None:
            self.diffusion_schedule.alphas_cumprod = _shift_alphas_cumprod(
                self.diffusion_schedule.alphas_cumprod,
                num_tokens=latent_num_tokens,
            )

    def forward(self, imgs, obs, ac_flat, mask_flat):
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        timesteps = torch.randint(
            low=0, high=self._train_diffusion_steps, size=(B,), device=device
        ).long()

        mask = mask_flat.reshape((B, self.ac_chunk, self.ac_dim))
        actions = ac_flat.reshape((B, self.ac_chunk, self.ac_dim))
        noise = torch.randn_like(actions)

        noise_acs = self.diffusion_schedule.add_noise(actions, noise, timesteps)
        _, noise_pred = self.noise_net(noise_acs, timesteps, s_t)

        loss = nn.functional.mse_loss(noise_pred, noise, reduction="none")
        loss = (loss * mask).sum(1)
        return loss.mean()

    def get_actions(self, imgs, obs, n_steps=None):
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        noise_actions = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)

        eval_steps = self._eval_diffusion_steps
        if n_steps is not None:
            assert (
                n_steps <= self._train_diffusion_steps
            ), f"can't be > {self._train_diffusion_steps}"
            eval_steps = n_steps

        # encode obs once; reuse the cached features across all denoising steps
        enc_cache = self.noise_net.forward_enc(s_t)

        self.diffusion_schedule.set_timesteps(eval_steps)
        self.diffusion_schedule.alphas_cumprod = (
            self.diffusion_schedule.alphas_cumprod.to(device)
        )
        for timestep in self.diffusion_schedule.timesteps:
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(device)
            noise_pred = self.noise_net.forward_dec(
                noise_actions, batched_timestep, enc_cache
            )
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample

        return noise_actions

    @property
    def ac_chunk(self):
        return self._ac_chunk

    @property
    def ac_dim(self):
        return self._ac_dim

