"""V3 Observation Encoder — matches Chi's conditioning structure.

Chi's approach: active camera features + proprio are concatenated into ONE flat
vector per timestep, then projected to d_model. This produces T_o conditioning
tokens (one per observation timestep), NOT separate tokens per view.

Key differences from previous version:
  - Only concatenates ACTIVE camera features (not all K slots with zeros)
  - LayerNorm on pooled features for scale normalization
  - For robomimic (2 cameras): concat_dim = 2*512 + 9 = 1033 (matches Chi)
  - For RLBench (4 cameras): concat_dim = 4*512 + 8 = 2056

Memory = [timestep_token, obs_t0, obs_t1] = 3 tokens for T_o=2.

Input:
  adapted_tokens: (B, T_o, K, 196, adapter_dim)  from Stage1Bridge
  proprio:        (B, T_o, proprio_dim)
  view_present:   (B, K) bool
  n_active_cams:  int — number of active cameras (for fixed projection dim)

Output dict:
  'tokens': (B, T_o, d_model)    — one token per observation timestep
  'global': (B, d_model)         — for U-Net FiLM (optional)
"""

import torch
import torch.nn as nn


class ObservationEncoder(nn.Module):
    """Encodes adapted visual tokens + proprio into conditioning tokens.

    Matches Chi's pipeline: pool each view → LayerNorm → concat active views
    + proprio per timestep → project to d_model.

    Args:
        adapter_dim:    Adapter output dimension (512).
        d_model:        Transformer hidden dimension (256).
        proprio_dim:    Proprioceptive state dimension.
        T_obs:          Observation horizon.
        n_active_cams:  Number of ACTIVE cameras (2 for robomimic, 4 for RLBench).
                        Only active camera features are concatenated.
    """

    def __init__(
        self,
        adapter_dim: int = 512,
        d_model: int = 256,
        proprio_dim: int = 9,
        T_obs: int = 2,
        n_active_cams: int = 2,
    ):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.d_model = d_model
        self.T_obs = T_obs
        self.n_active_cams = n_active_cams

        # LayerNorm on pooled features (scale normalization)
        self.feature_norm = nn.LayerNorm(adapter_dim)

        # Project concatenated [active_views + proprio] to d_model
        # Chi: 2 cameras → 1024 + 9 = 1033 → nn.Linear(1033, 256)
        concat_dim = n_active_cams * adapter_dim + proprio_dim
        self.obs_proj = nn.Linear(concat_dim, d_model)

        # Global conditioning vector for U-Net FiLM
        self.global_proj = nn.Sequential(
            nn.Linear(T_obs * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        proprio: torch.Tensor,
        view_present: torch.Tensor,
    ) -> dict:
        """
        Args:
            adapted_tokens: (B, T_o, K, N_patches, adapter_dim) — from Stage1Bridge
            proprio:        (B, T_o, proprio_dim)
            view_present:   (B, K) bool — which cameras are active

        Returns:
            dict with:
                'tokens': (B, T_o, d_model) — one conditioning token per timestep
                'global': (B, d_model) — for U-Net FiLM conditioning
        """
        B, T_o, K = adapted_tokens.shape[:3]

        # 1. Spatial average pool: (B, T_o, K, N, D) → (B, T_o, K, D)
        pooled = adapted_tokens.mean(dim=3)

        # 2. LayerNorm on pooled features (scale normalization)
        pooled = self.feature_norm(pooled)

        # 3. Select only active camera features
        # view_present is (B, K) — select columns where any sample has True
        # For robomimic: slots 0,3 are active → select those 2
        active_features = []
        for k in range(K):
            if view_present[:, k].any():
                active_features.append(pooled[:, :, k, :])  # (B, T_o, D)

        # Stack active cameras: (B, T_o, n_active, D)
        if len(active_features) > 0:
            active = torch.stack(active_features, dim=2)
        else:
            # Fallback: no cameras active (shouldn't happen)
            active = pooled[:, :, :1, :]

        # 4. Flatten active views per timestep: (B, T_o, n_active * adapter_dim)
        n_active = active.shape[2]
        active_flat = active.reshape(B, T_o, n_active * self.adapter_dim)

        # 5. Concatenate proprio: (B, T_o, n_active * adapter_dim + proprio_dim)
        obs_concat = torch.cat([active_flat, proprio], dim=-1)

        # 6. Project to d_model: (B, T_o, d_model)
        obs_tokens = self.obs_proj(obs_concat)

        # 7. Global conditioning vector (for U-Net option)
        global_vec = self.global_proj(obs_tokens.reshape(B, -1))

        return {
            "tokens": obs_tokens,
            "global": global_vec,
        }
