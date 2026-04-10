"""V3 Observation Encoder — supports both average pooling and spatial tokens.

Two modes controlled by spatial_pool_size:
  S=1 (default): Chi's exact design — avg pool → flat concat → denoiser projects.
    Memory = [timestep, obs_t0, ..., obs_{T_o-1}] = 1+T_o tokens.
  S>1 (spatial): Per-camera S×S pooling → project to d_model → camera + spatial embeddings.
    Memory = [timestep, spatial_tokens..., proprio_tokens...] = 1 + T_o*(n_active*S*S + 1).

Input:
  adapted_tokens: (B, T_o, K, 196, adapter_dim)  from Stage1Bridge
  proprio:        (B, T_o, proprio_dim)
  view_present:   (B, K) bool

Output dict (S=1):
  'tokens': (B, T_o, concat_dim)  — raw concat, projected by denoiser's cond_obs_emb
  'global': (B, d_model)          — for U-Net FiLM (optional)

Output dict (S>1):
  'tokens': (B, N_memory, d_model)  — pre-projected spatial + proprio tokens
  'pre_projected': True              — tells denoiser to skip cond_obs_emb
  'global': (B, d_model)            — zeros (unused for cross-attn transformer)

Owner: Swagman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Chi's SpatialSoftmax adapted for ViT patch grids.

    For each channel, computes soft-argmax over 2D positions → (x, y).
    Input:  (B, D, H, W) feature map
    Output: (B, D*2) expected coordinates
    """

    def __init__(self, H: int = 14, W: int = 14, temperature: float = 1.0):
        super().__init__()
        self.H, self.W = H, W
        self.temperature = temperature

        # Normalized coordinate grids [-1, 1]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, W),
            torch.linspace(-1, 1, H),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))  # (H*W,)
        self.register_buffer("pos_y", pos_y.reshape(-1))  # (H*W,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D, H, W) → (B, D*2)"""
        B, D = x.shape[:2]
        flat = x.reshape(B, D, -1)  # (B, D, H*W)
        attn = torch.softmax(flat / self.temperature, dim=-1)  # (B, D, H*W)

        exp_x = (attn * self.pos_x).sum(dim=-1)  # (B, D)
        exp_y = (attn * self.pos_y).sum(dim=-1)  # (B, D)

        return torch.cat([exp_x, exp_y], dim=-1)  # (B, D*2)


class ObservationEncoder(nn.Module):
    """Encodes adapted visual tokens + proprio into conditioning vectors.

    Args:
        adapter_dim:       Adapter output dimension (512).
        d_model:           Transformer hidden dimension.
        proprio_dim:       Proprioceptive state dimension.
        T_obs:             Observation horizon.
        n_active_cams:     Number of ACTIVE cameras (2 for robomimic, 4 for RLBench).
        spatial_pool_size: Pool each camera's 14×14 patches to S×S. 1 = avg pool (default).
    """

    def __init__(
        self,
        adapter_dim: int = 512,
        d_model: int = 256,
        proprio_dim: int = 9,
        T_obs: int = 2,
        n_active_cams: int = 2,
        spatial_pool_size: int = 1,
        use_spatial_softmax: bool = False,
    ):
        super().__init__()
        self.adapter_dim = adapter_dim
        self.d_model = d_model
        self.T_obs = T_obs
        self.n_active_cams = n_active_cams
        self.S = spatial_pool_size
        self.use_spatial_softmax = use_spatial_softmax

        if use_spatial_softmax:
            # --- SpatialSoftmax path: Chi-faithful spatial coordinates ---
            # 512 channels × 2 coords = 1024D per camera
            self.spatial_softmax = SpatialSoftmax(H=14, W=14)
            self.output_dim = n_active_cams * (adapter_dim * 2) + proprio_dim
            self.global_proj = nn.Sequential(
                nn.Linear(T_obs * self.output_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        elif spatial_pool_size == 1:
            # --- Legacy path: Chi's exact design ---
            self.output_dim = n_active_cams * adapter_dim + proprio_dim
            self.global_proj = nn.Sequential(
                nn.Linear(T_obs * self.output_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            # --- Spatial token path ---
            self.output_dim = d_model  # each token is d_model-dim
            self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_pool_size)
            self.token_proj = nn.Linear(adapter_dim, d_model)
            self.cam_embed = nn.Embedding(8, d_model)  # up to 8 cameras
            self.spatial_pos_embed = nn.Parameter(
                torch.randn(1, spatial_pool_size ** 2, d_model) * 0.02
            )
            self.proprio_proj = nn.Linear(proprio_dim, d_model)

    def _encode_single_cam_spatial(self, tokens, cam_idx):
        """Encode one camera's adapted tokens to S×S spatial tokens.

        Args:
            tokens: (B, 196, adapter_dim) — adapted tokens for one camera, one timestep
            cam_idx: int — camera index for camera embedding

        Returns:
            (B, S*S, d_model) — spatial tokens with camera + position embeddings
        """
        B, N, D = tokens.shape
        H = W = int(N ** 0.5)  # 14

        # Reshape to 2D grid → pool → flatten
        t2d = tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, 14, 14)
        pooled = self.spatial_pool(t2d)                          # (B, D, S, S)
        spatial = pooled.permute(0, 2, 3, 1).reshape(B, -1, D)  # (B, S*S, D)

        # Project to d_model
        spatial = self.token_proj(spatial)  # (B, S*S, d_model)

        # Add camera embedding + spatial position embedding
        cam_id = torch.tensor(cam_idx, device=tokens.device)
        spatial = spatial + self.cam_embed(cam_id) + self.spatial_pos_embed

        return spatial

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
            dict — see module docstring for output shapes per mode
        """
        B, T_o, K = adapted_tokens.shape[:3]

        if self.use_spatial_softmax or self.S == 1:
            return self._forward_legacy(adapted_tokens, proprio, view_present, B, T_o, K)
        else:
            return self._forward_spatial(adapted_tokens, proprio, view_present, B, T_o, K)

    def _forward_legacy(self, adapted_tokens, proprio, view_present, B, T_o, K):
        """Chi's original path: avg pool or SpatialSoftmax → flat concat per timestep."""
        if self.use_spatial_softmax:
            # SpatialSoftmax: (B, T_o, K, 196, D) → (B, T_o, K, D*2)
            N, D = adapted_tokens.shape[3], adapted_tokens.shape[4]
            H = W = int(N ** 0.5)  # 14
            flat = adapted_tokens.reshape(B * T_o * K, N, D)
            feat_2d = flat.reshape(B * T_o * K, H, W, D).permute(0, 3, 1, 2)  # (B*T_o*K, D, H, W)
            ss = self.spatial_softmax(feat_2d)  # (B*T_o*K, D*2)
            pooled = ss.reshape(B, T_o, K, D * 2)
            per_cam_dim = D * 2  # 1024
        else:
            # Average pool: (B, T_o, K, N, D) → (B, T_o, K, D)
            pooled = adapted_tokens.mean(dim=3)
            per_cam_dim = self.adapter_dim  # 512

        # 2. Select only active camera features
        active_features = []
        for k in range(K):
            if view_present[:, k].any():
                active_features.append(pooled[:, :, k, :])  # (B, T_o, per_cam_dim)

        # Stack active cameras: (B, T_o, n_active, per_cam_dim)
        if len(active_features) > 0:
            active = torch.stack(active_features, dim=2)
        else:
            active = pooled[:, :, :1, :]

        # 3. Flatten active views per timestep: (B, T_o, n_active * per_cam_dim)
        n_active = active.shape[2]
        active_flat = active.reshape(B, T_o, n_active * per_cam_dim)

        # 4. Concatenate proprio: (B, T_o, n_active * adapter_dim + proprio_dim)
        obs_concat = torch.cat([active_flat, proprio], dim=-1)

        # 5. Global conditioning vector (for U-Net option)
        global_vec = self.global_proj(obs_concat.reshape(B, -1))

        return {
            "tokens": obs_concat,
            "global": global_vec,
        }

    def _forward_spatial(self, adapted_tokens, proprio, view_present, B, T_o, K):
        """Spatial token path: per-camera S×S tokens + proprio token per timestep."""
        all_tokens = []

        for t in range(T_o):
            # Spatial tokens from each active camera
            for k in range(K):
                if not view_present[:, k].any():
                    continue
                cam_tokens = adapted_tokens[:, t, k]  # (B, 196, adapter_dim)
                spatial = self._encode_single_cam_spatial(cam_tokens, k)  # (B, S*S, d_model)
                all_tokens.append(spatial)

            # Proprio token for this timestep
            proprio_t = self.proprio_proj(proprio[:, t])  # (B, d_model)
            all_tokens.append(proprio_t.unsqueeze(1))     # (B, 1, d_model)

        # Concatenate all tokens: (B, N_memory, d_model)
        memory = torch.cat(all_tokens, dim=1)

        return {
            "tokens": memory,
            "pre_projected": True,
            "global": torch.zeros(B, self.d_model, device=memory.device),
        }
