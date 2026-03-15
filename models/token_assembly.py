"""TokenAssembly — C.4.

Adds spatial, view, and observation-timestep embeddings to adapted tokens,
embeds proprioceptive state via MLP, and concatenates everything into a
flat sequence ready for the observation encoder.

Output is (B, S_obs, d') batch-first, directly compatible with
_TransformerEncoder (after C.0 batch-first conversion).

S_obs = T_o * K * N + T_o   (visual tokens + proprio tokens, fixed length)
"""

import torch
import torch.nn as nn


class TokenAssembly(nn.Module):
    """Assemble visual + proprio tokens with all embeddings.

    Args:
        d_model: Working dimension (adapter output / transformer dim).
        num_patches: Patches per view (196 for 14x14).
        num_views: Maximum camera slots.
        num_obs_steps: Observation horizon T_o.
        proprio_dim: Proprioceptive state dimension.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_patches: int = 196,
        num_views: int = 4,
        num_obs_steps: int = 2,
        proprio_dim: int = 9,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.num_views = num_views
        self.num_obs_steps = num_obs_steps

        # Spatial position embedding: shared across views and timesteps
        self.spatial_pos_emb = nn.Parameter(
            torch.randn(1, num_patches, d_model) * 0.02
        )

        # View embedding: distinguishes cameras
        self.view_emb = nn.Embedding(num_views, d_model)

        # Observation timestep embedding: distinguishes T_o frames
        self.obs_time_emb = nn.Embedding(num_obs_steps, d_model)

        # View presence embedding: distinguishes real vs masked views
        self.view_present_emb = nn.Embedding(2, d_model)  # 0=masked, 1=real

        # Proprio projection: MLP maps proprio to a single token
        self.proprio_proj = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        proprio: torch.Tensor,
        view_present: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble all tokens into a flat observation sequence.

        Args:
            adapted_tokens: (B, T_o, K, N, d') adapted visual tokens
                (after view dropout — dropped views filled with mask token).
            proprio: (B, T_o, D_prop) normalized proprioceptive state.
            view_present: (B, K) bool mask (after view dropout).

        Returns:
            obs_tokens: (B, S_obs, d') flat sequence for the encoder.
                S_obs = T_o * K_present * N + T_o
                where K_present is the number of present views per sample.

        Note:
            To keep batching simple, we include ALL K views (present ones
            have real tokens + embeddings, absent ones have mask tokens).
            This gives a fixed sequence length S_obs = T_o * K * N + T_o.
        """
        B, T_o, K, N, d = adapted_tokens.shape
        device = adapted_tokens.device
        all_tokens = []

        for t in range(T_o):
            time_emb = self.obs_time_emb(
                torch.tensor(t, device=device)
            )  # (d',)

            for k in range(K):
                # Visual tokens for this (timestep, view): (B, N, d')
                vis = adapted_tokens[:, t, k]  # (B, N, d')

                # Add spatial + view + timestep embeddings
                vis = vis + self.spatial_pos_emb  # broadcast (1, N, d')
                vis = vis + self.view_emb(
                    torch.tensor(k, device=device)
                )  # broadcast (d',)
                vis = vis + time_emb  # broadcast (d',)

                # Add view presence embedding (0=masked, 1=real)
                vp_flag = view_present[:, k].long()  # (B,)
                vis = vis + self.view_present_emb(vp_flag).unsqueeze(1)  # (B, 1, d')

                all_tokens.append(vis)

            # Proprio token for this timestep: (B, 1, d')
            prop_token = self.proprio_proj(proprio[:, t])  # (B, d')
            prop_token = prop_token + time_emb  # add timestep embedding
            all_tokens.append(prop_token.unsqueeze(1))

        # Concatenate: (B, T_o * K * N + T_o, d')
        obs_tokens = torch.cat(all_tokens, dim=1)
        return obs_tokens
