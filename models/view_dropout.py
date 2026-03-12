"""C.3 ViewDropout — drop entire camera views during training.

Replaces all tokens of a dropped view with a learned mask token.
Applied AFTER adapter, BEFORE token assembly.
No-op at eval or when p_drop=0.
"""

import torch
import torch.nn as nn


class ViewDropout(nn.Module):
    """Replace entire camera view tokens with a learned mask token.

    Args:
        d_model: Token dimension (adapter output dim).
        p_drop:  Per-view drop probability during training. 0 disables.
    """

    def __init__(self, d_model: int = 512, p_drop: float = 0.15):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.p_drop = p_drop

    def forward(
        self,
        adapted_tokens: torch.Tensor,
        view_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adapted_tokens: [B, K, N, d] — adapted visual tokens per view.
            view_present:   [B, K] bool — which views are real (not padding).

        Returns:
            masked_tokens:       [B, K, N, d] — dropped views filled with mask_token.
            updated_view_present: [B, K] bool — dropped views set to False.
        """
        if not self.training or self.p_drop == 0.0:
            return adapted_tokens, view_present

        B, K, N, d = adapted_tokens.shape

        # Sample independent Bernoulli drop mask for each (batch, view)
        drop_mask = torch.rand(B, K, device=adapted_tokens.device) < self.p_drop

        # Only drop views that are actually present
        drop_mask = drop_mask & view_present

        # Safety: never drop ALL views for any sample — keep at least one
        all_dropped = (view_present & ~drop_mask).sum(dim=1) == 0  # [B]
        if all_dropped.any():
            # For samples where all views would be dropped, un-drop one random present view
            for b in all_dropped.nonzero(as_tuple=True)[0]:
                present_indices = view_present[b].nonzero(as_tuple=True)[0]
                keep = present_indices[torch.randint(len(present_indices), (1,))]
                drop_mask[b, keep] = False

        # Apply mask: replace dropped view tokens with mask_token
        out = adapted_tokens.clone()
        # drop_mask is [B, K], expand to [B, K, N, d]
        drop_expanded = drop_mask[:, :, None, None].expand_as(out)
        out = torch.where(drop_expanded, self.mask_token, out)

        # Update view_present
        updated_vp = view_present.clone()
        updated_vp[drop_mask] = False

        return out, updated_vp
