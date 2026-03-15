"""ViewDropout — C.3 (Swagman).

Randomly replaces entire camera views with a learned mask token during
training, making the policy robust to missing cameras at inference time.

Design (per spec):
  - Learned mask token: nn.Parameter of shape (d_model,), init N(0, 0.02²)
  - Input:  [B, K, N, d'] adapted tokens (after backbone + adapter)
  - Output: (masked_tokens [B, K, N, d'], updated_view_present [B, K] bool)
  - Does NOT drop already-absent views (view_present gates the drop logic)
  - At least one present view is always kept

Usage::
    vd = ViewDropout(d_model=512, p=0.1)
    tokens, view_present = vd(tokens, view_present)
"""

import torch
import torch.nn as nn


class ViewDropout(nn.Module):
    """Replace entire camera views with a learned mask token during training.

    Args:
        d_model: Token dimension (must match adapter output dim).
        p:       Per-view drop probability for views that ARE present.
                 0.0 disables the module entirely.
    """

    def __init__(self, d_model: int, p: float = 0.0):
        super().__init__()
        self.p = p
        # Learned mask token — initialised small so it starts near zero
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        view_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens:       (B, K, N, d')  adapted tokens for each view
            view_present: (B, K)         bool, True if the view exists in this sample

        Returns:
            tokens:       (B, K, N, d')  with dropped views replaced by mask token
            view_present: (B, K)         bool, updated (dropped views → False)
        """
        if not self.training or self.p == 0.0:
            return tokens, view_present

        B, K, N, d = tokens.shape
        device = tokens.device

        # Only eligible to drop views that are actually present
        # Sample a Bernoulli drop mask; True = drop this view
        drop = torch.rand(B, K, device=device) < self.p   # (B, K)
        drop = drop & view_present                         # never drop absent views

        # Guarantee at least one present view survives per sample
        # (among the views that were present before this forward pass)
        would_lose_all = (view_present & ~drop).sum(dim=1) == 0  # (B,)
        if would_lose_all.any():
            # For affected samples, un-drop their first present view
            first_present = view_present.float().argmax(dim=1)    # (B,)
            drop[would_lose_all, first_present[would_lose_all]] = False

        # Replace dropped views with the learned mask token
        # mask_token: (d,) → broadcast to (B, K, N, d)
        mask = self.mask_token.view(1, 1, 1, d).expand(B, K, N, d)
        drop_expanded = drop[:, :, None, None].expand_as(tokens)
        tokens = torch.where(drop_expanded, mask, tokens)

        # Update view_present: dropped views are now absent
        view_present = view_present & ~drop

        return tokens, view_present
