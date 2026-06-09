"""Initial latent predictor for deterministic video reconstruction.

The trajectory adaptor predicts the per-step text contexts. This module predicts
the sampler's initial latent so evaluation can start from a learned
action/image-specific point instead of a generic seed-noise sample.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


class ZInitPredictor(nn.Module):
    """Predict a full Wan latent delta from (actions, first-frame latent).

    The output is initialized to zero, so when a base latent is supplied the
    initial behavior is exactly the previous seed-noise path. That makes it
    safe to add to existing rollout overfit runs.
    """

    def __init__(
        self,
        latent_shape: Sequence[int],
        hidden: int = 512,
        action_dim: int = 7,
        action_len: int = 32,
        action_repr: str = "delta",
        attn_heads: int = 4,
    ):
        super().__init__()
        self.latent_shape = tuple(int(x) for x in latent_shape)
        if len(self.latent_shape) != 4:
            raise ValueError(f"expected latent_shape [C,F,H,W], got {self.latent_shape}")
        c, _, _, _ = self.latent_shape
        self.hidden = int(hidden)
        self.action_dim = int(action_dim)
        self.action_len = int(action_len)
        self.action_repr = action_repr

        self.action_enc = nn.Sequential(
            nn.Linear(self.action_dim * self.action_len, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.image_proj = nn.Linear(c, self.hidden)
        self.image_pool = nn.MultiheadAttention(
            self.hidden, attn_heads, batch_first=True)
        self.image_query = nn.Parameter(torch.randn(1, 1, self.hidden) * 0.02)
        self.fuse = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
        )
        self.out = nn.Linear(self.hidden, math.prod(self.latent_shape))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _delta(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[1] != self.action_len or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions shape mismatch: got {tuple(actions.shape)}; "
                f"expected [B, {self.action_len}, {self.action_dim}]")
        if self.action_repr == "delta":
            return (actions - actions[:, 0:1]).float()
        if self.action_repr == "raw":
            return actions.float()
        raise ValueError(f"unknown action_repr {self.action_repr!r}")

    def forward(
        self,
        actions: torch.Tensor,
        z_I0: torch.Tensor,
        base_z_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = actions.shape[0]
        h_a = self.action_enc(self._delta(actions).flatten(1))

        tok = z_I0.squeeze(2).flatten(2).transpose(1, 2).float()
        h_i = self.image_proj(tok)
        q = self.image_query.expand(bsz, -1, -1)
        pooled, _ = self.image_pool(q, h_i, h_i)
        h = self.fuse(torch.cat([h_a, pooled.squeeze(1)], dim=-1))

        delta = self.out(h).view(bsz, *self.latent_shape)
        if base_z_init is None:
            return delta
        if base_z_init.dim() == 4:
            base_z_init = base_z_init.unsqueeze(0).expand_as(delta)
        if tuple(base_z_init.shape) != tuple(delta.shape):
            raise ValueError(
                f"base_z_init shape {tuple(base_z_init.shape)} does not match "
                f"predicted shape {tuple(delta.shape)}")
        return base_z_init.float() + delta
