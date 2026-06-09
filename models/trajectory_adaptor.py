"""Trajectory adaptor f_θ(actions, z_I0) → {C_t} for a frozen Wan TI2V-5B.

Output is a per-sampling-step positive context embedding of shape
[B, N, L, D] consumed by Wan's cross-attention. The base is always the
shared, σ-indexed mean trajectory μ (warm-started from data); the
per-(action, I_0) part is a learned residual whose *form* is selectable:

    C_t = μ_t + residual_t(actions, z_I0)

The architecture is config-driven along two orthogonal axes so variants
can be A/B'd (see docs/adaptor_design.md §3.12):

  --arch_fusion : how (actions, z_I0) are combined into a conditioning h
      "concat"     : pool image to 1 token + MLP-encode actions, concat,
                     MLP. No spatial positions, no token interaction.
                     (Original design — the control arm.)
      "cross_attn" : keep image tokens (with 2-D sin-cos positional
                     encoding) AND per-step action tokens; learnable query
                     token(s) cross-attend over both across several layers.
                     Recovers spatial granularity + lets the two modalities
                     interact.

  --arch_head : how h maps to the per-step residual
      "rank1"   : residual_t = β_t · γ · b           (single shared dir)
      "rankk"   : residual_t = Σ_k β_{t,k} · γ_k · b_k   (K shared dirs)
      "perstep" : residual_t = g_θ([h, step_emb_t])   (general; drops the
                     linear / shared-direction assumption entirely)

Defaults (concat + rank1) reproduce the original rank-1 adaptor. Every
variant initializes so that C_t ≈ μ_t at step 0 (warm-start preserved).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_2d_sincos_pos_emb(h: int, w: int, dim: int, device, dtype=torch.float32):
    """Resolution-agnostic 2-D sin-cos positional embedding, [h*w, dim].
    `dim` must be divisible by 4."""
    assert dim % 4 == 0, "hidden dim must be divisible by 4 for 2-D sincos"
    gy, gx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    q = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(q, device=device, dtype=dtype) / q))
    ey = gy.flatten()[:, None] * omega[None, :]   # [h*w, q]
    ex = gx.flatten()[:, None] * omega[None, :]
    return torch.cat([ey.sin(), ey.cos(), ex.sin(), ex.cos()], dim=1)  # [h*w, dim]


class _CrossAttnBlock(nn.Module):
    """Pre-norm (cross-attn → FFN) block: query tokens attend over memory."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, q, memory):
        q = q + self.attn(self.norm_q(q), self.norm_kv(memory),
                          self.norm_kv(memory))[0]
        q = q + self.ffn(q)
        return q


# ---------------------------------------------------------------------------
# adaptor
# ---------------------------------------------------------------------------


class TrajectoryAdaptor(nn.Module):

    def __init__(
        self,
        adaptor_init_path: str,
        hidden: int = 512,
        action_dim: int = 8,
        action_len: int = 32,
        vae_channels: int = 48,
        attn_heads: int = 4,
        b_head_init_scale: float = 1e-4,
        action_repr: str = "delta",
        fusion: str = "concat",          # "concat" | "cross_attn"
        head: str = "rank1",             # "rank1" | "rankk" | "perstep"
        rank_k: int = 4,
        n_xattn_layers: int = 2,
        step_emb_dim: int = 128,
    ):
        super().__init__()
        init = torch.load(adaptor_init_path, weights_only=False, map_location="cpu")
        mu = init["mu"].float()          # [N, L, D]
        beta = init["beta"].float()      # [N]
        if mu.dim() != 3:
            raise ValueError(f"expected mu [N, L, D]; got {tuple(mu.shape)}")

        self.N, self.L, self.D = mu.shape
        self.hidden = hidden
        self.action_len = action_len
        self.action_dim = action_dim
        self.vae_channels = vae_channels
        self.action_repr = action_repr
        self.fusion = fusion
        self.head = head
        self.K = 1 if head == "rank1" else rank_k

        # ---- shared base trajectory μ (always) ----
        self.mu = nn.Parameter(mu.clone())               # [N, L, D]

        # ---- conditioning backbone (fusion) ----
        if fusion == "concat":
            # Original design: pooled image + MLP action, concat, MLP.
            self.action_enc = nn.Sequential(
                nn.Linear(action_dim * action_len, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.image_proj = nn.Linear(vae_channels, hidden)
            self.image_pool = nn.MultiheadAttention(hidden, attn_heads, batch_first=True)
            self.image_query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
            self.fuse = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
        elif fusion == "cross_attn":
            # Token-level: per-step action tokens + spatial image tokens
            # (with 2-D positional encoding) fed to cross-attention queries.
            self.action_token_proj = nn.Linear(action_dim, hidden)
            self.action_pos = nn.Parameter(torch.randn(1, action_len, hidden) * 0.02)
            self.image_token_proj = nn.Linear(vae_channels, hidden)
            self.type_emb = nn.Parameter(torch.randn(2, hidden) * 0.02)  # action / image
            self.query_tokens = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
            self.xattn_blocks = nn.ModuleList(
                [_CrossAttnBlock(hidden, attn_heads) for _ in range(n_xattn_layers)])
            self.cond_norm = nn.LayerNorm(hidden)
        else:
            raise ValueError(f"unknown fusion {fusion!r}")

        # ---- head: h → per-step residual ----
        if head in ("rank1", "rankk"):
            # β gate per (step, component). Warm-start col 0 from data; rest 0.
            beta_mat = torch.zeros(self.N, self.K)
            beta_mat[:, 0] = beta
            self.beta = nn.Parameter(beta_mat)            # [N, K]
            self.gamma_head = nn.Linear(hidden, self.K)
            self.b_head = nn.Linear(hidden, self.K * self.L * self.D)
            # γ ≈ 0 at init so residual ≈ 0 and C ≈ μ, but nonzero weight so
            # gradients still reach b_head + the backbone from step 1.
            nn.init.zeros_(self.gamma_head.bias)
            nn.init.normal_(self.gamma_head.weight, std=1e-3)
            nn.init.normal_(self.b_head.weight, std=b_head_init_scale)
            nn.init.zeros_(self.b_head.bias)
        elif head == "perstep":
            # General per-step residual from [h, step_emb_t]. No β, no shared
            # direction. Last layer zero-init ⇒ residual ≡ 0 ⇒ C ≡ μ at init.
            self.step_emb = nn.Parameter(torch.randn(self.N, step_emb_dim) * 0.02)
            self.residual_mlp = nn.Sequential(
                nn.Linear(hidden + step_emb_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, self.L * self.D),
            )
            # Small (not zero) last-layer init: residual ≈ 0 so C ≈ μ, but
            # gradients still reach the backbone + step_emb from step 1 (a
            # zero last layer would block backprop through it — same dead-
            # gradient trap avoided for the rank heads' γ).
            nn.init.normal_(self.residual_mlp[-1].weight, std=b_head_init_scale)
            nn.init.zeros_(self.residual_mlp[-1].bias)
        else:
            raise ValueError(f"unknown head {head!r}")

    # ----------------------- inspection helpers -----------------------------

    @property
    def num_steps(self) -> int:
        return self.N

    def _mu_beta_params(self):
        ps = [self.mu]
        if hasattr(self, "beta"):
            ps.append(self.beta)
        return ps

    def freeze_mu_beta(self):
        for p in self._mu_beta_params():
            p.requires_grad_(False)

    def unfreeze_mu_beta(self):
        for p in self._mu_beta_params():
            p.requires_grad_(True)

    def parameter_groups(self, lr: float, lr_mu_beta: Optional[float] = None):
        if lr_mu_beta is None:
            lr_mu_beta = lr * 0.1
        slow = self._mu_beta_params()
        slow_ids = {id(p) for p in slow}
        fast = [p for p in self.parameters() if id(p) not in slow_ids]
        return [
            {"params": slow, "lr": lr_mu_beta, "name": "mu_beta"},
            {"params": fast, "lr": lr, "name": "heads"},
        ]

    # ----------------------- token builders ---------------------------------

    def _delta(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[1] != self.action_len or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions shape mismatch: got {tuple(actions.shape)}; "
                f"expected [B, {self.action_len}, {self.action_dim}]")
        if self.action_repr == "delta":
            return (actions - actions[:, 0:1]).float()
        return actions.float()

    # ----------------------- conditioning (fusion) ---------------------------

    def _condition(self, actions: torch.Tensor, z_I0: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        if self.fusion == "concat":
            h_a = self.action_enc(self._delta(actions).flatten(1))       # [B, hidden]
            tok = z_I0.squeeze(2).flatten(2).transpose(1, 2).float()      # [B, P, C]
            x = self.image_proj(tok)                                      # [B, P, hidden]
            q = self.image_query.expand(B, -1, -1)
            pooled, _ = self.image_pool(q, x, x)                          # [B, 1, hidden]
            h_i = pooled.squeeze(1)
            return self.fuse(torch.cat([h_a, h_i], dim=-1))              # [B, hidden]

        # cross_attn
        _, C, Fz, Hz, Wz = z_I0.shape
        at = self.action_token_proj(self._delta(actions))                 # [B, T, hidden]
        at = at + self.action_pos + self.type_emb[0]
        img_tok = z_I0.flatten(2).transpose(1, 2).float()                 # [B, Fz*Hz*Wz, C]
        it = self.image_token_proj(img_tok)                               # [B, P, hidden]
        pe = build_2d_sincos_pos_emb(Hz, Wz, self.hidden, z_I0.device)    # [Hz*Wz, hidden]
        if Fz > 1:
            pe = pe.repeat(Fz, 1)
        it = it + pe[None] + self.type_emb[1]
        memory = torch.cat([at, it], dim=1)                               # [B, T+P, hidden]
        q = self.query_tokens.expand(B, -1, -1)                           # [B, 1, hidden]
        for blk in self.xattn_blocks:
            q = blk(q, memory)
        return self.cond_norm(q.squeeze(1))                              # [B, hidden]

    # ----------------------- forward ----------------------------------------

    def forward(self, actions: torch.Tensor, z_I0: torch.Tensor) -> torch.Tensor:
        B = actions.shape[0]
        h = self._condition(actions, z_I0)                                # [B, hidden]

        if self.head in ("rank1", "rankk"):
            gamma = self.gamma_head(h)                                    # [B, K]
            b_raw = self.b_head(h).view(B, self.K, self.L * self.D)
            b = b_raw / b_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            b = b.view(B, self.K, self.L, self.D)                         # [B, K, L, D]
            gb = gamma[:, :, None, None] * b                              # [B, K, L, D]
            # resid[b,n,l,d] = Σ_k β[n,k] · gb[b,k,l,d]
            resid = torch.einsum("nk,bkld->bnld", self.beta, gb)          # [B, N, L, D]
        else:  # perstep
            hN = h[:, None, :].expand(B, self.N, self.hidden)             # [B, N, hidden]
            se = self.step_emb[None].expand(B, -1, -1)                    # [B, N, e]
            resid = self.residual_mlp(torch.cat([hN, se], dim=-1))        # [B, N, L*D]
            resid = resid.view(B, self.N, self.L, self.D)

        return self.mu.unsqueeze(0) + resid                              # [B, N, L, D]

    # ----------------------- diagnostics ------------------------------------

    @torch.no_grad()
    def initial_factors(self, actions, z_I0) -> dict:
        """Lightweight diagnostic for the M0 check. For rank heads returns
        (γ, b); for perstep returns the per-step residual norm."""
        h = self._condition(actions, z_I0)
        B = actions.shape[0]
        if self.head in ("rank1", "rankk"):
            gamma = self.gamma_head(h)
            b_raw = self.b_head(h).view(B, self.K, self.L * self.D)
            b = b_raw / b_raw.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            return {"gamma": gamma, "b": b.view(B, self.K, self.L, self.D)}
        hN = h[:, None, :].expand(B, self.N, self.hidden)
        se = self.step_emb[None].expand(B, -1, -1)
        resid = self.residual_mlp(torch.cat([hN, se], dim=-1))
        return {"residual_norm": resid.view(B, self.N, -1).norm(dim=-1)}
