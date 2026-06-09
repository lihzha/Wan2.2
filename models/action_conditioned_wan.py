"""Action-conditioned wrappers for a frozen Wan DiT.

The existing trajectory adaptor predicts text-context embeddings outside the
Wan model. This module prototypes a deeper alternative: keep Wan frozen and
inject action conditioning into places where Wan already represents the noisy
latent, timestep, prompt context, and pinned first-frame conditioning.

Four modes are supported:

``append_context``
    Encode actions into extra T5-space context tokens and append them to the
    normal prompt context before calling the frozen WanModel.

``replace_context``
    Encode actions into T5-space context tokens and use those tokens instead of
    text. This is the pure action adapter requested for ablations.

``pre_context``
    Run a context-free feature pass through Wan's patch/time embedding and the
    first block's self-attention, predict T5-space context tokens from those
    features plus action tokens, then run the frozen WanModel with the predicted
    context. This avoids circular dependence because the feature pass stops
    before cross-attention ever consumes context.

``side_adapter``
    Reimplement WanModel.forward with the same frozen blocks, and after selected
    transformer blocks add a zero-initialized trainable residual computed by
    cross-attending current video hidden states to action tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
import torch.nn as nn

from wan.modules.model import WanModel, sinusoidal_embedding_1d


ActionMode = Literal[
    "append_context", "replace_context", "pre_context", "side_adapter"
]
OverflowPolicy = Literal["error", "truncate_text"]


@dataclass(frozen=True)
class ActionConditionedWanConfig:
    mode: ActionMode = "side_adapter"
    action_dim: int = 8
    action_len: int = 32
    action_repr: str = "delta"
    action_tokens: int = 8
    action_hidden: int = 512
    action_heads: int = 4
    pre_context_tokens: int = 8
    pre_context_heads: int | None = None
    side_adapter_layers: tuple[int, ...] | None = None
    side_adapter_hidden: int = 512
    side_adapter_heads: int | None = 8
    context_overflow: OverflowPolicy = "truncate_text"


class ActionTokenEncoder(nn.Module):
    """Encode a fixed-length action trajectory into a small token set."""

    def __init__(
        self,
        action_dim: int,
        action_len: int,
        token_dim: int,
        num_tokens: int,
        hidden: int = 512,
        heads: int = 4,
        action_repr: str = "delta",
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.action_dim = int(action_dim)
        self.action_len = int(action_len)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.action_repr = action_repr

        self.action_proj = nn.Linear(action_dim, hidden)
        self.action_pos = nn.Parameter(torch.randn(1, action_len, hidden) * 0.02)
        self.query_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden) * 0.02)
        self.pool = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
        )

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.dim() != 3:
            raise ValueError(f"actions must be [B,T,A], got {tuple(actions.shape)}")
        if actions.shape[1] != self.action_len or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"actions shape mismatch: got {tuple(actions.shape)}; expected "
                f"[B, {self.action_len}, {self.action_dim}]")
        if self.action_repr == "delta":
            return (actions - actions[:, 0:1]).float()
        if self.action_repr == "raw":
            return actions.float()
        raise ValueError(f"unknown action_repr {self.action_repr!r}")

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        actions = self._prepare_actions(actions)
        memory = self.action_proj(actions) + self.action_pos.to(actions.device)
        query = self.query_tokens.expand(actions.shape[0], -1, -1)
        tokens, _ = self.pool(query, memory, memory)
        return self.out(tokens)


class ActionSideAdapter(nn.Module):
    """Zero-initialized bottleneck residual reading Wan hidden states/actions."""

    def __init__(self, dim: int, action_dim: int, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        self.norm_x = nn.LayerNorm(dim)
        self.norm_action = nn.LayerNorm(action_dim)
        self.x_proj = nn.Linear(dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, dim)
        self.gate = nn.Parameter(torch.ones(()))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        action_tokens = self.action_proj(self.norm_action(action_tokens))
        q = self.x_proj(self.norm_x(x))
        delta, _ = self.attn(q, action_tokens, action_tokens)
        delta = delta + self.ffn(delta)
        return self.gate * self.out(delta)


class PreContextFeatureContextHead(nn.Module):
    """Predict T5-space context from context-free Wan features and actions."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        text_dim: int,
        context_tokens: int,
        heads: int,
    ):
        super().__init__()
        if context_tokens <= 0:
            raise ValueError("context_tokens must be positive")
        if feature_dim % heads != 0:
            raise ValueError("feature_dim must be divisible by heads")
        self.context_tokens = int(context_tokens)
        self.norm_features = nn.LayerNorm(feature_dim)
        self.norm_actions = nn.LayerNorm(action_dim)
        self.action_proj = (
            nn.Identity() if action_dim == feature_dim
            else nn.Linear(action_dim, feature_dim)
        )
        self.query_tokens = nn.Parameter(
            torch.randn(1, context_tokens, feature_dim) * 0.02)
        self.attn = nn.MultiheadAttention(feature_dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 4 * feature_dim),
            nn.SiLU(),
            nn.Linear(4 * feature_dim, feature_dim),
        )
        self.to_context = nn.Linear(feature_dim, text_dim)

    def forward(
        self,
        features: torch.Tensor,
        action_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        bsz, feature_len, _ = features.shape
        features = self.norm_features(features)
        action_tokens = self.action_proj(self.norm_actions(action_tokens))
        memory = torch.cat([features, action_tokens], dim=1)

        pos = torch.arange(feature_len, device=features.device)[None, :]
        feature_mask = pos >= seq_lens.to(features.device)[:, None]
        action_mask = torch.zeros(
            bsz, action_tokens.shape[1], dtype=torch.bool, device=features.device)
        key_padding_mask = torch.cat([feature_mask, action_mask], dim=1)

        query = self.query_tokens.expand(bsz, -1, -1)
        h, _ = self.attn(
            query, memory, memory, key_padding_mask=key_padding_mask)
        h = h + self.ffn(h)
        return self.to_context(h)


class ActionConditionedWanModel(nn.Module):
    """Trainable action wrapper around a frozen :class:`WanModel`.

    The public forward is WanModel.forward plus an ``actions`` tensor.
    ``append_context`` and ``replace_context`` delegate to the frozen base model
    after changing the context list. ``pre_context`` predicts a context from a
    context-free Wan feature pass, then delegates to the frozen base model.
    ``side_adapter`` uses the same Wan forward calculation but adds trainable
    action residuals after selected blocks.
    """

    def __init__(
        self,
        base: WanModel,
        config: ActionConditionedWanConfig | None = None,
        *,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base
        self.config = config or ActionConditionedWanConfig()
        self.mode = self.config.mode
        if self.mode not in (
            "append_context", "replace_context", "pre_context", "side_adapter"
        ):
            raise ValueError(f"unknown action conditioning mode {self.mode!r}")
        if freeze_base:
            self.freeze_base()

        action_token_dim = (
            base.dim if self.mode in ("pre_context", "side_adapter")
            else base.text_dim
        )
        self.action_encoder = ActionTokenEncoder(
            action_dim=self.config.action_dim,
            action_len=self.config.action_len,
            token_dim=action_token_dim,
            num_tokens=self.config.action_tokens,
            hidden=self.config.action_hidden,
            heads=self.config.action_heads,
            action_repr=self.config.action_repr,
        )

        self.side_adapter_layers = self._normalize_side_layers(
            self.config.side_adapter_layers)
        side_heads = self.config.side_adapter_heads or base.num_heads
        pre_context_heads = self.config.pre_context_heads or base.num_heads
        self.pre_context_head = None
        if self.mode == "pre_context":
            if self.config.pre_context_tokens > base.text_len:
                raise ValueError(
                    f"pre_context_tokens {self.config.pre_context_tokens} "
                    f"exceeds base.text_len={base.text_len}")
            self.pre_context_head = PreContextFeatureContextHead(
                feature_dim=base.dim,
                action_dim=action_token_dim,
                text_dim=base.text_dim,
                context_tokens=self.config.pre_context_tokens,
                heads=pre_context_heads,
            )
        self.side_adapters = nn.ModuleDict()
        if self.mode == "side_adapter":
            for idx in self.side_adapter_layers:
                self.side_adapters[str(idx)] = ActionSideAdapter(
                    dim=base.dim,
                    action_dim=action_token_dim,
                    hidden_dim=self.config.side_adapter_hidden,
                    heads=side_heads,
                )

    def freeze_base(self) -> None:
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, param in self.named_parameters():
            if not name.startswith("base."):
                yield param

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
            if not k.startswith("base.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor], strict=True):
        own = self.state_dict()
        missing = [k for k in own if not k.startswith("base.") and k not in state]
        unexpected = [k for k in state if k not in own or k.startswith("base.")]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"adapter state mismatch: missing={missing} unexpected={unexpected}")
        loadable = {k: v for k, v in state.items() if k in own and not k.startswith("base.")}
        own.update(loadable)
        self.load_state_dict(own, strict=False)
        return missing, unexpected

    def _normalize_side_layers(
        self, layers: Sequence[int] | None
    ) -> tuple[int, ...]:
        if layers is None:
            return tuple(range(self.base.num_layers))
        out = tuple(int(x) for x in layers)
        bad = [x for x in out if x < 0 or x >= self.base.num_layers]
        if bad:
            raise ValueError(
                f"side_adapter_layers contains invalid indices {bad}; "
                f"num_layers={self.base.num_layers}")
        return out

    def _action_context(self, actions: torch.Tensor) -> list[torch.Tensor]:
        tokens = self.action_encoder(actions)
        if tokens.shape[1] > self.base.text_len:
            raise ValueError(
                f"action token count {tokens.shape[1]} exceeds "
                f"base.text_len={self.base.text_len}")
        return [tokens[i] for i in range(tokens.shape[0])]

    def _combine_context(
        self, context: list[torch.Tensor], actions: torch.Tensor
    ) -> list[torch.Tensor]:
        action_ctx = self._action_context(actions)
        if self.mode == "replace_context":
            return action_ctx

        combined = []
        for text, act in zip(context, action_ctx):
            if text.shape[-1] != self.base.text_dim:
                raise ValueError(
                    f"context last dim {text.shape[-1]} != base.text_dim "
                    f"{self.base.text_dim}")
            if act.shape[-1] != self.base.text_dim:
                raise ValueError(
                    f"action token dim {act.shape[-1]} != base.text_dim "
                    f"{self.base.text_dim}")
            if text.shape[0] + act.shape[0] > self.base.text_len:
                if self.config.context_overflow == "error":
                    raise ValueError(
                        f"context length {text.shape[0]} + action tokens "
                        f"{act.shape[0]} exceeds base.text_len={self.base.text_len}")
                keep = self.base.text_len - act.shape[0]
                if keep <= 0:
                    raise ValueError(
                        f"action tokens {act.shape[0]} exceed "
                        f"base.text_len={self.base.text_len}")
                text = text[:keep]
            combined.append(torch.cat([text, act.to(text.dtype)], dim=0))
        return combined

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        actions: torch.Tensor,
        y: list[torch.Tensor] | None = None,
        return_adapter_residuals: bool = False,
    ):
        if actions.shape[0] != len(x):
            raise ValueError(
                f"actions batch {actions.shape[0]} must match len(x) {len(x)}")
        if len(context) != len(x):
            raise ValueError(
                f"context batch {len(context)} must match len(x) {len(x)}")
        if y is not None and len(y) != len(x):
            raise ValueError(f"y batch {len(y)} must match len(x) {len(x)}")
        if self.mode in ("append_context", "replace_context"):
            out = self.base(
                x=x,
                t=t,
                context=self._combine_context(context, actions),
                seq_len=seq_len,
                y=y,
            )
            if return_adapter_residuals:
                return out, []
            return out
        if self.mode == "pre_context":
            return self._forward_pre_context(
                x=x,
                t=t,
                seq_len=seq_len,
                actions=actions,
                y=y,
                return_adapter_residuals=return_adapter_residuals,
            )
        return self._forward_side_adapter(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            actions=actions,
            y=y,
            return_adapter_residuals=return_adapter_residuals,
        )

    def _patchify_and_time_embed(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        seq_len: int,
        y: list[torch.Tensor] | None,
    ):
        base = self.base
        if base.model_type == "i2v":
            assert y is not None
        device = base.patch_embedding.weight.device
        if base.freqs.device != device:
            base.freqs = base.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [base.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=u.device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor(
            [u.size(1) for u in x], dtype=torch.long, device=x[0].device)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast("cuda", dtype=torch.float32):
            bt = t.size(0)
            t_flat = t.flatten()
            e = base.time_embedding(
                sinusoidal_embedding_1d(
                    base.freq_dim, t_flat).unflatten(0, (bt, seq_len)).float())
            e0 = base.time_projection(e).unflatten(2, (6, base.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        return x, e, e0, seq_lens, grid_sizes

    def _extract_pre_context_features(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        seq_len: int,
        y: list[torch.Tensor] | None,
    ):
        """Return hidden tokens after first-block self-attention, before context.

        This mirrors the first part of :meth:`WanAttentionBlock.forward` and
        stops before the first cross-attention call, so no text/action context
        has entered the feature pass.
        """
        base = self.base
        with torch.no_grad():
            x, _, e0, seq_lens, grid_sizes = self._patchify_and_time_embed(
                x, t, seq_len, y)
            block = base.blocks[0]
            assert e0.dtype == torch.float32
            with torch.amp.autocast("cuda", dtype=torch.float32):
                e_chunks = (block.modulation.unsqueeze(0) + e0).chunk(6, dim=2)
            y_self = block.self_attn(
                block.norm1(x).float() * (1 + e_chunks[1].squeeze(2))
                + e_chunks[0].squeeze(2),
                seq_lens,
                grid_sizes,
                base.freqs,
            )
            with torch.amp.autocast("cuda", dtype=torch.float32):
                features = x + y_self * e_chunks[2].squeeze(2)
        return features.detach(), seq_lens.detach()

    def _forward_pre_context(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        seq_len: int,
        actions: torch.Tensor,
        y: list[torch.Tensor] | None,
        return_adapter_residuals: bool,
    ):
        if self.pre_context_head is None:
            raise RuntimeError("pre_context_head is not initialized")
        features, seq_lens = self._extract_pre_context_features(
            x=x, t=t, seq_len=seq_len, y=y)
        action_tokens = self.action_encoder(actions).to(features.dtype)
        predicted_context = self.pre_context_head(
            features, action_tokens, seq_lens)
        context = [predicted_context[i] for i in range(predicted_context.shape[0])]
        out = self.base(x=x, t=t, context=context, seq_len=seq_len, y=y)
        if return_adapter_residuals:
            return out, [{
                "mode": "pre_context",
                "predicted_context": predicted_context.detach().float(),
                "context_rms": torch.sqrt(
                    torch.mean(predicted_context.float().square())),
                "context_max_abs": predicted_context.float().abs().max(),
            }]
        return out

    def _forward_side_adapter(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        context: list[torch.Tensor],
        seq_len: int,
        actions: torch.Tensor,
        y: list[torch.Tensor] | None,
        return_adapter_residuals: bool,
    ):
        base = self.base
        x, e, e0, seq_lens, grid_sizes = self._patchify_and_time_embed(
            x, t, seq_len, y)

        context_lens = None
        for u in context:
            if u.size(0) > base.text_len:
                raise ValueError(
                    f"context length {u.size(0)} exceeds base.text_len={base.text_len}")
        context = base.text_embedding(
            torch.stack([
                torch.cat([
                    u,
                    u.new_zeros(base.text_len - u.size(0), u.size(1)),
                ])
                for u in context
            ]))

        action_tokens = self.action_encoder(actions).to(x.dtype)
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=base.freqs,
            context=context,
            context_lens=context_lens,
        )
        residuals = []
        for idx, block in enumerate(base.blocks):
            x = block(x, **kwargs)
            key = str(idx)
            if key in self.side_adapters:
                adapter = self.side_adapters[key]
                delta = adapter(x, action_tokens)
                x = x + delta
                if return_adapter_residuals:
                    residuals.append({
                        "layer": idx,
                        "delta": delta.detach().float(),
                        "rms": torch.sqrt(torch.mean(delta.float().square())),
                        "max_abs": delta.float().abs().max(),
                    })

        x = base.head(x, e)
        x = base.unpatchify(x, grid_sizes)
        out = [u.float() for u in x]
        if return_adapter_residuals:
            return out, residuals
        return out
