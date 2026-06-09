import copy
import os
import sys

import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models.action_conditioned_wan import (
    ActionConditionedWanConfig,
    ActionConditionedWanModel,
)
from wan.modules import attention as _wan_attn
import wan.modules.model as _wan_model
from wan.modules.model import WanModel


def _install_sdpa_fallback():
    def _flash_attention_sdpa_fallback(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.0,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        if q_scale is not None:
            q = q * q_scale
        out_dtype = q.dtype
        compute_dtype = q.dtype if q.device.type == "cpu" else dtype
        q_ = q.transpose(1, 2).to(compute_dtype)
        k_ = k.transpose(1, 2).to(compute_dtype)
        v_ = v.transpose(1, 2).to(compute_dtype)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, dropout_p=dropout_p, is_causal=causal)
        return out.transpose(1, 2).contiguous().to(out_dtype)

    _wan_attn.flash_attention = _flash_attention_sdpa_fallback
    _wan_model.flash_attention = _flash_attention_sdpa_fallback


_install_sdpa_fallback()


def _tiny_wan(seed=0):
    torch.manual_seed(seed)
    model = WanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=12,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=20,
        out_dim=4,
        num_heads=4,
        num_layers=3,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
    ).eval()
    nn.init.xavier_uniform_(model.head.head.weight)
    nn.init.zeros_(model.head.head.bias)
    return model


def _inputs(batch=1, seed=1):
    gen = torch.Generator().manual_seed(seed)
    x = [
        torch.randn(4, 2, 4, 4, generator=gen)
        for _ in range(batch)
    ]
    context = [
        torch.randn(5, 20, generator=gen)
        for _ in range(batch)
    ]
    actions = torch.randn(batch, 4, 3, generator=gen)
    t = torch.full((batch,), 500.0)
    return x, t, context, actions


def _side_cfg(**kwargs):
    params = dict(
        mode="side_adapter",
        action_dim=3,
        action_len=4,
        action_tokens=3,
        action_hidden=32,
        action_heads=4,
        side_adapter_layers=(0, 2),
        side_adapter_hidden=16,
        side_adapter_heads=4,
    )
    params.update(kwargs)
    return ActionConditionedWanConfig(**params)


def _context_cfg(mode):
    return ActionConditionedWanConfig(
        mode=mode,
        action_dim=3,
        action_len=4,
        action_tokens=3,
        action_hidden=32,
        action_heads=4,
        pre_context_tokens=4,
        pre_context_heads=4,
    )


def _enable_side_residuals(model):
    for adapter in model.side_adapters.values():
        nn.init.normal_(adapter.out.weight, std=0.2)
        nn.init.zeros_(adapter.out.bias)


def _max_delta_diff(left, right):
    return max(
        (a["delta"] - b["delta"]).abs().max().item()
        for a, b in zip(left, right)
    )


def test_side_adapter_zero_init_matches_frozen_wan():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _side_cfg())
    x, t, context, actions = _inputs()

    with torch.no_grad():
        expected = base(x=copy.deepcopy(x), t=t, context=context, seq_len=8)[0]
        actual = wrapped(
            x=copy.deepcopy(x),
            t=t,
            context=context,
            seq_len=8,
            actions=actions,
        )[0]

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_side_adapter_uses_bottleneck_width():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _side_cfg(side_adapter_hidden=16))

    adapter = wrapped.side_adapters["0"]
    assert adapter.x_proj.out_features == 16
    assert adapter.action_proj.out_features == 16
    assert adapter.attn.embed_dim == 16
    assert adapter.out.in_features == 16
    assert adapter.out.out_features == base.dim


def test_side_adapter_freezes_base_but_receives_gradients():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _side_cfg())
    x, t, context, actions = _inputs()

    out = wrapped(x=x, t=t, context=context, seq_len=8, actions=actions)[0]
    loss = out.square().mean()
    loss.backward()

    assert all(p.requires_grad is False for p in wrapped.base.parameters())
    assert all(p.grad is None for p in wrapped.base.parameters())
    adapter_grad_norms = [
        p.grad.abs().sum().item()
        for p in wrapped.adapter_parameters()
        if p.grad is not None
    ]
    assert adapter_grad_norms
    assert max(adapter_grad_norms) > 0.0


def test_side_adapter_action_residual_depends_on_action_and_noisy_hidden_state():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _side_cfg())
    _enable_side_residuals(wrapped)
    x, t, context, actions = _inputs()
    actions_2 = actions.clone()
    actions_2[:, 1:] = actions_2[:, 1:] + 1.0
    x_2 = [u.clone() for u in x]
    x_2[0] = x_2[0].flip(-1) * 3.0

    _, diag_a = wrapped(
        x=copy.deepcopy(x),
        t=t,
        context=context,
        seq_len=8,
        actions=actions,
        return_adapter_residuals=True,
    )
    _, diag_action = wrapped(
        x=copy.deepcopy(x),
        t=t,
        context=context,
        seq_len=8,
        actions=actions_2,
        return_adapter_residuals=True,
    )
    _, diag_noise = wrapped(
        x=x_2,
        t=t,
        context=context,
        seq_len=8,
        actions=actions,
        return_adapter_residuals=True,
    )

    assert len(diag_a) == 2
    assert _max_delta_diff(diag_a, diag_action) > 1e-6
    assert _max_delta_diff(diag_a, diag_noise) > 1e-6


def test_context_token_modes_run_and_return_wan_shapes():
    for mode in ("append_context", "replace_context"):
        base = _tiny_wan()
        wrapped = ActionConditionedWanModel(base, _context_cfg(mode))
        x, t, context, actions = _inputs()

        out = wrapped(x=x, t=t, context=context, seq_len=8, actions=actions)

        assert isinstance(out, list)
        assert len(out) == 1
        assert tuple(out[0].shape) == tuple(x[0].shape)


def test_replace_context_ignores_text_context():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _context_cfg("replace_context"))
    x, t, context, actions = _inputs()
    other_context = [context[0] * 0.0 + 100.0]

    with torch.no_grad():
        out_a = wrapped(
            x=copy.deepcopy(x), t=t, context=context, seq_len=8, actions=actions)[0]
        out_b = wrapped(
            x=copy.deepcopy(x), t=t, context=other_context, seq_len=8,
            actions=actions)[0]

    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_pre_context_mode_runs_and_returns_diagnostics():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _context_cfg("pre_context"))
    x, t, context, actions = _inputs()

    out, diag = wrapped(
        x=x,
        t=t,
        context=context,
        seq_len=8,
        actions=actions,
        return_adapter_residuals=True,
    )

    assert isinstance(out, list)
    assert len(out) == 1
    assert tuple(out[0].shape) == tuple(x[0].shape)
    assert len(diag) == 1
    assert diag[0]["mode"] == "pre_context"
    assert tuple(diag[0]["predicted_context"].shape) == (1, 4, 20)


def test_pre_context_ignores_input_text_context():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _context_cfg("pre_context"))
    x, t, context, actions = _inputs()
    other_context = [context[0] * 0.0 + 100.0]

    with torch.no_grad():
        out_a = wrapped(
            x=copy.deepcopy(x),
            t=t,
            context=context,
            seq_len=8,
            actions=actions,
        )[0]
        out_b = wrapped(
            x=copy.deepcopy(x),
            t=t,
            context=other_context,
            seq_len=8,
            actions=actions,
        )[0]

    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_pre_context_context_depends_on_action_and_noisy_state():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _context_cfg("pre_context"))
    x, t, context, actions = _inputs()
    actions_2 = actions.clone()
    actions_2[:, 1:] = actions_2[:, 1:] + 1.0
    x_2, _, _, _ = _inputs(seed=99)

    _, diag_a = wrapped(
        x=copy.deepcopy(x),
        t=t,
        context=context,
        seq_len=8,
        actions=actions,
        return_adapter_residuals=True,
    )
    _, diag_action = wrapped(
        x=copy.deepcopy(x),
        t=t,
        context=context,
        seq_len=8,
        actions=actions_2,
        return_adapter_residuals=True,
    )
    _, diag_noise = wrapped(
        x=x_2,
        t=t,
        context=context,
        seq_len=8,
        actions=actions,
        return_adapter_residuals=True,
    )

    pred = diag_a[0]["predicted_context"]
    pred_action = diag_action[0]["predicted_context"]
    pred_noise = diag_noise[0]["predicted_context"]
    assert (pred - pred_action).abs().max() > 1e-6
    assert (pred - pred_noise).abs().max() > 1e-6


def test_pre_context_freezes_base_but_receives_gradients():
    base = _tiny_wan()
    wrapped = ActionConditionedWanModel(base, _context_cfg("pre_context"))
    x, t, context, actions = _inputs()

    out = wrapped(x=x, t=t, context=context, seq_len=8, actions=actions)[0]
    loss = out.square().mean()
    loss.backward()

    assert all(p.requires_grad is False for p in wrapped.base.parameters())
    assert all(p.grad is None for p in wrapped.base.parameters())
    adapter_grad_norms = [
        p.grad.abs().sum().item()
        for p in wrapped.adapter_parameters()
        if p.grad is not None
    ]
    assert adapter_grad_norms
    assert max(adapter_grad_norms) > 0.0


def test_trainable_state_dict_round_trip_excludes_base():
    base_a = _tiny_wan(seed=3)
    base_b = copy.deepcopy(base_a)
    wrapped_a = ActionConditionedWanModel(base_a, _side_cfg())
    wrapped_b = ActionConditionedWanModel(base_b, _side_cfg())
    _enable_side_residuals(wrapped_a)
    state = wrapped_a.trainable_state_dict()

    assert state
    assert all(not key.startswith("base.") for key in state)
    missing, unexpected = wrapped_b.load_trainable_state_dict(state)
    assert missing == []
    assert unexpected == []

    x, t, context, actions = _inputs()
    with torch.no_grad():
        out_a = wrapped_a(
            x=copy.deepcopy(x), t=t, context=context, seq_len=8, actions=actions)[0]
        out_b = wrapped_b(
            x=copy.deepcopy(x), t=t, context=context, seq_len=8, actions=actions)[0]

    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[test] {name}")
            fn()
    print("[test] PASS")
