# Action-Conditioned Wan Prototype

This document describes the prototype in
`models/action_conditioned_wan.py`. The goal is to test whether action control
works better when action is injected into the frozen Wan DiT representation
instead of predicting a full text-context trajectory outside the model.

## Motivation

The current trajectory adaptor computes:

```text
C_traj = adaptor(actions, z_I0)
v_t = Wan(z_t, t, C_t)
```

`C_t` is selected by step index, but it is computed before the actual diffusion
trajectory is known. Therefore the context cannot adapt to the sampled initial
noise basin or the current noisy latent.

Wan already handles three pieces of information we need:

- current noisy latent `z_t`, through patch embedding and self-attention;
- timestep/sigma `t`, through per-token time modulation in every block;
- initial frame `I0`, because TI2V pins the first-frame latent into `z_t` at
  every sampler step.

The new hypothesis is:

```text
action should be the only new external signal.
noise, timestep, and I0 should be reused from Wan's own hidden states.
```

This is strongest for block-level side adapters and pre-context feature
predictors, because the trainable module sees Wan hidden video tokens instead
of only external `(action, I0)` features.

## Modes

`ActionConditionedWanModel` wraps a frozen `WanModel` and adds one `actions`
argument to the normal forward call.

### `append_context`

Encode actions into T5-space tokens and append them to the text context:

```text
context' = concat(text_context, action_context)
v = frozen_Wan(z_t, t, context')
```

This is the smallest possible change. It reuses Wan's existing cross-attention
stack, but it asks frozen text projections to interpret non-text action tokens.
It is useful as a cheap baseline, not as the main expected solution.

If `text_context + action_tokens` exceeds `WanModel.text_len`, the default
policy truncates text tokens to leave room for actions. This is configurable
with `context_overflow`.

### `replace_context`

Encode actions into T5-space tokens and use them instead of text:

```text
context' = action_context
v = frozen_Wan(z_t, t, context')
```

This is the pure action adapter ablation. It answers whether action tokens alone
can steer the frozen model when text is removed. It is intentionally severe: it
will likely lose semantic prompt support, but it isolates action-only control.

### `pre_context`

Extract context-free Wan features, predict context tokens from those features
and action, then run the frozen Wan model with the predicted context:

```text
h_pre = WanPatchTimeSelfAttn(z_t, t, I0)     # stops before first cross-attn
C_pred = ContextHead(h_pre, action_tokens)
v = frozen_Wan(z_t, t, C_pred)
```

The feature pass intentionally does not use text, action context, null context,
or any default context. It mirrors Wan's patch embedding, timestep embedding,
and the first block's timestep-modulated self-attention, then stops before the
first cross-attention site. This avoids circularity:

```text
C_pred depends on h_pre and action
h_pre does not depend on C_pred
```

This mode costs more than `append_context`/`replace_context`, because each
training/eval step does a feature pass and then a normal Wan velocity pass. It
is still cheaper and cleaner than a two-pass design where the first pass runs
the entire Wan model with a default context.

### `side_adapter`

Run the normal frozen Wan block, then add an action residual after selected
blocks:

```text
action_tokens = ActionTokenEncoder(actions)

for block l:
    x_l = frozen_WanBlock_l(x_l, t, text_context)
    if l in adapter_layers:
        delta_l = ActionSideAdapter_l(x_l, action_tokens)
        x_l = x_l + delta_l
```

`ActionSideAdapter_l` is bottleneck cross-attention from current video tokens to
action tokens:

```text
x_l          -> LayerNorm -> Linear(D_wan, H_side)
action_tokens -> LayerNorm -> Linear(D_action, H_side)
delta_hidden = CrossAttention_H_side(x_hidden, action_hidden)
delta_l      = zero_init_Linear(H_side, D_wan)(delta_hidden + FFN_H_side(delta_hidden))
```

The default `H_side` is `512` with 8 attention heads, not the full Wan hidden
width/head count. This keeps the side branch an adapter rather than another
full-width transformer stack. The last projection is zero-initialized, so at
initialization `delta_l == 0` and the wrapped model is exactly the frozen Wan
model.

Because the query is `x_l`, this residual is a function of:

```text
action, current z_t, timestep, pinned I0, text context, and block depth
```

This is the main "bigger redesign" candidate.

## Shape Contract

The wrapper forward signature is:

```python
out = wrapped_wan(
    x=[latent],              # Wan list input, each [C,F,H,W]
    t=timestep_tensor,       # same as WanModel.forward
    context=[context],       # same T5-space context list as WanModel.forward
    seq_len=seq_len,
    actions=actions,         # [B, action_len, action_dim]
    y=None,
)
```

The output is the same list of `[C,F,H,W]` tensors returned by `WanModel`.

For `pre_context`, the `context` argument is only present for API
compatibility; it is ignored by the feature pass and replaced by the predicted
context during the velocity pass.

## Trainable Parameters

The wrapped base model is frozen by default. Trainable parameters are:

- `ActionTokenEncoder`;
- for `pre_context`, `PreContextFeatureContextHead`;
- for `side_adapter`, one bottleneck `ActionSideAdapter` per selected block.

Use `adapter_parameters()` for optimizers and `trainable_state_dict()` for
checkpoints that exclude the frozen Wan backbone.

## Why This Should Be More Noise-Aware

The previous `perstep` context adaptor has learned `step_emb_t`, but its input
is still only `(actions, z_I0)`. It cannot react to a different sampled `z_t`.

The `pre_context` predictor reads first-block pre-cross-attention features, and
the side adapter reads later hidden states `x_l`. These features change with:

- initial noise basin;
- current diffusion step;
- Euler trajectory history;
- first-frame pinning;
- text context.

Therefore the same action can produce different predicted contexts or residuals
under different noise basins without requiring the action encoder itself to
predict low-level noise geometry.

## Suggested Ablations

Run the same train/eval protocol with matched guidance scale:

1. current context adaptor baseline;
2. `append_context`;
3. `replace_context`;
4. `pre_context`;
5. `side_adapter` on late blocks only, for example layers 20-29;
6. `side_adapter` on every other block;
7. `side_adapter` on all blocks.

Late-block adapters are the safest first trial: they modify semantics/motion
after the lower-level latent representation is established. Full-depth adapters
are more expressive but have higher memory and optimization risk.

## Limitations

- `append_context` and `replace_context` are not identity-preserving by design,
  because adding/replacing context changes frozen cross-attention inputs.
- `pre_context` is also not identity-preserving; it replaces the context with a
  predicted context generated from context-free Wan features.
- `side_adapter` is identity-preserving at initialization, but the first update
  mainly trains the zero output projections; deeper action encoder gradients
  become meaningful after those projections move away from zero.
- The prototype does not yet wire into `train_adaptor.py`; it provides a tested
  module boundary for the next training-script integration.
