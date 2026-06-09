# Action-Conditioned Video Generation via a Frozen Wan TI2V-5B Adaptor

## 0. TL;DR

We want to generate videos from `(action_sequence, I_0)` using a **frozen**
Wan 2.2 TI2V-5B model. To do this we train a small **adaptor** that maps
`(action, I_0)` to a per-diffusion-step positive context embedding `{C_t}`
which the frozen DiT consumes via cross-attention.

The adaptor uses a **rank-1 architecture**

    C_t = μ_t + β_t · γ · b

where `μ_t` and `β_t` are σ-indexed lookup tables shared across all videos
(learned, initialized from data), and `(γ, b)` are produced by heads
conditioned on `(action, I_0)`. Training is **end-to-end with Wan's
flow-matching denoising loss** — no factor-regression step.

The rank-1 architecture is motivated by an empirical factorization
finding (Sec. 1.3) showing that across 50 reference videos, all the
σ-dependent structure of `{C_t^v}` is shared and the only video-specific
quantity is a scalar γ and a direction in `R^4096`.

---

## 1. What we have today

### 1.1 Goal restated

For each `(I_0, action a)` we want a generated video that depicts the
result of executing `a` starting from `I_0`. The Wan TI2V-5B model
(8×H100-class, 5B parameters) is pretrained for image-to-video generation
with text-prompt conditioning. We do **not** want to fine-tune Wan; we
want a small trainable module on top.

### 1.2 Capacity verification (`positive_inversion`)

For each of 50 triplets in `data/triplets/<i>/` (each containing
`video.mp4`, `actions.npy`, `prompt.txt`, two `frame_<N>.png` markers),
we ran per-clip optimization: find a per-step positive text embedding
`{C_t}_{t=0..24}` (with `L_pos=1`, i.e. a single 4096-vector per
sampling step) such that the Wan CFG sampler with frozen
`null = T5("")` reproduces the reference video slice
`video[start : start + 33]`.

Implementation: `scripts/embedding_search.py --mode positive_inversion`,
batched across all 50 triplets via `scripts/batch_inversion.py` (slurm
array, 5 shards × 10 triplets, ~75 GPU-min total).

**Result:** all 50 reconstructions land near the VAE round-trip ceiling.
**Wan can express each video** through per-step `L=1` positive text
embedding alone. Capacity is not the limiting factor.

Output: `runs/batch_inv_positive/triplet_<i>/positive_embeddings.pt` for
each video, plus `vae_roundtrip.mp4` / `ddim_inv_only.mp4` /
`reconstruction.mp4` for visual inspection.

### 1.3 Cross-video factorization (`cross_video_factorize`)

`scripts/cross_video_factorize.py` loads all 50 `positive_embeddings.pt`,
stacks them into `[V=50, N=25, 1, 4096]`, and computes:

  - **Shared mean trajectory** `μ_t = mean_v C_t^v` ∈ `R^{25 × 1 × 4096}`
  - **Per-video residual** `R^v = C^v − μ`
  - **Rank-1 SVD of each `R^v`** → per-video `(α^v_t, b^v)` with `α^v ∈ R^25`, `b^v ∈ R^{4096}` (unit)
  - **Shared α curve** `β_t = mean_v α^v_t` (after sign-alignment)
  - **Per-video scalars** `γ^v = ⟨α^v, β⟩ / ⟨β, β⟩`

Cross-video diagnostics over V=50:

| Quantity | Mean | Min | Max |
|---|---|---|---|
| Pairwise cosine of `α^v` curves | **0.997** | 0.963 | 1.000 |
| Pairwise cosine of `b^v` directions | -0.020 | -0.260 | +0.352 |
| Per-video rank-1 fraction of residual | 0.699 | 0.638 | 0.765 |
| γ_v | 1.000 | 0.827 | 1.837 |

The α-curve is essentially identical across every video (cosine 0.96–1.00),
while the b-directions are mutually near-orthogonal in R^4096. Each
video's deviation from the shared mean lives on a video-specific 1D
direction with a shared σ-schedule. The factorization

    C_t^v  ≈  μ_t  +  β_t · γ^v · b^v

captures ~70% of each video's residual variance. The remaining ~30% is
higher-order (rank 2+) structure or noise.

Saved: `runs/batch_inv_positive/factorization/adaptor_init.pt` with
`μ [25, 1, 4096]`, `β [25]`, per-video `γ [50]` and `b [50, 1, 4096]`,
`sigmas [26]`, and `names`.

### 1.4 Linear probe failure (`probe_actions_to_factors`)

`scripts/probe_actions_to_factors.py` runs 5-fold-CV ridge regression
from action representations to `(γ_v, b_v)`. Tested representations:
`flat` (raw `32×8` flatten), `delta` (each row minus row 0), `velocity`
(finite differences), `delta_plus_first` (concat of initial pose and
deltas).

Result table:

| Action repr | Action cosine (off-diag mean) | γ R² | b cos (full) | b cos (top-10 PC) |
|---|---|---|---|---|
| flat                 | 0.898 | -0.236 | -0.044 | -0.054 |
| delta                | 0.070 | -0.136 | -0.023 | +0.004 |
| velocity             | 0.021 | -0.126 | -0.031 | -0.031 |
| delta_plus_first     | 0.529 | -0.141 | -0.038 | -0.046 |

**The action-diversity problem was real for `flat`** (cosine 0.9 across
videos = nearly the same vector for every triplet). `delta`/`velocity`
fix it (cosine drops to ~0.02–0.07).

**The factors remain unpredictable** even after fixing action diversity.
Both γ R² and b cosine are at chance. Linear regression from actions
alone cannot recover `(γ_v, b_v)`.

### 1.5 Identifiability check

Re-ran `positive_inversion` on `triplet_0` with `seed=1` instead of
`seed=0`. Compared the resulting `{C_t}` trajectory to the saved one:

    trajectory cosine (seed 0 vs seed 1):  0.515

`arccos(0.515) ≈ 59°`. About half the trajectory direction is *not* a
function of `(action, I_0)` — it is whichever Adam basin happened to
win for that seed.

### 1.6 What §1.4 + §1.5 together imply

Linear-probe failure has at least three possible causes:
  (a) `(action, I_0) → factors` is non-linear (need a deeper probe);
  (b) actions alone are insufficient (need I_0 features);
  (c) factors are not a function of input (basin-dependent noise).

The 0.515 identifiability result confirms **(c) is real**: half the
target is by construction not predictable from input. No model — linear,
MLP, or arbitrarily-deep — can predict a non-function.

This kills the original two-stage plan ("run inversion to label data,
then train adaptor with MSE on factors"). The targets it would produce
are too noisy for the adaptor to fit.

The fix is to drop the factor-regression stage entirely and train the
adaptor against the ground-truth videos directly, using Wan's own
flow-matching denoising loss. Stochastic `t` and `ε` sampling at every
training step naturally averages over basin noise.

### 1.7 Repository inventory

What's in the repo right now:

| Path | Status | Role |
|---|---|---|
| `wan/` | upstream | Wan TI2V-5B model and inference pipeline |
| `scripts/embedding_search.py` | ✓ | Per-clip null/positive/embed/noise inversion |
| `scripts/batch_inversion.py` | ✓ | Orchestrator that reuses one loaded Wan pipeline across triplets |
| `scripts/cross_video_factorize.py` | ✓ | Cross-video factorization → `adaptor_init.pt` |
| `scripts/probe_actions_to_factors.py` | ✓ | Linear probe — failed (Sec. 1.4) |
| `scripts/analyze_embeddings.py` | ✓ | Within-video PCA on a single inversion run |
| `run_batch_inversion.sh`, `run_factorize.sh` | ✓ | SLURM submission scripts |
| `data/triplets/<i>/video.mp4` | ✓ | 50 reference clips at 60 fps |
| `data/triplets/<i>/actions.npy` | ✓ | (32, 8) action sequence per triplet |
| `data/triplets/<i>/frame_<N>.png` | ✓ | Marker frames; encode `(start_index, delta_index)` per triplet |
| `runs/batch_inv_positive/triplet_<i>/positive_embeddings.pt` | ✓ | 50 saved per-clip inversion results (L=1 positive context, 25 steps) |
| `runs/batch_inv_positive/factorization/adaptor_init.pt` | ✓ | `μ`, `β`, per-video `γ`/`b`, sigmas |
| `scripts/precompute_features.py` | ✓ | VAE-encode `I_0` *and* `z_video` per triplet → `data/triplets/<i>/{z_I0,z_video,meta}.{pt,json}` |
| `models/trajectory_adaptor.py` | ✓ | Rank-1 adaptor `nn.Module` |
| `scripts/train_adaptor.py` | ✓ | End-to-end denoising-loss training with frozen Wan DiT |
| `run_precompute_features.sh`, `run_train_adaptor.sh` | ✓ | SLURM wrappers |
| **`scripts/eval_adaptor.py`** | ✗ | Held-out reconstruction + SSIM (next phase) |

---

## 2. What we want to implement and why

### 2.1 Goal

Build a small adaptor `f_θ(actions, I_0) → {C_t}_{t=0..N-1}` whose output
is fed into Wan's cross-attention as the positive context. At inference,
sample with Wan's normal CFG flow-matching sampler using the adaptor's
`{C_t}` and a frozen `T5("")` as null context.

### 2.2 Architecture: rank-1 with shared σ-tables

```
inputs:
    actions ∈ R^{32 × 8}       (32 timesteps × 8 dims; 7 joint commands + 1 gripper)
    z_I0    ∈ R^{48 × 1 × H_z × W_z}   (Wan VAE encoding of I_0)

action_enc:   small MLP / 1-D temporal block      →  R^{hidden}
image_enc:    attention-pool over VAE patch tokens →  R^{hidden}
fuse:         MLP([h_action; h_image])             →  R^{hidden}

gamma_head:   Linear → R^{1}
b_head:       Linear → R^{L*D=4096}, unit-normalized after reshape to [L, D]

trajectory assembly:
    C_t = μ[t] + β[t] · γ · b      for t = 0..24
output shape: [B, 25, 1, 4096]

trainable parameters (with defaults hidden=512, L=1, D=4096):
    μ           [25, 1, 4096]   = 102,400    init from adaptor_init.pt['mu']
    β           [25]                          init from adaptor_init.pt['beta']
    action_enc  ~ 0.4M
    image_enc   ~ 0.07M  (linear proj + 4-head MHA + learned 1-token query)
    fuse        ~ 0.79M
    gamma_head  513
    b_head      2.1M
    total       ~4.5M parameters
```

The realized parameter count is ~4.5M (not the 2.5M estimated in the
original spec); the bulk is `b_head` (hidden=512 → 4096).

### 2.3 Why rank-1 (not per-σ MLP)

| | Rank-1 (this design) | Per-σ MLP `f(action, I_0, σ) → C_σ` |
|---|---|---|
| Matches measured data structure | ✓ (Sec. 1.3) | — must rediscover |
| Initialization | μ, β warm-started from data | random; first ~10³ steps spent leaving OOD |
| σ-coverage | smooth β interpolation across N | depends on `t`-sampling distribution |
| Sample efficiency at V=50 | high | low |
| Expressive ceiling | rank-1 only | strictly higher |

For V=50 training videos the inductive bias matters. Per-σ MLPs become
competitive somewhere around V = 10³–10⁴. If the rank-1 adaptor
underfits, the escalation is rank-K (add `(γ_2, b_2)` heads) before
dropping the structure entirely.

### 2.4 Why include I_0 even though Wan's i2v latent pin sees I_0

Wan's i2v latent pin gives **self-attention** access to I_0. The
adaptor's output, however, feeds **cross-attention** — a separate
pathway whose K, V projections were trained to consume embeddings that
described both scene content and motion (T5 captions).

Without I_0 in the adaptor:
  - Adaptor K/V only encode action info → distributional mismatch with
    what the frozen cross-attention expects.
  - Possible failure modes: cross-attention attends uselessly (no signal,
    drift toward unconditional), or worse, adaptor invents scene content
    that conflicts with self-attention's I_0 readout (wrong colors/objects).

Including I_0 in the adaptor lets it produce K, V that lie in the same
distribution Wan's cross-attention was trained on. The frozen DiT cannot
re-route information internally, so we must give it the right channel
shape.

### 2.5 Why end-to-end denoising loss (not factor regression)

For each training step:

```
L(θ) = E_{(I_0, video, action), t, ε}  || v_base( z_t, σ_t, f_θ(action, I_0)[t] ) − v_target ||²
where:
    z_0    = VAE.encode(video)
    ε      ~ N(0, I)
    z_t    = (1 − σ_t) · z_0 + σ_t · ε       (flow-matching forward path)
    z_t    = first_frame_pin(z_t, z_I0)
    v_target = ε − z_0                        (flow-matching target)
    t      ~ p(t)                             (logit-normal or uniform)
```

Three reasons this beats label-then-SFT against the saved `(γ, b)`:

1. **Stochastic `t` and `ε` average over basin noise.** Every step
   integrates over many noise realizations; the adaptor learns the
   *deterministic* part of the conditioning, which is exactly the
   "function of `(action, I_0)`" we want.
2. **Targets are ground-truth videos**, not optimization outputs. No
   per-seed noise to inherit.
3. **Loss surface matches deployment.** The adaptor is trained against
   the same loss Wan was trained against, so the resulting K/V land
   on-distribution.

The factorization analysis isn't wasted — we use `μ` and `β` as
architectural warm-starts, and the rank-1 form as the inductive bias.

### 2.6 What's out of scope (for now)

- Multi-seed factor averaging to "denoise" labels (only revisit if
  end-to-end stalls).
- DINOv2 / SigLIP image features (start with Wan VAE; swap later if
  needed).
- Longer horizons than `delta_index=32` (33 frames, ~1.4s @ 24fps).
  Architecture supports it; just need new inversions to refit `μ`/`β`
  if the schedule changes.
- Action augmentation, multi-step rollout, planning.

---

## 3. Concrete implementation (as built)

### 3.1 Files

| Path | Role |
|---|---|
| `scripts/precompute_features.py` | One-time: VAE-encode `I_0` and 33-frame clip per triplet |
| `models/__init__.py` | (empty) marks `models/` as a package |
| `models/trajectory_adaptor.py` | `TrajectoryAdaptor` `nn.Module` (rank-1, σ-indexed) |
| `scripts/train_adaptor.py` | End-to-end denoising-loss training loop |
| `run_precompute_features.sh` | SLURM wrapper for precompute |
| `run_train_adaptor.sh` | SLURM wrapper for training |

Deviation from the original §3.1 plan: `precompute_i0_features.py` and a
separate `z_video` job were merged into a single
`scripts/precompute_features.py` so the Wan VAE is loaded only once.

The eval script (`scripts/eval_adaptor.py`) is still TODO — milestone M2
will need it to compare held-out reconstructions to ground truth.

### 3.2 `scripts/precompute_features.py`

Reads each `data/triplets/<i>/` directory:

1. Parses `(start_index, delta_index)` from the two `frame_<NNN>.png`
   filenames (same logic as `scripts/batch_inversion.py`).
2. Loads `delta_index + 1 = 33` consecutive frames from `video.mp4` using
   decord.
3. Re-uses `embedding_search.fit_clip_to_pipeline` so the pixel grid
   exactly matches the grid used during `positive_inversion`. Defaults to
   `max_area=230400` (480×480 budget — what the inversions actually ran
   at; **not** the 480×480 default in the script's CLI).
4. VAE-encodes:
   - `I_0` → `z_I0.pt` shape `[48, 1, H_z, W_z]`
   - the 33-frame clip → `z_video.pt` shape `[48, 9, H_z, W_z]`
5. Writes `meta.json` with `(start_index, delta_index, oh, ow,
   z_I0_shape, z_video_shape)` for sanity checks at training time.

The job skips any triplet whose three artifacts already exist, so it is
resumable. Verified that all 50 triplets share the same `(H_z, W_z) =
(22, 40)` (so dataloader collation can stack tensors without padding).

CLI:
```
python scripts/precompute_features.py \
    --triplets_root data/triplets \
    --ckpt_dir Wan2.2-TI2V-5B \
    --max_area 230400              # MUST match the inversion's max_area
```

Cost: ~5 min total for 50 triplets on a single H100.

### 3.3 `models/trajectory_adaptor.py`

```python
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
        action_repr: str = "delta",       # "delta" (default) or "raw"
    ):
```

Components (matches §2.2 with one deviation: `b_head` outputs `L*D` not
just `D`, so the same module supports `L > 1` if a future inversion run
uses longer per-step embeddings):

- `μ`, `β`: `nn.Parameter` (learnable; warm-started from `adaptor_init.pt`).
- `action_enc`: `Linear(action_len*action_dim, hidden) → SiLU → Linear(hidden, hidden)`.
  Input is **delta-actions** (`actions − actions[:, 0:1]`) by default,
  motivated by the §1.4 probe (delta brought cross-video action cosine
  from 0.90 to 0.07). `action_repr="raw"` is an opt-out.
- `image_enc`: `Linear(48 → hidden) → MultiheadAttention(heads=4)` with a
  single learned query token. Pools over the spatial+temporal latent
  tokens of `z_I0` (handles `F_z > 1` if someone ever passes a multi-frame
  conditioning latent).
- `fuse`: 2-layer MLP.
- `gamma_head`: `Linear(hidden, 1)`, weight init `std=1e-3`, zero bias.
- `b_head`: `Linear(hidden, L*D)`, weight init `std=1e-4`, zero bias.

**Key initialization choice (deviation from spec).** The original §3.7
risk row recommended `b_head ≈ 0` so initial output = `μ`. We instead
keep both heads at *small but nonzero* init. Strict-zero `γ` was tried
first; it made `∂loss/∂(b_head, encoders) = 0` at step 0 (because in
`γ · b` the multiplier `γ` zeroes the chain rule), which would freeze
all upstream parameters for the first optimizer step. The current small-
nonzero init yields `||C_init − μ|| / ||μ|| ≈ 6e-4` (still satisfies the
M0 "≈ μ" check) and gradients reach every parameter from step 1.

Helper APIs:

- `freeze_mu_beta()` / `unfreeze_mu_beta()` — supports the staged plan in
  open question §4.3 ("start frozen for ~10k steps, then unfreeze").
- `parameter_groups(lr, lr_mu_beta=None)` — returns two AdamW groups so
  μ, β get a smaller LR (default `lr/10`). Avoids the warm-start being
  torn apart in the first few steps.
- `initial_factors(actions, z_I0)` — returns `(γ, b)` for the M0
  diagnostic ("at init, γ should be ~0").

Forward returns `[B, N, L, D]`. Internally:

```
γ      = gamma_head(h)                                   # [B, 1]
b̂      = b_head(h)                                       # [B, L*D]
b_unit = b̂ / ||b̂||                                       # [B, L*D]
b      = b_unit.view(B, L, D)                            # [B, L, D]
C_t    = μ[None,:,:,:] + β[None,:,None,None] · (γ·b)[:, None, :, :]
```

### 3.4 `scripts/train_adaptor.py`

Top-level structure:

1. **Pipeline build.** `WanTI2V(...)` with `convert_model_dtype=True`;
   `pipe.model`, `pipe.vae.model`, `pipe.text_encoder.model` all forced
   to `.eval().requires_grad_(False)`.
2. **σ-schedule.** Built once via `FlowUniPCMultistepScheduler` with
   `sampling_steps=25, shift=5.0`. Verified to match the σ-tensor stored
   in `adaptor_init.pt` to float-precision (L∞ diff = 0).
3. **Cache `T5("")`.** Encoded once, then T5 moves to CPU and stays
   there. Used both for CFG-dropout and for inference-time null context.
4. **Adaptor build.** `TrajectoryAdaptor(...)` on GPU, optionally with
   `freeze_mu_beta()` for the first `freeze_mu_beta_steps` steps.
5. **Latent geometry.** `seq_len` and `mask2` (i2v pin mask, 0 at frame
   0) computed from `z_video.shape`.
6. **Optimizer.** AdamW with two param groups (`heads`, `mu_beta`),
   cosine-with-warmup schedule, gradient clipping (`grad_clip=1.0`).
7. **Gradient checkpointing.** Reuses `embedding_search._model_grad_
   checkpointing` (re-implemented as `model_grad_checkpointing` for
   import cleanliness): wraps every DiT block in
   `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
   `use_reentrant=False` is required so gradients flow back through the
   `context=` kwarg into the adaptor.

#### Per-step computation (`adaptor_denoising_loss`)

```python
actions   : [B, T, A]   on device
z_I0      : [B, 48, 1, H_z, W_z]
z_video   : [B, 48, F_z, H_z, W_z]
z_I0_full : z_I0 broadcast to [B, 48, F_z, H_z, W_z]  (for the i2v pin)

t       = sample_step_indices(B, N=25, sigmas, kind, ...)   # [B], int
σ       = sigmas[t]                                          # [B]
ε       ~ N(0, I)   shape of z_video
z_t     = (1 − σ) · z_video + σ · ε
z_t     = (1 − mask2) · z_I0_full + mask2 · z_t              # first-frame pin

C_traj  = adaptor(actions, z_I0)                             # [B, N, L, D]
C_t     = C_traj.gather(dim=1, index=t)                      # [B, L, D]
# CFG dropout: with prob p, replace C_t[b] with T5("")
# (T5("") is right-padded/truncated to L so the gather stays clean).

timestep = build_timestep_tensor(σ, mask2_zero, patch_size, seq_len, 1000)
                                                              # [B, seq_len]
# frame-0 patches → timestep 0; rest → σ · num_train_timesteps.
# Matches WanTI2V.i2v_diff's temp_ts construction.

with autocast(bf16):
    v_pred_list = pipe.model(
        [z_t[b] for b in range(B)],
        t=timestep,
        context=[C_t[b] for b in range(B)],
        seq_len=seq_len,
    )
v_pred   = torch.stack([v.float() for v in v_pred_list], 0)

v_target = ε − z_video
# Masked MSE over non-frame-0 positions:
mse = (v_pred − v_target)^2 · mask2_b
loss = mse.sum() / (mask2_b.sum() · C)
```

Notes on subtle choices:

- **Per-position timestep tensor.** Wan was trained with the first latent
  frame held clean (timestep 0) while the rest get the diffusion
  timestep. The training loss therefore lives entirely on the noisy
  positions; frame-0 velocity prediction would be irrelevant
  (overwritten by the pin) and including it in the loss just adds
  variance, so the loss is masked to `mask2 == 1`.
- **Mask normalization.** Loss divides by `mask2.sum() * C` so its
  magnitude is independent of how many frame slots are pinned — handy if
  the horizon ever changes from 33 frames.
- **Batch>1 handling.** Wan's DiT accepts lists of per-example tensors
  (used internally as the batch dimension) and a `t` of shape `[B,
  seq_len]`. `build_timestep_tensor` constructs `[B, seq_len]` directly,
  avoiding the `[1, seq_len]` shortcut in `embedding_search.py`.
- **CFG dropout.** With probability `cfg_dropout` (default 0.1) the
  positive context for that example is replaced with `T5("")`. This
  keeps the unconditional pathway from atrophying so that CFG > 1 at
  inference still steers in the intended direction. The replacement
  pads or truncates `T5("")` to length L so the gathered slice stays
  shape-aligned with the rest of the batch.
- **Autocast.** Wan's DiT runs in bf16; the adaptor stays fp32. Gradient
  cast happens implicitly across the autocast boundary.

#### CLI knobs (defaults)

| Knob | Default | Notes |
|---|---|---|
| `--batch_size` | 1 | DiT memory dominates; checkpointing on |
| `--total_steps` | 50,000 | Tune from training curves |
| `--lr` | 1e-4 | AdamW |
| `--lr_mu_beta` | `lr/10` | Auto if unset |
| `--weight_decay` | 1e-2 | |
| `--warmup_steps` | 500 | Linear warm-up |
| `--lr_min_ratio` | 0.1 | Cosine floor → `lr/10` |
| `--grad_clip` | 1.0 | Norm clip |
| `--t_sampling` | `uniform` | Or `logit_normal` (μ=0, σ=1) |
| `--cfg_dropout` | 0.1 | |
| `--freeze_mu_beta_steps` | 0 | Set >0 to enable §4.3 staging |
| `--val_interval` | 1000 | Steps between val passes |
| `--val_eps_repeats` | 8 | (ε, t) draws averaged per val example |
| `--ckpt_interval` | 5000 | |
| `--val_names` | `45 46 47 48 49` | Held-out triplet folder names |
| `--overfit_one` | `None` | Single-triplet mode (M1) |
| `--sampling_steps` | 25 | Must match inversion's σ-grid |
| `--shift` | 5.0 | Must match inversion |

Note the divergence from the spec table at §3.4: default `t_sampling` is
`uniform` rather than `logit_normal`. `uniform` was easier to reason
about for the first run; `--t_sampling logit_normal` is a one-flag swap
once curves are stable.

### 3.5 Caching / dataloader

`scripts/precompute_features.py` writes `z_I0.pt` (~80 KB) and
`z_video.pt` (~700 KB) per triplet. Total ~40 MB for 50 triplets — much
smaller than the originally estimated 500 MB.

`TripletLatentDataset` in `scripts/train_adaptor.py` returns per item:

```python
{
    "name":    "triplet_<i>",
    "actions": torch.Tensor [32, 8],
    "z_I0":    torch.Tensor [48, 1, H_z, W_z],
    "z_video": torch.Tensor [48, 9, H_z, W_z],
}
```

A simple `collate` stacks the four tensors along a batch dim. No padding
needed because all triplets share `(H_z, W_z) = (22, 40)`.

### 3.6 Milestones / sanity checks (in this order)

**M0. Adaptor forward pass at init.**
Status: **passes**.
- `||C_init − μ|| / ||μ|| ≈ 6e-4` — output is essentially μ.
- Initial γ has tiny but nonzero magnitude (~1e-3 scale) so all parameters
  receive gradient from step 1.
- Decode-through-Wan: `scripts/eval_adaptor.py --ckpt_path none` renders
  the μ-only trajectory; it is a coherent, on-manifold video.

**M1. Single-triplet overfit.**
```
python scripts/train_adaptor.py --overfit_one 0 ... --total_steps 5000
```
Reconstruction should approach `runs/batch_inv_positive/triplet_0/
vae_roundtrip.mp4` quality.

Status: **passes** (5k-step overfit on triplet 0). The adaptor's
`sample.mp4` is coherent and visually matches `ground_truth.mp4`,
sampled from *fresh random noise*.

**Empirical confirmation of §1.5/§1.6 (noise-basin robustness).** When
`scripts/eval_adaptor.py --include_oracle` replays the *saved* per-step
inversion embeddings from fresh random noise, the decode blows out to
pure white — the inversion's `{C_t}` are tied to the specific noise
basin (`z_init = traj[0]`) they were optimized in and break outside it.
The adaptor, trained with stochastic ε at every step, samples correctly
from any noise. This is exactly the identifiability argument that
motivated dropping factor-regression in favor of end-to-end denoising.
Consequence for evaluation: `eval_adaptor.py` replays the oracle from
its *saved* `z_init` (not fresh noise) so it reproduces the inversion's
own `reconstruction.mp4` ceiling; the adaptor sample uses fresh noise.
The two videos therefore use different initial noise and are not a pixel
A/B — compare them on motion/content fidelity, not per-pixel SSIM.

**M2. Hold-out small.**
Default config in `run_train_adaptor.sh`: train on triplets 0–44, validate
on 45–49.

**M3. Hold-out scaling.**
Re-train at V=100, 200 once more triplets are collected; track
generalization curve.

### 3.7 Risks and fallbacks

| Risk | Detection | Fallback |
|---|---|---|
| Rank-1 underfits | Held-out reconstructions blurry / drift | Add `(γ_2, b_2)` head (rank-2). Strict capacity bump, no rewrite of the module. |
| `μ`, `β` warm-starts are wrong because of basin noise in original inversions | At training-step 0 the model output already looks worse than reference | Either re-average `μ`, `β` over multiple seeds per video, or unfreeze and let them learn from scratch with a longer schedule. |
| Wan VAE features inadequate | Reconstructions content-drift; wrong object color/shape | Swap `encode_image` to a frozen DINOv2/SigLIP; one-line change at the interface. |
| Action representation insufficient | γ doesn't correlate with motion magnitude | Try end-effector pose via forward kinematics; or add velocity features alongside deltas. |
| Memory OOM | Backward fails during DiT forward | Already on per-block grad-checkpointing + bf16; drop `batch_size` to 1; or skip CFG-dropout to free memory. |
| CFG-at-inference behaves badly | Optimized embeddings work at training σ but fail under deployment CFG | `--cfg_dropout 0.1` already on. Increase to 0.2 or train at deployment CFG explicitly. |
| Optimizer stuck because γ=0 at init zeros all upstream grads | Loss flat for first ~hundreds of steps; b_head / encoder weights unchanged | Already mitigated: init keeps `γ` small but nonzero (§3.3). If still observed, increase `gamma_head` init std. |

### 3.8 Definition of "done" for this phase

- M0, M1, M2 all pass.
- A short qualitative report comparing held-out (`reconstruction.mp4`)
  vs ground-truth videos for 5 held-out triplets.
- A regression metric (SSIM avg over all frames, mean over held-out)
  better than the heuristic-prompt baseline used during inversion
  studies.

After this we either:
  - Scale data (more triplets) and retrain → see if generalization
    improves.
  - Move on to longer horizons / multi-step rollout.
  - Move on to inference-time experimentation (CFG ablations, action
    extrapolation).

### 3.9 End-to-end run recipe

```bash
# 1. One-time VAE preprocessing (~5 min on one H100).
sbatch run_precompute_features.sh

# 2. Train. Defaults: 50k steps, batch_size=1, 45/5 train/val split.
sbatch run_train_adaptor.sh
#   environment overrides honored:
#   OUTPUT_DIR=runs/adaptor_train_v1 TOTAL_STEPS=10000 sbatch run_train_adaptor.sh

# 3. Single-triplet overfit (M1). Run interactively or via sbatch:
.venv/bin/python scripts/train_adaptor.py \
    --triplets_root data/triplets \
    --adaptor_init runs/batch_inv_positive/factorization/adaptor_init.pt \
    --ckpt_dir Wan2.2-TI2V-5B \
    --overfit_one 0 \
    --total_steps 5000 \
    --val_interval 500 \
    --output_dir runs/adaptor_overfit_t0
```

Training outputs (under `--output_dir`):

| File | Contents |
|---|---|
| `config.json` | All CLI args verbatim |
| `train_log.csv` | per-step `loss, sigma_mean, t_mean, |v|, lr, grad_norm` |
| `val_log.csv` | per-validation `val_loss` |
| `ckpt_step<N>.pt` | adaptor state + optimizer state every `ckpt_interval` |
| `ckpt_latest.pt` | symlink-style latest copy |

### 3.10 Second dataset / resolution: DROID at 320×192

A separate verification track confirms the method transfers to another
dataset and resolution: the `droid_ctrl_world` dataset
(`/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world`), 320×192 per camera
view.

Dataset shape (differs from the original triplets):

| | Original triplets | DROID `droid_ctrl_world` |
|---|---|---|
| Pixel resolution | fitted to ~480² grid (608×352 px) | native **320×192** |
| Latent grid (H_z, W_z) | (22, 40) | **(12, 20)** |
| `seq_len` | 1980 | **540** |
| Action vector | `[32, 8]` (7 joints + gripper) | `[32, 7]` (6 cartesian Δ + gripper) |
| Views | 1 | 3 (we use **view 0** for verification) |
| Source fps | 60 | ~5 |
| Provided latents | — | `latent_videos/*.pt` are `[T,4,24,40]` from a **4-channel image VAE**, NOT Wan's — ignored; we re-encode with Wan's VAE |

Why this works without re-deriving the warm-start:
- **Resolution.** 320×192 is divisible by Wan's `vae_stride·patch = 32` in
  both axes, so `best_output_size` returns it unchanged when
  `max_area == 320·192 = 61440`. (Leaving `max_area=230400` would *upscale*
  to 608×352 — the precompute pins native area.)
- **μ/β warm-start is resolution-independent.** μ `[25,1,4096]` and β `[25]`
  live in T5 text-embedding space, not pixel/latent space. They are reused
  from the 480²-grid factorization as a generic on-manifold init. The image
  encoder attention-pools over whatever spatial token count z_I0 has, so
  (12, 20) needs no code change. `seq_len`, `mask2`, and the per-position
  timestep tensor are all derived from the loaded latent shape at runtime.
- **Action dim is a constructor arg.** `--action_dim 7` is the only model-side
  change; `b_head`, `gamma_head`, μ, β are untouched.

Files / commands:

```bash
# 1. Encode a verification subset (view 0) with Wan's VAE at native 320×192.
#    Writes the SAME per-dir cache format train_adaptor.py already reads, into
#    data/droid_cache/{train,val}/<epK_v0>/{z_I0,z_video,actions,meta}.
TRAIN_LIMIT=300 VAL_LIMIT=30 sbatch run_precompute_droid.sh

# 2. Train (separate train/ val/ roots via --val_triplets_root; action_dim 7).
sbatch run_train_droid.sh

# 3. Evaluate held-out DROID episodes (no oracle — DROID has no inversion run).
CKPT_PATH=runs/adaptor_droid_v0/ckpt_latest.pt \
OUTPUT_DIR=runs/eval_droid_v0 \
EVAL_NAMES="$(ls data/droid_cache/val | tr '\n' ' ')" \
GUIDE_SCALE=5.0 \
sbatch run_eval_adaptor.sh   # drop --include_oracle for DROID
```

Implementation notes:
- `scripts/precompute_features_droid.py` reads `actions/<split>/<ep>.json`
  and `annotation/<split>/<ep>.json`, slices a 33-frame clip from
  `videos/<split>/<ep>/<view>.mp4` starting at `--start_frame`, encodes I_0
  and the clip, and writes `actions.npy` `[32, 7]` aligned to the clip
  transitions plus a `meta.json` (records `video_path`, `start_frame`,
  `frames`, `max_area` so eval can re-load real frames for SSIM).
- `train_adaptor.py` gained `--val_triplets_root`: when set, train uses all
  subdirs of `--triplets_root` and val uses all subdirs of the val root
  (DROID has its own train/ and val/ folders). Legacy single-root
  name-carving still works when it is unset.
- `eval_adaptor.py:load_ground_truth_pixels` reads `meta.json` when present
  (DROID) to fetch the real frames; for the old triplets it still uses the
  `frame_<NNN>.png` markers. If the source video isn't reachable it scores
  SSIM against the VAE round-trip of `z_video` instead.
- **Temporal-domain caveat.** DROID is ~5 fps, so a 33-frame clip spans
  ~6.6 s — much slower motion than the 60 fps clips the original μ/β were
  fit from. Wan is frozen; the adaptor just learns the mapping that fits.
  Acceptable for a resolution-verification run; revisit frame striding if
  motion looks too fast/slow.

Verification status: precompute + full datapath validated at 320×192
(latents `[48, 9, 12, 20]`, actions `[32, 7]`, `seq_len=540`, adaptor
`C_init == μ`); an end-to-end training-step pass through the frozen DiT at
this resolution runs without error.

### 3.11 DROID v0 result: the reused warm-start does NOT transfer

First DROID run (`runs/adaptor_droid_v0`, 300 train / 30 val, 30k steps,
reusing the 480²-grid `adaptor_init.pt`): **negative**. Diagnosis via the
§3.6 2×2 (eval at `runs/eval_droid_train`, `runs/eval_droid_init`):

| Probe | Observation | Implication |
|---|---|---|
| Eval on **training** episodes | quality ≈ held-out val (both mediocre) | not a generalization gap — the adaptor **underfits even seen data** |
| Eval at **init** (μ-only, γ=0) | "completely off" — off-manifold video | the reused μ/β is the **wrong baseline** for DROID |

Loss had looked benign (val ↓ 24%, plateau by ~6k steps) precisely because
at high σ the cross-attention context barely moves the per-step denoising
MSE — so a low/decreasing loss coexisted with broken video. This is the
concrete demonstration of "the loss curve cannot certify the method; only
decoded eval can."

Root cause: μ is the *mean of the per-step contexts that reconstructed the
old 480²/60fps clips*. Although it lives in T5 text space and is
dimensionally valid at any resolution, it is anchored to the old domain's
manifold region and is off-manifold for DROID. The small rank-1 residual
`γ·b` could not drag generation back. Capacity is **not** the issue — the
320×192 `positive_inversion` capacity test passed (per-step *optimized*
embeddings reconstruct DROID near the VAE ceiling), so Wan is expressive
here; we simply handed the adaptor the wrong center μ and residual scale β.

**Fix: re-derive μ/β from DROID at 320×192** (the choice deferred in the
§3.10 question, now required):

```bash
# 1. Invert ~32 DROID clips at 320x192 + factorize -> in-domain adaptor_init.pt
LIMIT=32 sbatch run_inversion_droid.sh
#    (= scripts/batch_inversion_droid.py then scripts/cross_video_factorize.py)

# 2. Inspect runs/droid_inv/train/factorization/cross_video_report.json:
#      alpha_cosine_off_diag.mean ~0.99  AND  rank1_fraction.mean ~0.7
#    -> rank-1 structure holds in DROID, proceed.
#    If much lower -> rank-1 is the wrong bias here; escalate to rank-2
#    (add a second (γ_2, b_2) head) before scaling data.

# 3. Retrain pointing at the new warm-start:
OUTPUT_DIR=runs/adaptor_droid_v1 \
  ADAPTOR_INIT=runs/droid_inv/train/factorization/adaptor_init.pt \
  sbatch run_train_droid.sh        # (edit the wrapper's --adaptor_init, or pass through)

# 4. Re-run the §3.6 2×2. Init (μ-only) should now be a coherent DROID
#    video; trained eval should beat null and show action-driven motion.
```

The DROID inversion driver (`scripts/batch_inversion_droid.py`) is the
DROID analogue of `batch_inversion.py` — it reuses one loaded pipeline
across episodes and writes the same `positive_embeddings.pt` layout
`cross_video_factorize.py` already consumes. It also yields **oracle**
references per clip: replay an episode's saved embeddings from its saved
`z_init` and compare to the adaptor sample to separate a remaining
learnability gap from any residual capacity gap.

Eval-script footguns fixed while diagnosing this (commit alongside):
- a non-`none` `--ckpt_path` that doesn't exist now raises instead of
  silently dropping to init mode (a missing `ckpt_step028000.pt` had
  masqueraded as an init-mode run);
- `eval_adaptor.py` gained `--action_dim/--action_len/--action_repr/--hidden`
  for init mode (no saved args to read), so μ-only DROID eval needs
  `--action_dim 7`.

### 3.12 Configurable architecture variants

The adaptor (`models/trajectory_adaptor.py`) is now config-driven along two
orthogonal axes so design choices can be A/B'd. Both fusions emit a
conditioning vector `h ∈ R^hidden`; both heads consume it; defaults
(`concat` + `rank1`) reproduce the original rank-1 network. Every variant
preserves `C_init ≈ μ` (warm-start intact) **and** full gradient flow to all
parameters at step 1 (small-but-nonzero last-layer init, not zero).

`--arch_fusion` — how (actions, z_I0) combine into `h`:

| value | mechanism | addresses |
|---|---|---|
| `concat` (default) | image attention-pooled to **1 token**, actions MLP'd from flattened deltas, the two vectors concatenated → MLP. No spatial positions, no cross-modal interaction. | — (control) |
| `cross_attn` | image kept as **all spatial tokens** with 2-D sin-cos positional encoding (resolution-agnostic), actions kept as **per-step tokens** with temporal positional encoding; a learnable query token cross-attends over the concatenation across `--n_xattn_layers` layers. | image-granularity loss; lack of action↔image interaction |

`--arch_head` — how `h` maps to the per-step residual `C_t − μ_t`:

| value | residual | addresses |
|---|---|---|
| `rank1` (default) | `β_t · γ · b`, one shared unit direction | — (control) |
| `rankk` (`--rank_k K`) | `Σ_{k=1}^K β_{t,k} · γ_k · b_k`, K shared directions; β is `[N,K]` (col 0 warm-started, rest 0) | rank-1 may be too low-rank |
| `perstep` | `g_θ([h, step_emb_t])` — a step-conditioned MLP predicting the residual directly | the shared-direction / linear assumption may not hold at all |

Param counts (hidden=512): concat/rank1 4.5M, concat/rankk(K=4) 10.8M,
concat/perstep 4.8M, cross_attn variants +~4.1M for the token encoders +
attention stack.

Checkpoints save the arch flags in `ckpt["args"]`; `eval_adaptor.py`
reconstructs the right variant automatically (saved args win; CLI flags are
the fallback for init mode).

Sweep recipe (after the §3.11 in-domain warm-start is in place):

```bash
INIT=runs/droid_inv/train/factorization/adaptor_init.pt
for cfg in "concat rank1" "cross_attn rank1" "cross_attn rankk" "cross_attn perstep"; do
  set -- $cfg
  OUTPUT_DIR=runs/droid_${1}_${2} ADAPTOR_INIT=$INIT \
  ARCH_FUSION=$1 ARCH_HEAD=$2 sbatch run_train_droid.sh   # wrapper passes these through
done
```

Recommended escalation order if a variant still underfits: `concat/rank1`
(control) → `cross_attn/rank1` (isolates the fusion fix) → `cross_attn/rankk`
→ `cross_attn/perstep` (isolates dropping the linear bias). Comparing the
control to `cross_attn/rank1` tells you how much the granularity+interaction
matter; comparing `rank1 → rankk → perstep` at fixed fusion tells you how
much the rank-1 assumption was costing.

### 3.12b Action-conditioned Wan side adapters

The context-trajectory family above still predicts `C_t` outside the DiT, so it
does not see the actual sampled noisy latent `z_t`. The prototype in
`models/action_conditioned_wan.py` tests a deeper alternative: keep Wan frozen
and inject action into Wan's own hidden states, which already encode `z_t`,
timestep, text, and the pinned first-frame latent.

See `docs/action_conditioned_wan.md` for the implementation details. The three
prototype modes are:

| mode | mechanism | purpose |
|---|---|---|
| `append_context` | append action tokens to normal text context | cheapest action-token baseline |
| `replace_context` | replace text context with action tokens | pure action-only ablation |
| `pre_context` | use Wan pre-cross-attention features + action to predict context, then run Wan with that context | non-circular context generator aware of `z_t`, `t`, and pinned `I0` |
| `side_adapter` | add zero-init bottleneck action residuals after selected frozen Wan blocks | main noise/timestep-aware redesign |

### 3.13 The init/train/eval CFG-consistency bug (train AND eval at w=5)

First arch-sweep result was bad on **all** variants, including **training**
episodes — the signature of a regime mismatch, not a learning failure (an
overfit model must reconstruct its own training clips if eval is faithful).
The pipeline had a **three-way CFG inconsistency**:

| Stage | CFG scale it assumed |
|---|---|
| μ/β warm-start + oracle (from `positive_inversion`) | **w = 5** |
| Training (original `adaptor_denoising_loss`) | w = 1 (no CFG) |
| Eval (original default) | w = 5 |

What the w=5 inversion actually optimized:
```
v_uncond + 5·(v_cond − v_uncond) ≈ v_target
⇒ v_cond(C_t) ≈ v_uncond + (1/5)(v_target − v_uncond)
```
So the *conditional* velocity at the inverted `C_t` — and at their mean **μ** —
is only ~1/5 of the way from the unconditional flow to the true flow; it is
deliberately "deflated" because CFG multiplies it back ×5.

Two symptoms, both explained:
- **Eval at w=5 of a w=1-trained model → broken/over-saturated.** The model
  was trained so `v_cond ≈ v_target`; `5·v_cond − 4·v_uncond` over-shoots.
- **Eval at w=1 of the same model → static, "replays frame 0".** Training
  started from the w=5 μ (whose `v_cond ≈ v_uncond`), and at w=1 there is no
  amplification, so the near-unconditional velocity (first frame pinned →
  no motion) is what gets integrated. The user's "guidance too small" was
  exactly right.

Why CFG-dropout did not rescue it: the DiT is **frozen**, so a dropout step
(`C_t → T5("")`) involves no adaptor parameters and yields **zero adaptor
gradient**. The adaptor never learns the unconditional branch — dropout is a
no-op here. (`cfg_dropout` default is now 0.)

**Fix (implemented): make all three stages consistent at w=5.** Training now
regresses the *CFG-combined* velocity to the flow target:
```
v_pred = v_uncond + w·(v_cond − v_uncond),   loss = ‖v_pred − (ε − z_video)‖²
```
`v_uncond` (frozen DiT + T5("")) does not depend on the adaptor, so it is
computed once under `no_grad` — ~1 extra forward, no extra backward.
`--train_guide_scale` defaults to **5.0**, matching the warm-start's
inversion scale; gradients flow through `v_cond` (scaled by w). At init
(`C_t ≈ μ`) the w=5-combined velocity ≈ the mean flow target, so μ is now a
*correct* warm-start and the adaptor only has to learn the per-video
residual — exactly the rank-1 design intent.

Eval auto-matches: `eval_adaptor.py --guide_scale -1` (the new default) reads
the checkpoint's `train_guide_scale`; an explicit value that disagrees prints
a warning. `run_eval_droid.sh`/`run_eval_sweep.sh` default `GUIDE_SCALE=auto`.

Wiring: `train_adaptor.py` gained `--train_guide_scale` (saved in
`ckpt["args"]`); `run_train_droid.sh` passes `TRAIN_GUIDE_SCALE` (default 5.0).

**Action: re-run the sweep with CFG-aware training, then eval (auto w=5).**
```bash
./run_sweep_arch.sh        # now trains the w=5 CFG-combined velocity
./run_eval_sweep.sh        # eval auto-matches train_guide_scale (=5)
```
The earlier `runs/droid_*` checkpoints were trained in the inconsistent w=1
regime and should be discarded/retrained. If w=5-consistent training still
underfits on *training* episodes, it is then a genuine capacity/data problem
(pursue rank-K/perstep, more episodes/views, multi-window).

Alternative regime (not default): fully no-CFG — re-derive μ/β from **w=1**
inversions (`--inversion_guide_scale`/replay at w=1) and train+eval at w=1.
Self-consistent too, but discards CFG's quality headroom; w=5 is preferred
because the existing inversions/oracle are already w=5.

### 3.14 Fixed-noise training (deterministic target; `--fixed_noise`)

Even CFG-consistent (w=5) training gave `sample ≈ null` on training clips —
the conditional velocity collapses toward the unconditional, i.e. the
adaptor doesn't steer. The factorization is sound (§3.13 table: rank-1
holds at 320×192, capacity 0.87 vs 0.96 ceiling), so this is a *fit* failure,
and it traces back to the §1.5 identifiability result: stochastic-ε training
asks the adaptor to learn the **noise-marginal** `C_t` — the context that
pulls *every* noise sample to the target. But the per-clip `C_t` that
reconstructs is basin-dependent (seed-0 vs seed-1 cosine 0.515), so the
noise-marginal target is washed-out / nearly non-functional, and the gradient
that should sharpen `b` is weak → steering collapses.

The task does not need noise-marginal conditioning: it is **deterministic**
(one ground-truth video per `(action, I_0)`). Fixing a single ε₀, shared by
training and eval, makes the target well-defined:

- training path `z_t = (1−σ)z_0 + σ·ε₀` (same ε₀ every step) lies on the
  *same* trajectory the eval ODE integrates from ε₀ — shrinking the
  teacher-forcing train/eval gap;
- the "correct" `C_t` for a clip becomes a deterministic function of
  `(action, I_0)` — there is one noise, one target, one answer; the §1.5
  basin ambiguity disappears;
- eval already fixes the seed (`--seed 0`, one global ε₀ since all clips
  share the latent shape), so this only changes *training* to match.

This turns the adaptor into an **amortized inversion** (learn, from
`(action, I_0)`, the per-step context that reconstructs from the shared ε₀),
which is far easier than the noise-marginal problem. Implemented as
`train_adaptor.py --fixed_noise` (ε₀ seeded by `--seed`, reused every step;
validation auto-uses 1 ε-draw). Wrappers expose `FIXED_NOISE=1`
(`run_overfit_droid.sh` defaults it ON; `run_train_droid.sh` OFF).

Caveat: the 0.87 capacity number used each clip's *own inverted* z_init; a
*shared* ε₀ puts more burden on `C_t` (only the i2v pin supplies `I_0`).
The single-clip overfit (`run_overfit_droid.sh`) tests this directly — if a
fixed-noise overfit reconstructs the clip, feasibility is confirmed and the
sweep failure was the noise-marginalization burden + 289-clip mapping; if it
still `≈ null`, the bottleneck is per-step context capacity (try `L>1`,
shorter horizon).

---

## 4. Open questions

1. **How much does I_0 actually matter for this dataset?**
   With more time, an ablation training "actions-only" vs
   "actions + I_0" tells us. Spec. of the answer here is in 2.4 — we
   believe it matters but can be cheaply tested.

2. **Will the rank-1 form generalize to unseen actions?**
   Within-distribution we have evidence the structure is there. Out-of-
   distribution (e.g., faster/slower motion, different start poses) is
   not measured yet; expect a separate evaluation pass.

3. **Do we need `μ` and `β` to be learnable at all, or can we freeze
   them as constants from `adaptor_init.pt`?**
   Frozen → simpler, less expressive. Learnable → more flexible, more
   parameters to overfit. The current default is learnable with a 10×
   smaller LR than the heads (so warm-start isn't torn apart in early
   steps). `--freeze_mu_beta_steps 10000` is the one-flag way to test
   the "frozen-then-unfreeze" variant.

4. **Should `b` be unit-normalized, or should we allow `b_head` outputs
   to grow freely?**
   Unit-norm + scalar `γ` matches the factorization. Free-norm `b`
   absorbs `γ`'s job and is equivalent up to redundancy. Unit-norm is
   cleaner and likely more stable; revisit only if training is unstable.

5. **Is `--t_sampling uniform` good enough, or should we switch to
   `logit_normal`?**
   Logit-normal would match flow-matching pretraining (and the table in
   the original §3.4). Uniform is the current default for interpretability
   of step-by-step loss curves. Plan: confirm M0/M1 with uniform, then
   re-run M2 with `--t_sampling logit_normal` and compare validation loss.
