# Embedding Search Study for Wan 2.2 TI2V

## Research question

Given a frozen Wan 2.2 TI2V-5B model and a pair `(I_0, I_T*)` (start frame, desired goal frame), does there exist a text-embedding input `e` such that the generated video transitions from `I_0` to `I_T*`? And how close does a heuristic natural-language caption come to the best achievable `e`?

This is a **capacity / controllability** study, not a prompt-recovery study — we do **not** have ground-truth captions. Each optimized `e` is per-triplet (textual inversion setting).

## Connection to existing literature

Closest analogs in LLM / diffusion prompt-optimization literature:
- **Textual Inversion** (Gal et al. 2022) — optimize a few token embeddings to represent a concept.
- **Prompt Tuning / Prefix Tuning** (Lester 2021; Li & Liang 2021) — continuous soft prompts for frozen LLMs.
- **Null-Text Inversion** (Mokady et al. 2022) — per-timestep embedding optimization for DDIM inversion.
- **DOODL** (Wallace et al. 2023) — backprop through sampling to optimize noise; same engineering as ours but we optimize `e` instead of noise.
- **PEZ** (Wen et al. 2023) — projects learned soft embeddings to nearest real tokens; the bridge between our optimized `e_opt` and a readable prompt.
- **SDS / DDS** (DreamFusion; Hertz 2023) — gradient approximation that avoids backprop through sampling; fallback if full backprop OOMs.
- **GCG / AutoPrompt** (Zou 2023; Shin 2020) — discrete prompt search via gradient signals.
- **OPRO / APE / DSPy** — black-box LLM-as-optimizer; useful as a non-gradient baseline for comparison.

## Architecture targets (Wan 2.2 TI2V-5B)

Relevant entry points:

| Component | File:Line | Shape / fact |
|---|---|---|
| Pipeline | `wan/textimage2video.py:34` (`WanTI2V`) | Top-level inference class |
| Main inference | `wan/textimage2video.py:413` (`i2v`) | Wraps T5, DiT sampling loop, VAE |
| Text encoder | `wan/modules/t5.py:472` (UMT5-XXL) | Output: list of `[L_i, 4096]` |
| DiT | `wan/modules/model.py:294` (`WanModel`) | 30 blocks, dim=3072, text_dim=4096, text MLP at `model.py:380` |
| VAE | `wan/modules/vae2_2.py:888` (`Wan2_2_VAE`) | z_dim=48, stride (4,16,16) |
| Scheduler | `wan/utils/fm_solvers_unipc.py:657` (`step`) | Flow matching, deterministic |
| First-frame clamp | `wan/textimage2video.py:598` | `latent = (1-mask2)*z[0] + mask2*latent` |

Key facts that enable embedding search:
- Text embeddings are a `List[Tensor[L, 4096]]` passed as `context` to `WanModel.forward`. We can replace them with a learnable tensor without touching the DiT.
- Scheduler step is deterministic (flow matching) → graph flows cleanly through all sampling steps.
- First-frame clamp is differentiable (mask multiply).

## Multi-phase plan

### Phase 1 — Differentiable wrapper (MVP plumbing)

**Goal**: make the TI2V sampling loop support gradient flow from the final decoded latent back to a learnable `context` tensor.

**Changes**:
1. Add `WanTI2V.i2v_diff(...)` that mirrors `i2v` but:
   - Accepts precomputed `context` and `context_null` tensors (`[L, 4096]`) instead of strings.
   - Drops `torch.no_grad()` around the sampling loop.
   - Keeps `torch.amp.autocast(dtype=param_dtype)` for forward efficiency.
   - Keeps `vae.encode(I_0)` behind `torch.no_grad()` (image latent is fixed).
   - Returns the final **latent** (not just decoded video) and optionally the decoded video.
   - Accepts `frame_num` (short-horizon generations, e.g. 17 frames).
   - Optional `use_gradient_checkpointing: bool` — wraps each DiT block's forward in `torch.utils.checkpoint.checkpoint` during the sampling loop, restored on exit.
2. Do **not** modify `WanModel.forward` or the scheduler — all changes scoped to `textimage2video.py`.

**Exit criterion**: smoke test (`scripts/embedding_search_smoke.py`) shows non-zero `context.grad` after `.backward()` from a scalar loss on the final latent.

### Phase 2 — Optimization loop

`scripts/embedding_search.py`:
- Inputs: `I_0`, `I_T*`, heuristic seed caption, `L_opt` (number of learnable tokens), `sampling_steps` (e.g. 8), `frames` (e.g. 17).
- Pre-compute `z_T* = VAE.encode(I_T*)` (detached); reference `e_heur = T5(heuristic)` (detached, logging only).
- Initialize `e ∈ ℝ^{L_opt × 4096}` as an `nn.Parameter` from three settings (ablation): `e_heur`, `T5("")`, Gaussian matched to T5 output stats.
- Per-step loss (log all three, optimize one):
  - latent MSE on last frame latent vs `z_T*`
  - LPIPS on decoded last frame
  - 1 − SSIM on decoded last frame
- Optimizer: Adam, lr ~1e-2, 200–500 steps.
- Log: train loss curves, intermediate decoded frames every N steps.

### Phase 3 — Evaluation

For each `(I_0, heuristic, I_T*)` in dataset:
- **Task recovery**: LPIPS(decoded_last_frame, I_T*) with `e_opt` vs with `e_heur` (at final 40-step inference, not search-time 8-step).
- **Capacity ceiling**: DDIM inversion of `I_T*` to bound the best-possible reconstruction from the model, separating "model can't" from "search can't find it."
- **Outcome matrix** (2×2 on heuristic-loss × opt-loss): maps the dataset into cells that characterize the model's controllability without needing reference prompts:

| opt_loss ↓ \ heur_loss → | low | high |
|---|---|---|
| **low** | heuristic already good | model can do transition, heuristic bad (interesting positive) |
| **high** | optimization bug | model lacks capacity (or search failed) |

- **Hard-prompt decode (PEZ-style, optional)**: nearest-UMT5-token projection of `e_opt` for interpretability.
- **Transfer (optional)**: does `e_opt` work on a new `I_0'` close to `I_0`?

## Engineering notes / risks

- **Memory**: on H200 (141GB), full backprop through 8 sampling steps of the 5B DiT should fit with gradient checkpointing on blocks. If OOM → lower `frames`, reduce `L_opt`, or fall back to SDS (Strategy C in original design).
- **VAE decoder in graph**: only required for pixel-space losses. For latent-MSE first, skip it entirely.
- **Scheduler state**: `FlowUniPCMultistepScheduler` stores `last_sample`, `model_outputs` across calls. These hold tensors with grad history — need a fresh scheduler per optimization step (already the case if we re-`set_timesteps` each step).
- **CFG cost**: classifier-free guidance doubles DiT forward passes. For search, consider setting `guide_scale=1.0` to halve cost; re-enable for final validation.

## File map

- `wan/textimage2video.py` — `i2v_diff` method (Phase 1).
- `wan/modules/vae2_2.py` — `clamp_` → `clamp` in `Wan2_2_VAE.decode` (1-line autograd safety fix).
- `scripts/embedding_search_smoke.py` — Phase 1 smoke test (deps-free; tiny model + stub VAE).
- `scripts/embedding_search_losses.py` — SSIM (no extra dep) and lazy LPIPS.
- `scripts/embedding_search.py` — Phase 2 per-triplet optimization script.
- `docs/embedding_search_plan.md` — this document.

## Phase 2 usage

```bash
python scripts/embedding_search.py \
    --start_frame  data/triplets/001/I0.png \
    --goal_frame   data/triplets/001/IT.png \
    --ckpt_dir     /path/to/Wan2.2-TI2V-5B \
    --output_dir   runs/triplet_001 \
    --heuristic    "the cup is moved to the right edge of the table" \
    --init         heuristic \
    --loss         lpips \
    --num_iters    300 \
    --frames       17 \
    --sampling_steps 8 \
    --guide_scale  1.0 \
    --max_area     230400 \
    --validate
```

Output layout:

```
runs/triplet_001/
  config.json                 # all CLI args
  I_0.png, I_T_target.png     # cropped/resized inputs (sanity check)
  embedding_init.pt           # {e: Tensor[L_opt, 4096], init, heuristic}
  embedding_final.pt          # final optimized embedding + meta
  loss_log.csv                # per-step: loss, latent_mse, ssim, lpips, |g|, t
  step_00000_decoded.png ...  # last-frame previews every --log_every steps
  final_decoded.png           # last frame at full sampling_steps_val (with --validate)
  final_video.mp4             # full validation video (with --validate)
  heuristic_decoded.png       # baseline last frame at full settings, if heuristic given
  heuristic_video.mp4         # baseline video
```

The validation block prints the Δ(opt − heuristic) on SSIM and latent-MSE — the
key quantity for the 2×2 outcome matrix in Phase 3.

### Study constraints (enforced)

These three properties are baked into the script (`run_search` prints a banner
confirming them at startup):

1. **Initial noise is fixed.** `i2v_diff` constructs `torch.Generator(...)
   .manual_seed(args.seed)` every call with a constant `--seed`, regenerating
   identical noise; `torch.manual_seed` and `torch.cuda.manual_seed_all` also
   pin global PRNGs. Only `context` (the text embedding) varies between
   iterations.
2. **The text embedding is shared across every layer and every step.** The
   learnable parameter is a single `[L_opt, 4096]` tensor. Inside
   `WanModel.forward` it is projected once (4096 → 3072 via the model's
   `text_embedding` MLP) and that result is passed *unchanged* to all 30 DiT
   blocks; the same `arg_c['context']` reference is reused at every UniPC
   sampling step. No per-layer or per-timestep parameters exist.
3. **Classifier-free guidance is disabled.** Both `--guide_scale` and
   `--guide_scale_val` default to 1.0; non-1.0 values raise an error in
   `parse_args`. With CFG=1.0, `i2v_diff` short-circuits the unconditional DiT
   forward, so `context_null` is never consumed (a placeholder zero tensor
   replaces the T5(neg-prompt) encoding to skip wasted T5 work).

### H200 starting recipe

For a 480p-ish video at 17 frames with 8 search steps:

| Knob | Value | Reason |
|---|---|---|
| `--frames` | 17 | shortest 4n+1 horizon that still has a "non-input" final frame |
| `--sampling_steps` | 8 | UniPC quality knee for diffusion search |
| `--guide_scale` | 1.0 | CFG disabled (study constraint #3) — single DiT call per step |
| `--max_area` | 230400 | 480²; smaller = faster iterations |
| `--L_opt` | 16 | textual-inversion-scale prompt budget |
| `--lr` | 1e-2 | embedding-space scale is large; tune in [1e-3, 5e-2] |
| `--num_iters` | 300 | typical convergence point; watch loss curve |
| `--validate` | on | regenerates at 40 steps (still no CFG) for final eval |

If OOM: drop `--max_area` to ~150000, lower `--frames` to 13 (only 4 latent
slots), or fall back to `--loss latent_mse` (skips VAE decoder in the graph).

## Phase 3 sketch (next)

A driver script that loops over a triplets manifest, runs `embedding_search.py`
per triplet, then aggregates the per-triplet `loss_log.csv` and
`opt vs heuristic` deltas into the 2×2 outcome matrix described above. Not yet
implemented.
