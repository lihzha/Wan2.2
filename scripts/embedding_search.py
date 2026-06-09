"""Per-triplet text-embedding search for Wan 2.2 TI2V.

Given a start frame I_0 and a goal frame I_T*, this script supports three
search modes for steering the TI2V pipeline from I_0 toward I_T*:

  --mode embed          Optimize a learnable positive text embedding e
                        ∈ R^{L_opt × 4096}. Loss on the last decoded frame.

  --mode noise          Optimize the initial latent noise with a frozen
                        heuristic prompt. Capacity test for whether the
                        target is on the i2v manifold for this seed.

  --mode null_inversion Null-text inversion (Mokady et al. 2022, arXiv:2211.09794),
                        adapted to Wan's flow-matching sampler. (1) Build a
                        target video latent from I_0 / I_T*, (2) run a
                        reverse-Euler trajectory at w=1 with the heuristic
                        prompt to obtain a per-timestep pivot trajectory
                        {z*_t}, (3) optimize a per-timestep null embedding
                        ∅_t at the deployment CFG scale so the CFG-guided
                        forward pass tracks {z*_t}. The model and the
                        positive text embedding are kept frozen.

Example (null-text inversion against a real reference video — capacity study):
  python scripts/embedding_search.py \
      --mode null_inversion \
      --reference_video examples/clip.mp4 \
      --start_index 0 --delta_index 32 \
      --ckpt_dir /path/to/Wan2.2-TI2V-5B \
      --heuristic "the cup moves to the right" \
      --output_dir runs/triplet_001 \
      --sampling_steps 25 --null_inner_iters 10 --guide_scale 5.0

Capacity-study notes:
  * The script inverts the slice video[start_index : start_index + delta_index + 1].
    delta_index must be a positive multiple of 4 (Wan VAE has temporal stride 4;
    clips must be 4n+1 frames long). Wan TI2V-5B is trained at frame_num=121
    sample_fps=24, so delta_index=120 is the design point. Smaller values save
    compute at the cost of pushing the model out of distribution.
  * Outputs (each as MP4 + last-frame PNG):
      reference_clip.mp4   — the real input slice
      vae_roundtrip.mp4    — decode(encode(reference)). UPPER BOUND: the DiT
                             cannot output anything sharper than the VAE allows.
      ddim_inv_only.mp4    — Euler replay from the inversion endpoint with
                             frozen ∅ = T5(""). LOWER BOUND: inversion only,
                             no null-text optimization.
      reconstruction.mp4   — Euler replay with optimized per-step null
                             embeddings. The null-text inversion result.
  * Sampler note: inversion + replay use Euler-style flow-matching steps;
    Wan deploys UniPC multistep. Both inversion and replay use the SAME
    Euler sampler so they're internally self-consistent. Use
    --sampling_steps >= 25 so Euler well-approximates the underlying ODE.
    The optimized {∅_t} are tied to this Euler sampler — they will not
    transfer 1:1 to UniPC at deployment.

Hardware: assumes a single ~80GB+ GPU (H100/H200) for the full 5B model.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for losses module

# Import wan via the real package path; full pipeline expects all deps
# (flash_attn, librosa, etc.) to be installed.
import embedding_search_losses as L  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

from wan.configs.wan_ti2v_5B import ti2v_5B  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402
from wan.utils.utils import best_output_size  # noqa: E402

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reference_video",
        required=True,
        help="Path to the real reference video (mp4/etc). The script reads "
        "the slice video[start_index : start_index + delta_index + 1]; "
        "the start frame is I_0, the end frame is I_T*. For null_inversion "
        "the full slice is the inversion target.",
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Index of the first frame to use as I_0.",
    )
    p.add_argument(
        "--delta_index",
        type=int,
        default=16,
        help="Frame gap between I_0 and I_T*. Total clip length is "
        "delta_index + 1, which must be 4n + 1 (Wan VAE has temporal "
        "stride 4), so delta_index must be a positive multiple of 4. "
        "Useful values: 16 (17-frame clip, ~0.7s @ 24fps), 32 (33), 64 "
        "(65), 120 (121, the TI2V-5B training design point — most "
        "on-distribution but most expensive).",
    )
    p.add_argument("--ckpt_dir", required=True, help="Wan 2.2 TI2V-5B checkpoint dir.")
    p.add_argument("--output_dir", required=True)

    p.add_argument(
        "--heuristic",
        default="",
        help="Seed prompt; required for --mode noise; ignored in --mode embed "
        "unless --init heuristic.",
    )
    p.add_argument("--init", choices=["heuristic", "empty", "random"], default="empty")
    p.add_argument("--L_opt", type=int, default=16, help="Number of learnable tokens.")
    p.add_argument(
        "--mode",
        choices=["embed", "noise", "null_inversion", "positive_inversion"],
        default="embed",
        help="What to optimize. 'embed': learn the positive text embedding "
        "(capacity test). 'noise': learn the initial latent noise with a "
        "frozen heuristic prompt (control: if noise reaches I_T* but embed "
        "doesn't, the model can express the transition and text "
        "conditioning is the bottleneck; if neither does, I_T* is off the "
        "i2v manifold for this seed/horizon). 'null_inversion': Mokady "
        "et al. 2022 — DDIM-style trajectory inversion at w=1 with the "
        "heuristic prompt, then per-timestep null-text optimization at the "
        "deployment CFG scale. 'positive_inversion': role swap — same "
        "DDIM-style w=1 inversion, but the per-timestep optimization runs "
        "on the POSITIVE context C_t (warm-started from T5(heuristic)) "
        "while ∅ stays frozen at T5(\"\"). Tests whether per-step *positive* "
        "embeddings can reconstruct the reference; gradients flow through "
        "v_cond directly so it works at any CFG ≥ 1 (default 1.0 is fine).",
    )

    p.add_argument("--loss", choices=["latent_mse", "lpips", "ssim"], default="lpips")
    p.add_argument("--num_iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)

    # Null-text inversion knobs (used only when --mode null_inversion).
    p.add_argument(
        "--null_inner_iters",
        type=int,
        default=10,
        help="N in Algorithm 1 of Mokady et al. — inner Adam iterations "
        "spent optimizing ∅_t at each diffusion timestep.",
    )
    p.add_argument(
        "--null_lr",
        type=float,
        default=1e-2,
        help="Adam learning rate for the per-timestep null embedding.",
    )
    p.add_argument(
        "--L_pos",
        type=int,
        default=0,
        help="positive_inversion only: force the per-timestep positive "
        "embedding to have exactly L_pos tokens. 0 (default) = use the "
        "natural length of T5(heuristic). Setting --L_pos 1 minimizes "
        "the adaptor's b_head output dimensionality (one 4096-vector per "
        "video). Truncates if heuristic tokenizes to more than L_pos; "
        "zero-pads if fewer.",
    )
    p.add_argument(
        "--inversion_guide_scale",
        type=float,
        default=1.0,
        help="CFG scale during the DDIM-inversion pass. The paper uses 1.0 "
        "(no CFG) because higher guidance amplifies inversion error and "
        "drives the trajectory off the Gaussian manifold.",
    )

    p.add_argument(
        "--sampling_steps",
        type=int,
        default=25,
        help="Euler steps for inversion + replay (and the deployed-sampler "
        "search loop in --mode embed/noise). For null_inversion the script "
        "uses Euler internally while Wan deploys UniPC; >= 25 steps lets "
        "Euler approximate the underlying ODE well.",
    )
    p.add_argument(
        "--guide_scale",
        type=float,
        default=1.0,
        help="CFG scale during search. Default 1.0 (CFG off) keeps the search "
        "clean: only the conditional pass affects the gradient. CFG > 1 "
        "amplifies steering at deployment but mixes in a fixed "
        "unconditional context during the search; gradients still only "
        "flow through the conditional pass.",
    )
    p.add_argument(
        "--max_area",
        type=int,
        default=480 * 480,
        help="Latent-resolution budget. Lower = faster, less memory.",
    )
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed initial-noise seed across opt steps (essential for stable gradients).",
    )

    p.add_argument(
        "--log_every",
        type=int,
        default=25,
        help="Decode + save last frame every N opt steps.",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="After search, regenerate at full sampling_steps_val and save.",
    )
    p.add_argument("--sampling_steps_val", type=int, default=40)
    p.add_argument(
        "--guide_scale_val",
        type=float,
        default=1.0,
        help="CFG scale at validation. Default 1.0 (matches search). Set >1 "
        "to see how much CFG amplifies the optimized embedding/noise.",
    )

    args = p.parse_args()

    if args.delta_index < 4 or args.delta_index % 4 != 0:
        raise SystemExit(
            f"--delta_index must be a positive multiple of 4 (so the clip "
            f"has 4n+1 frames; Wan VAE temporal stride is 4). Got "
            f"{args.delta_index}."
        )
    args.frames = args.delta_index + 1  # auto-derived from delta_index

    if args.frames < 33:
        print(
            f"[search] NOTE: --delta_index={args.delta_index} -> clip length "
            f"{args.frames} ({args.frames / 24:.2f}s @ 24fps). Wan TI2V-5B "
            f"is trained at frame_num=121; very short clips push the model "
            f"out of distribution and may distort the capacity result."
        )

    if args.mode in ("null_inversion", "positive_inversion") and args.sampling_steps < 25:
        print(
            f"[search] WARNING: --sampling_steps={args.sampling_steps} is "
            f"low for {args.mode}. Inversion + replay use Euler steps; "
            f"Wan deploys UniPC multistep. Use >=25 steps so the Euler "
            f"discretization closely tracks the underlying ODE."
        )

    if args.mode == "noise" and not args.heuristic:
        raise SystemExit(
            "--mode noise requires --heuristic '<text>' to provide a fixed "
            "context for the search."
        )

    if args.mode in ("null_inversion", "positive_inversion") and not args.heuristic:
        raise SystemExit(
            f"--mode {args.mode} requires --heuristic '<text>' as the source "
            "caption (used to build the w=1 inversion pivot, and as the "
            "warm-start for positive_inversion's per-step C_t)."
        )

    if args.mode == "null_inversion" and args.guide_scale <= 1.0 + 1e-6:
        print(
            f"[search] WARNING: --mode null_inversion with --guide_scale="
            f"{args.guide_scale} ≤ 1. Null-text optimization only has "
            f"signal when CFG > 1; consider --guide_scale 5.0 (paper used "
            f"7.5 for Stable Diffusion)."
        )

    if args.mode not in ("null_inversion", "positive_inversion") and (
        abs(args.guide_scale - 1.0) > 1e-6 or abs(args.guide_scale_val - 1.0) > 1e-6
    ):
        print(
            f"[search] WARNING: CFG != 1.0 (search={args.guide_scale}, "
            f"val={args.guide_scale_val}). The unconditional pass will use a "
            f"zero-tensor placeholder as context_null, which may not match "
            f"the model's training distribution for empty prompts."
        )
    return args


# ---------------------------------------------------------------------------
# Image preprocessing — match the resize/crop the pipeline applies to I_0
# ---------------------------------------------------------------------------


def fit_image_to_pipeline(img: Image.Image, vae_stride, patch_size, max_area):
    """Mirror the resize+center-crop logic in WanTI2V.i2v so the goal frame
    is in the same pixel grid as the model's output."""
    ih, iw = img.height, img.width
    dh = patch_size[1] * vae_stride[1]
    dw = patch_size[2] * vae_stride[2]
    ow, oh = best_output_size(iw, ih, dw, dh, max_area)
    scale = max(ow / iw, oh / ih)
    img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (img.width - ow) // 2
    y1 = (img.height - oh) // 2
    img = img.crop((x1, y1, x1 + ow, y1 + oh))
    assert img.width == ow and img.height == oh
    return img, (oh, ow)


def to_tensor_signed(img: Image.Image, device) -> torch.Tensor:
    """PIL → tensor in [-1, 1], shape [3, H, W]."""
    t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
    return t


def load_reference_clip(path: str, start_index: int, delta_index: int):
    """Read `delta_index + 1` consecutive frames starting at `start_index`
    from a real reference video. Returns a list of PIL.Image (RGB)."""
    n_frames = delta_index + 1
    import decord
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(path)
    if start_index < 0 or start_index + n_frames > len(vr):
        raise SystemExit(
            f"Reference video {path!r} has {len(vr)} frames; cannot read "
            f"frames [{start_index}, {start_index + n_frames}). Pick a "
            f"smaller --start_index or --delta_index."
        )
    indices = list(range(start_index, start_index + n_frames))
    frames_np = vr.get_batch(indices).asnumpy()  # [F, H, W, 3] uint8
    return [Image.fromarray(f).convert("RGB") for f in frames_np]


def fit_clip_to_pipeline(frames_pil, vae_stride, patch_size, max_area):
    """Apply the same fit_image_to_pipeline transform to every frame, sized
    by the first frame so spatial geometry is consistent across the clip."""
    f0_fit, (oh, ow) = fit_image_to_pipeline(
        frames_pil[0], vae_stride, patch_size, max_area)
    fitted = [f0_fit]
    for f in frames_pil[1:]:
        ih, iw = f.height, f.width
        scale = max(ow / iw, oh / ih)
        f = f.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
        x1 = (f.width - ow) // 2
        y1 = (f.height - oh) // 2
        f = f.crop((x1, y1, x1 + ow, y1 + oh))
        assert f.width == ow and f.height == oh
        fitted.append(f)
    return fitted, (oh, ow)


def clip_to_tensor_signed(frames_pil, device) -> torch.Tensor:
    """List of PIL frames → [3, F, H, W] tensor in [-1, 1] on `device`."""
    return torch.stack(
        [TF.to_tensor(f).sub_(0.5).div_(0.5) for f in frames_pil], dim=1
    ).to(device)


# ---------------------------------------------------------------------------
# Embedding initialization
# ---------------------------------------------------------------------------


def init_context(
    pipe: WanTI2V, init_mode: str, heuristic: str, L_opt: int
) -> torch.Tensor:
    """Return a [L_opt, 4096] initial context tensor on the pipe's device.
    `heuristic` is encoded with T5 when init=heuristic; otherwise ignored."""
    device = pipe.device
    if init_mode == "heuristic":
        prompt = heuristic if heuristic else ""
        e = pipe.encode_prompt(prompt)  # [L_h, 4096], L_h depends on tokens
    elif init_mode == "empty":
        e = pipe.encode_prompt("")
    elif init_mode == "random":
        # Match magnitude of T5 outputs by sampling a reference encoding first.
        ref = pipe.encode_prompt("")
        std = ref.std().item()
        e = torch.randn(L_opt, 4096, device=device) * std
        return e.detach().clone()
    else:
        raise ValueError(init_mode)

    # Truncate / pad to L_opt.
    if e.shape[0] >= L_opt:
        e = e[:L_opt]
    else:
        pad = torch.zeros(L_opt - e.shape[0], e.shape[1], device=device, dtype=e.dtype)
        e = torch.cat([e, pad], dim=0)
    return e.detach().clone()


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


@dataclass
class LossInputs:
    pipe: WanTI2V
    z_target: torch.Tensor  # [C_z, H_z, W_z] — encoded I_T* (single frame slot)
    img_target_pixels: torch.Tensor  # [3, H, W] in [-1, 1]


def compute_loss(out: dict, kind: str, li: LossInputs) -> tuple[torch.Tensor, dict]:
    """Returns (loss_to_optimize, dict_of_logged_metrics).

    `out` is the dict returned by i2v_diff. For pixel losses this expects
    `decode_video=True, return_last_frame_only=True` so out['last_frame']
    is a [3, H, W] tensor with grad."""
    metrics: dict = {}
    pred_latent_last = out["latent_last_frame"]  # [C_z, H_z, W_z]

    # Always log latent MSE (cheap, no decode required when called as detach).
    with torch.no_grad():
        metrics["latent_mse"] = F.mse_loss(pred_latent_last, li.z_target).item()

    if kind == "latent_mse":
        loss = F.mse_loss(pred_latent_last, li.z_target)
        return loss, metrics

    # Pixel-space losses use the decoded last frame.
    pred_pixels = out["last_frame"]  # [3, H, W], in [-1, 1]
    with torch.no_grad():
        metrics["ssim"] = L.ssim(pred_pixels, li.img_target_pixels).item()
    if kind == "ssim":
        loss = L.ssim_loss(pred_pixels, li.img_target_pixels)
        return loss, metrics
    if kind == "lpips":
        loss = L.lpips_loss(pred_pixels, li.img_target_pixels)
        metrics["lpips"] = loss.item()
        return loss, metrics

    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Null-text inversion (Mokady et al. 2022, arXiv:2211.09794)
# Adapted from DDIM/Stable-Diffusion to Wan's flow-matching sampler.
#
# Notation map (paper → here):
#   ψ(P)         T5 encoding of the source caption (--heuristic). Frozen.
#   ∅            T5("") — initial null embedding. Optimized per-timestep.
#   z*_t         pivot latents from a w=1 reverse-Euler trajectory.
#   ¯z_t         the null-text-optimized trajectory; tracks z*_t at deploy CFG.
#
# Flow-matching specifics:
#   The deployed sampler iterates sigmas[0]→sigmas[N]=0 with Euler-like steps
#   z_{i+1} = z_i + (σ_{i+1} − σ_i) · v_θ(z_i, σ_i, C).
#   Inversion runs the same recurrence in reverse (clean → noise). Since the
#   model is OOD at σ=0, we approximate the very last inversion step using
#   v_θ(z*_0, σ_{N−1}, C) — the standard small-step DDIM-inversion trick.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _model_grad_checkpointing(pipe):
    """Wrap each DiT block in torch.utils.checkpoint for the duration of
    the with-block. Mirrors the pattern in WanTI2V.i2v_diff."""
    original = []
    for block in pipe.model.blocks:
        original.append(block.forward)

        def make_ckpt(orig_forward):
            def ckpt_forward(x, **kwargs):
                return checkpoint_utils.checkpoint(
                    orig_forward, x, use_reentrant=False, **kwargs)
            return ckpt_forward

        block.forward = make_ckpt(block.forward)
    try:
        yield
    finally:
        for block, orig in zip(pipe.model.blocks, original):
            block.forward = orig


def _format_timestep(sigma_value, num_train_timesteps, mask2_zero, seq_len, device):
    """Build the [1, seq_len] timestep tensor the DiT expects, replicating
    WanTI2V.i2v_diff's temp_ts construction. mask2_zero is mask2[0][0]
    (shape [F_z, H_z, W_z]); positions where mask2 = 0 (i.e. frame 0 of
    the i2v conditioning) get timestep 0, the rest get sigma * T_train."""
    t_scalar = torch.tensor(
        [sigma_value * num_train_timesteps], dtype=torch.float32, device=device)
    temp_ts = (mask2_zero[:, ::2, ::2] * t_scalar).flatten()
    temp_ts = torch.cat([
        temp_ts,
        temp_ts.new_ones(seq_len - temp_ts.size(0)) * t_scalar,
    ])
    return temp_ts.unsqueeze(0)


def _dit_velocity(pipe, latent, sigma_value, context_list, seq_len,
                  mask2_zero, param_dtype):
    """Single DiT forward returning the predicted flow velocity v at
    (latent, sigma). Output is cast back to float32 so subsequent latent
    arithmetic stays in fp32 even though the model runs under autocast."""
    timestep = _format_timestep(
        sigma_value, pipe.num_train_timesteps, mask2_zero, seq_len, pipe.device)
    with torch.amp.autocast('cuda', dtype=param_dtype):
        v = pipe.model([latent], t=timestep, context=context_list,
                       seq_len=seq_len)[0]
    return v.float()


def _apply_first_frame_pin(latent, z_I0_full, mask2_full):
    """Mirror i2v's `latent = (1 − mask2) · z_I0 + mask2 · latent` so the
    first frame of every trajectory step holds the encoded I_0."""
    return (1. - mask2_full) * z_I0_full + mask2_full * latent


def compute_inversion_trajectory(
    pipe,
    z_target_video,    # [C_z, F_z, H_z, W_z], clean target latent (= z*_0 / pivot end)
    z_I0_full,         # [C_z, F_z, H_z, W_z], encoded I_0 broadcast across frames
    mask2_full,        # [C_z, F_z, H_z, W_z], 0 at frame 0, 1 elsewhere
    seq_len,
    sigmas,            # 1D tensor, length N+1, going sigma_max → 0
    context_list,      # list with one [L, 4096] tensor — T5(heuristic)
    inversion_guide_scale,
    context_null_list, # list with [L, 4096] T5("") — only used if CFG > 1
    param_dtype,
):
    """Reverse-Euler trajectory clean → noise.

    The deployed sampler iterates sampler step i = 0..N−1, transitioning
    sigmas[i] → sigmas[i+1] using v_θ(·, σ_i, C). Inversion fills `traj`
    so that traj[i] is the latent at sigma=σ_i:

        traj[N] = z_target_video                             (clean pivot)
        for k = 0..N−1:
            i = N−1−k                                        (descending i)
            v = v_θ(traj[i+1], σ=σ_i, C[, ∅])               (DDIM-inversion approx)
            traj[i] = traj[i+1] + (σ_i − σ_{i+1}) · v

    Returns: list of length N+1 of detached float32 latents."""
    N = len(sigmas) - 1
    traj = [None] * (N + 1)
    traj[N] = _apply_first_frame_pin(
        z_target_video.detach().float(), z_I0_full, mask2_full)

    use_cfg = abs(inversion_guide_scale - 1.0) > 1e-6
    mask2_zero = mask2_full[0]  # [F_z, H_z, W_z] — per-patch mask used in temp_ts

    print(f"[null-inv] DDIM-style inversion: N={N} steps, w={inversion_guide_scale}")
    with torch.no_grad():
        for k in range(N):
            i = N - 1 - k                # the slot we're filling
            sigma_i = sigmas[i].item()
            sigma_ip1 = sigmas[i + 1].item()
            v = _dit_velocity(
                pipe, traj[i + 1], sigma_i, context_list, seq_len,
                mask2_zero, param_dtype)
            if use_cfg:
                v_null = _dit_velocity(
                    pipe, traj[i + 1], sigma_i, context_null_list, seq_len,
                    mask2_zero, param_dtype)
                v = v_null + inversion_guide_scale * (v - v_null)
            step = (sigma_i - sigma_ip1) * v
            traj[i] = _apply_first_frame_pin(
                traj[i + 1] + step, z_I0_full, mask2_full).detach()
    return traj


def optimize_null_embeddings(
    pipe,
    traj,                  # list[N+1] of pivot latents from inversion
    z_I0_full,
    mask2_full,
    seq_len,
    sigmas,
    context_list,          # T5(heuristic), frozen
    null_init,             # T5("") — [L_null, 4096]
    inner_iters,
    null_lr,
    guide_scale,           # deployment CFG (paper: 7.5; we default 5.0 in CLI)
    param_dtype,
    log_dir,
    log_every=None,
):
    """Algorithm 1 of Mokady et al. — per-timestep null-text optimization.

    Outer loop over diffusion steps i = 0..N−1 (sampling direction, σ
    decreasing). At each step we hold ¯z_i fixed and run `inner_iters`
    of Adam on ∅_i to minimize ||¯z_{i+1}(¯z_i, ∅_i, C) − z*_{i+1}||².
    After convergence we update ¯z_{i+1} via one more forward (no grad)
    and warm-start ∅_{i+1} ← ∅_i.

    Returns: list[N] of optimized null embeddings (one per sampling step),
    and the final ¯z trajectory (list[N+1])."""
    N = len(sigmas) - 1
    mask2_zero = mask2_full[0]

    null_embeds = []   # one per sampling step i = 0..N-1
    z_bar = [None] * (N + 1)
    z_bar[0] = traj[0].clone().detach()
    z_bar[0] = _apply_first_frame_pin(z_bar[0], z_I0_full, mask2_full)

    log_path = os.path.join(log_dir, "null_inv_loss_log.csv")
    with open(log_path, "w") as f:
        f.write("step_i,inner_j,sigma,loss,target_l2,wall_s\n")

    null_emb_param = null_init.detach().clone().float().requires_grad_(True)
    print(
        f"[null-inv] null-text optimization: N={N} steps × {inner_iters} "
        f"inner iters, w={guide_scale}, null_init shape={tuple(null_init.shape)}"
    )
    t_start = time.time()

    with _model_grad_checkpointing(pipe):
        for i in range(N):
            sigma_i = sigmas[i].item()
            sigma_ip1 = sigmas[i + 1].item()
            dsigma = sigma_ip1 - sigma_i  # negative (denoising direction)

            # Cache the conditional velocity once: C and z_bar[i] are fixed
            # across the inner loop, so v_cond doesn't depend on ∅_i.
            with torch.no_grad():
                v_cond = _dit_velocity(
                    pipe, z_bar[i], sigma_i, context_list, seq_len,
                    mask2_zero, param_dtype)

            optim = torch.optim.Adam([null_emb_param], lr=null_lr)
            target = traj[i + 1].detach()
            target_l2 = target.norm().item()

            for j in range(inner_iters):
                v_uncond = _dit_velocity(
                    pipe, z_bar[i], sigma_i, [null_emb_param], seq_len,
                    mask2_zero, param_dtype)
                v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
                z_pred = z_bar[i] + dsigma * v_cfg
                z_pred = _apply_first_frame_pin(z_pred, z_I0_full, mask2_full)

                loss = F.mse_loss(z_pred, target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                if log_every is None or (j % log_every == 0) or j == inner_iters - 1:
                    with open(log_path, "a") as f:
                        f.write(
                            f"{i},{j},{sigma_i:.6f},{loss.item():.6e},"
                            f"{target_l2:.4f},{time.time() - t_start:.1f}\n"
                        )

            # Lock in this step's ∅_i and roll the trajectory forward.
            null_embeds.append(null_emb_param.detach().clone())
            with torch.no_grad():
                v_uncond = _dit_velocity(
                    pipe, z_bar[i], sigma_i, [null_emb_param], seq_len,
                    mask2_zero, param_dtype)
                v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
                z_next = z_bar[i] + dsigma * v_cfg
                z_bar[i + 1] = _apply_first_frame_pin(
                    z_next, z_I0_full, mask2_full).detach()

            with torch.no_grad():
                final_inner = F.mse_loss(z_bar[i + 1], target).item()
                pivot_drift = (z_bar[i + 1] - target).norm().item()
            print(
                f"[null-inv] step={i:3d}/{N} σ={sigma_i:.4f}→{sigma_ip1:.4f} "
                f"final_loss={final_inner:.4e} pivot_L2={pivot_drift:.4f} "
                f"t={time.time() - t_start:6.1f}s"
            )
            torch.cuda.empty_cache()

    return null_embeds, z_bar


def optimize_positive_embeddings(
    pipe,
    traj,                  # list[N+1] of pivot latents from inversion
    z_I0_full,
    mask2_full,
    seq_len,
    sigmas,
    null_context_list,     # T5(""), frozen
    positive_init,         # T5(heuristic) — [L_pos, 4096], warm start
    inner_iters,
    pos_lr,
    guide_scale,           # deployment CFG; pos-inv works at any w >= 1
    param_dtype,
    log_dir,
    log_every=None,
):
    """Per-timestep POSITIVE-text optimization (role swap of Mokady's null-text).

    Outer loop over diffusion steps i = 0..N−1 (sampling direction, σ
    decreasing). At each step we hold ¯z_i fixed and run `inner_iters`
    of Adam on C_i to minimize ||¯z_{i+1}(¯z_i, ∅, C_i) − z*_{i+1}||².
    ∅ is frozen at T5(""). C_i warm-starts from T5(heuristic) at i=0,
    then from C_{i-1} thereafter (single tensor reused across outer steps).

    Asymmetry with null-text inversion: gradients flow through v_cond,
    not v_uncond, so v_uncond is what we cache once per outer step.

    Returns: list[N] of optimized positive embeddings (one per sampling
    step), and the final ¯z trajectory (list[N+1])."""
    N = len(sigmas) - 1
    mask2_zero = mask2_full[0]

    pos_embeds = []   # one per sampling step i = 0..N-1
    z_bar = [None] * (N + 1)
    z_bar[0] = traj[0].clone().detach()
    z_bar[0] = _apply_first_frame_pin(z_bar[0], z_I0_full, mask2_full)

    log_path = os.path.join(log_dir, "pos_inv_loss_log.csv")
    with open(log_path, "w") as f:
        f.write("step_i,inner_j,sigma,loss,target_l2,wall_s\n")

    pos_emb_param = positive_init.detach().clone().float().requires_grad_(True)
    print(
        f"[pos-inv] positive-text optimization: N={N} steps × {inner_iters} "
        f"inner iters, w={guide_scale}, positive_init shape="
        f"{tuple(positive_init.shape)}"
    )
    t_start = time.time()

    with _model_grad_checkpointing(pipe):
        for i in range(N):
            sigma_i = sigmas[i].item()
            sigma_ip1 = sigmas[i + 1].item()
            dsigma = sigma_ip1 - sigma_i  # negative (denoising direction)

            # Cache the unconditional velocity once: ∅ and z_bar[i] are fixed
            # across the inner loop, so v_uncond doesn't depend on C_i.
            with torch.no_grad():
                v_uncond = _dit_velocity(
                    pipe, z_bar[i], sigma_i, null_context_list, seq_len,
                    mask2_zero, param_dtype)

            optim = torch.optim.Adam([pos_emb_param], lr=pos_lr)
            target = traj[i + 1].detach()
            target_l2 = target.norm().item()

            for j in range(inner_iters):
                v_cond = _dit_velocity(
                    pipe, z_bar[i], sigma_i, [pos_emb_param], seq_len,
                    mask2_zero, param_dtype)
                v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
                z_pred = z_bar[i] + dsigma * v_cfg
                z_pred = _apply_first_frame_pin(z_pred, z_I0_full, mask2_full)

                loss = F.mse_loss(z_pred, target)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                if log_every is None or (j % log_every == 0) or j == inner_iters - 1:
                    with open(log_path, "a") as f:
                        f.write(
                            f"{i},{j},{sigma_i:.6f},{loss.item():.6e},"
                            f"{target_l2:.4f},{time.time() - t_start:.1f}\n"
                        )

            # Lock in this step's C_i and roll the trajectory forward.
            pos_embeds.append(pos_emb_param.detach().clone())
            with torch.no_grad():
                v_cond = _dit_velocity(
                    pipe, z_bar[i], sigma_i, [pos_emb_param], seq_len,
                    mask2_zero, param_dtype)
                v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
                z_next = z_bar[i] + dsigma * v_cfg
                z_bar[i + 1] = _apply_first_frame_pin(
                    z_next, z_I0_full, mask2_full).detach()

            with torch.no_grad():
                final_inner = F.mse_loss(z_bar[i + 1], target).item()
                pivot_drift = (z_bar[i + 1] - target).norm().item()
            print(
                f"[pos-inv] step={i:3d}/{N} σ={sigma_i:.4f}→{sigma_ip1:.4f} "
                f"final_loss={final_inner:.4e} pivot_L2={pivot_drift:.4f} "
                f"t={time.time() - t_start:6.1f}s"
            )
            torch.cuda.empty_cache()

    return pos_embeds, z_bar


def regenerate_with_null_embeds(
    pipe,
    z_init,                # the inversion endpoint = traj[0]
    z_I0_full,
    mask2_full,
    seq_len,
    sigmas,
    context_list,          # T5(heuristic)
    null_embeds,           # list[N] of optimized ∅_i
    guide_scale,
    param_dtype,
):
    """Replay the deployed CFG-guided sampler from z_init using {∅_t}.
    Returns the final clean latent (last entry of the trajectory)."""
    N = len(sigmas) - 1
    mask2_zero = mask2_full[0]
    z = _apply_first_frame_pin(z_init.detach().float(), z_I0_full, mask2_full)
    with torch.no_grad():
        for i in range(N):
            sigma_i = sigmas[i].item()
            sigma_ip1 = sigmas[i + 1].item()
            v_cond = _dit_velocity(
                pipe, z, sigma_i, context_list, seq_len, mask2_zero, param_dtype)
            v_uncond = _dit_velocity(
                pipe, z, sigma_i, [null_embeds[i]], seq_len, mask2_zero, param_dtype)
            v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
            z = _apply_first_frame_pin(
                z + (sigma_ip1 - sigma_i) * v_cfg, z_I0_full, mask2_full)
    return z


def regenerate_with_positive_embeds(
    pipe,
    z_init,                # the inversion endpoint = traj[0]
    z_I0_full,
    mask2_full,
    seq_len,
    sigmas,
    null_context_list,     # T5(""), frozen
    positive_embeds,       # list[N] of optimized C_i
    guide_scale,
    param_dtype,
):
    """Replay the deployed CFG-guided sampler from z_init using per-step
    optimized POSITIVE embeddings {C_t}; ∅ is frozen at T5(""). Returns
    the final clean latent."""
    N = len(sigmas) - 1
    mask2_zero = mask2_full[0]
    z = _apply_first_frame_pin(z_init.detach().float(), z_I0_full, mask2_full)
    with torch.no_grad():
        for i in range(N):
            sigma_i = sigmas[i].item()
            sigma_ip1 = sigmas[i + 1].item()
            v_cond = _dit_velocity(
                pipe, z, sigma_i, [positive_embeds[i]], seq_len,
                mask2_zero, param_dtype)
            v_uncond = _dit_velocity(
                pipe, z, sigma_i, null_context_list, seq_len,
                mask2_zero, param_dtype)
            v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
            z = _apply_first_frame_pin(
                z + (sigma_ip1 - sigma_i) * v_cfg, z_I0_full, mask2_full)
    return z


def _build_inversion_geometry(pipe, oh, ow, n_frames, sampling_steps, shift, device):
    """Compute (sigmas, mask2, seq_len) shared across inversion + opt loops."""
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=pipe.num_train_timesteps,
        shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
    sigmas = scheduler.sigmas.to(device).float()  # length N+1

    F_z = (n_frames - 1) // pipe.vae_stride[0] + 1
    H_z = oh // pipe.vae_stride[1]
    W_z = ow // pipe.vae_stride[2]
    z_dim = pipe.vae.model.z_dim

    seq_len = (F_z * H_z * W_z) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int((seq_len + pipe.sp_size - 1) // pipe.sp_size) * pipe.sp_size

    # Build the i2v mask the same way masks_like(zero=True, generator=None) does.
    template = torch.empty(z_dim, F_z, H_z, W_z, device=device, dtype=torch.float32)
    mask2 = torch.ones_like(template)
    mask2[:, 0] = 0.0
    return sigmas, mask2, seq_len, (z_dim, F_z, H_z, W_z)


def run_null_inversion(
    args, pipe, img0_pil, imgT_pil, imgT_pixels, target_clip_pixels, oh, ow,
):
    """Top-level driver for --mode null_inversion against a real reference clip.
    Outputs (all in args.output_dir):
       - vae_roundtrip.mp4       : decode(encode(reference)). UPPER BOUND.
       - ddim_inv_only.mp4       : Euler replay from inversion endpoint with
                                   frozen ∅ = T5(""). LOWER BOUND.
       - reconstruction.mp4      : Euler replay with optimized {∅_t}.
       - inversion_trajectory.pt : pivot latents from the w=1 inversion.
       - null_embeddings.pt      : optimized ∅_t (one per sampling step).
       - heuristic_video.mp4     : (with --validate) UniPC heuristic at the
                                   deployed sampler from random noise — the
                                   deployment-fidelity baseline."""
    device = pipe.device

    # ---- T5 encodings (heuristic = positive C, "" = null init) ----
    pipe.text_encoder.model.to(device)
    with torch.no_grad():
        context_pos = pipe.encode_prompt(args.heuristic).detach().float()
        null_init = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    print(
        f"[null-inv] T5(heuristic) shape={tuple(context_pos.shape)}; "
        f"T5('') shape={tuple(null_init.shape)}"
    )

    # ---- Geometry (sigmas, masks, seq_len) ----
    sigmas, mask2_full, seq_len, (z_dim, F_z, H_z, W_z) = _build_inversion_geometry(
        pipe, oh, ow, args.frames, args.sampling_steps, args.shift, device)
    print(
        f"[null-inv] latent geom z=[{z_dim}, {F_z}, {H_z}, {W_z}] seq_len={seq_len}; "
        f"sigmas: {sigmas[0].item():.4f} → {sigmas[-1].item():.4f} (N={len(sigmas) - 1})"
    )

    # ---- Encode I_0 and the real reference clip ----
    img0_pixels = to_tensor_signed(img0_pil, device)
    print(
        f"[null-inv] real reference clip: shape={tuple(target_clip_pixels.shape)}"
    )

    with torch.no_grad():
        z_I0_singleframe = pipe.vae.encode([img0_pixels.unsqueeze(1)])[0]  # [C_z, 1, H_z, W_z]
        z_I0_full = z_I0_singleframe.expand(-1, F_z, -1, -1).contiguous().float()
        z_target_video = pipe.vae.encode([target_clip_pixels])[0].float()  # [C_z, F_z, H_z, W_z]
    print(
        f"[null-inv] z_I0 shape={tuple(z_I0_full.shape)}, "
        f"z_target_video shape={tuple(z_target_video.shape)}"
    )

    # ---- VAE round-trip ceiling (upper bound on what any DiT output can match) ----
    print("[null-inv] VAE round-trip ceiling: decode(encode(reference))…")
    with torch.no_grad():
        vae_recon = pipe.vae.decode([z_target_video])[0]  # [3, F, H, W]
    save_video(vae_recon, os.path.join(args.output_dir, "vae_roundtrip.mp4"))
    save_frame(
        vae_recon[:, -1],
        os.path.join(args.output_dir, "vae_roundtrip_last_frame.png"),
    )
    with torch.no_grad():
        ssim_per_frame_vae = torch.tensor([
            L.ssim(vae_recon[:, k], target_clip_pixels[:, k]).item()
            for k in range(target_clip_pixels.shape[1])
        ])
        vae_ssim_avg = ssim_per_frame_vae.mean().item()
        vae_ssim_last = ssim_per_frame_vae[-1].item()
    print(
        f"[null-inv] VAE-CEILING: ssim_avg={vae_ssim_avg:.4f} "
        f"ssim_last={vae_ssim_last:.4f}"
    )

    # ---- DDIM-style inversion at w=1 ----
    pipe.model.to(device)
    traj = compute_inversion_trajectory(
        pipe=pipe,
        z_target_video=z_target_video,
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        context_list=[context_pos],
        inversion_guide_scale=args.inversion_guide_scale,
        context_null_list=[null_init],
        param_dtype=pipe.param_dtype,
    )

    # Sanity-log the inversion endpoint statistics: should be ≈ N(0, I) for a
    # clean inversion (matching paper Fig. 9a). Far from 1 → inversion is off
    # the Gaussian manifold and the capacity claim weakens.
    with torch.no_grad():
        z0 = traj[0]
        z0_mean = z0.mean().item()
        z0_std = z0.std().item()
    print(
        f"[null-inv] Inversion endpoint stats: mean={z0_mean:+.3e} "
        f"std={z0_std:.3f} (Gaussian → 0, 1)"
    )

    torch.save(
        {
            "sigmas": sigmas.detach().cpu(),
            "trajectory": [t.detach().cpu() for t in traj],
            "heuristic": args.heuristic,
            "inversion_guide_scale": args.inversion_guide_scale,
            "start_index": args.start_index,
            "delta_index": args.delta_index,
        },
        os.path.join(args.output_dir, "inversion_trajectory.pt"),
    )

    # ---- Lower bound: Euler replay from inversion endpoint with frozen ∅ ----
    # Mirrors the paper's "DDIM inversion" baseline (Fig. 4): inversion only,
    # no null-text opt. If the optimized null-text inversion below doesn't beat
    # this, the optimization is contributing nothing.
    print("[null-inv] Lower bound: Euler replay from inversion endpoint, frozen ∅=T5(\"\")…")
    n_steps = len(sigmas) - 1
    z_ddim_only = regenerate_with_null_embeds(
        pipe=pipe,
        z_init=traj[0],
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        context_list=[context_pos],
        null_embeds=[null_init] * n_steps,
        guide_scale=args.guide_scale,
        param_dtype=pipe.param_dtype,
    )
    with torch.no_grad():
        video_ddim_only = pipe.vae.decode([z_ddim_only])[0]
    save_video(
        video_ddim_only,
        os.path.join(args.output_dir, "ddim_inv_only.mp4"),
    )
    save_frame(
        video_ddim_only[:, -1],
        os.path.join(args.output_dir, "ddim_inv_only_last_frame.png"),
    )
    with torch.no_grad():
        ssim_per_frame_ddim = torch.tensor([
            L.ssim(video_ddim_only[:, k], target_clip_pixels[:, k]).item()
            for k in range(target_clip_pixels.shape[1])
        ])
        ddim_ssim_avg = ssim_per_frame_ddim.mean().item()
        ddim_ssim_last = ssim_per_frame_ddim[-1].item()
    print(
        f"[null-inv] DDIM-INV-ONLY: ssim_avg={ddim_ssim_avg:.4f} "
        f"ssim_last={ddim_ssim_last:.4f}"
    )

    # ---- Null-text optimization ----
    null_embeds, z_bar_traj = optimize_null_embeddings(
        pipe=pipe,
        traj=traj,
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        context_list=[context_pos],
        null_init=null_init,
        inner_iters=args.null_inner_iters,
        null_lr=args.null_lr,
        guide_scale=args.guide_scale,
        param_dtype=pipe.param_dtype,
        log_dir=args.output_dir,
    )

    torch.save(
        {
            "null_embeddings": [e.detach().cpu() for e in null_embeds],
            "context_pos": context_pos.detach().cpu(),
            "sigmas": sigmas.detach().cpu(),
            "guide_scale": args.guide_scale,
            "heuristic": args.heuristic,
            "z_init": traj[0].detach().cpu(),
        },
        os.path.join(args.output_dir, "null_embeddings.pt"),
    )

    # ---- Reconstruction: replay deployed sampler with optimized {∅_t} ----
    print("[null-inv] Replaying deployed CFG sampler with optimized null embeddings…")
    z_recon = regenerate_with_null_embeds(
        pipe=pipe,
        z_init=traj[0],
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        context_list=[context_pos],
        null_embeds=null_embeds,
        guide_scale=args.guide_scale,
        param_dtype=pipe.param_dtype,
    )
    with torch.no_grad():
        video_recon = pipe.vae.decode([z_recon])[0]
    save_video(
        video_recon, os.path.join(args.output_dir, "reconstruction.mp4"))
    save_frame(
        video_recon[:, -1],
        os.path.join(args.output_dir, "reconstruction_last_frame.png"),
    )

    with torch.no_grad():
        ssim_per_frame_recon = torch.tensor([
            L.ssim(video_recon[:, k], target_clip_pixels[:, k]).item()
            for k in range(target_clip_pixels.shape[1])
        ])
        recon_ssim_avg = ssim_per_frame_recon.mean().item()
        recon_ssim_last = ssim_per_frame_recon[-1].item()
        last_latent_mse = F.mse_loss(z_recon[:, -1], z_target_video[:, -1]).item()
        full_latent_mse = F.mse_loss(z_recon, z_target_video).item()
    print(
        f"[null-inv] RECON: ssim_avg={recon_ssim_avg:.4f} "
        f"ssim_last={recon_ssim_last:.4f} "
        f"latent_mse_last={last_latent_mse:.5f} "
        f"latent_mse_full={full_latent_mse:.5f}"
    )
    print(
        f"[null-inv] CAPACITY-STUDY SUMMARY (ssim_avg over all {target_clip_pixels.shape[1]} frames):\n"
        f"             VAE-CEILING (upper):   {vae_ssim_avg:.4f}\n"
        f"             NULL-TEXT-INV result:  {recon_ssim_avg:.4f}\n"
        f"             DDIM-INV-ONLY (lower): {ddim_ssim_avg:.4f}\n"
        f"           Closer to VAE-CEILING => Wan can express this clip;\n"
        f"           closer to DDIM-INV-ONLY => null-text opt is not helping."
    )

    # ---- Heuristic baseline (no null-text opt) for comparison ----
    if args.validate:
        print("[null-inv] Heuristic baseline at deployment CFG (no null-text opt)…")
        with torch.no_grad():
            seed_g = torch.Generator(device=device)
            seed_g.manual_seed(int(args.seed))
            noise_baseline = torch.randn(
                z_dim, F_z, H_z, W_z, generator=seed_g, device=device)
            out_h = pipe.i2v_diff(
                img=img0_pil,
                context=context_pos,
                context_null=null_init,
                max_area=args.max_area,
                frame_num=args.frames,
                shift=args.shift,
                sampling_steps=args.sampling_steps_val,
                guide_scale=args.guide_scale_val,
                seed=args.seed,
                use_gradient_checkpointing=False,
                decode_video=True,
                return_last_frame_only=False,
                noise=noise_baseline,
            )
            video_h = out_h["video"]
            heur_ssim = L.ssim(video_h[:, -1], imgT_pixels).item()
            heur_latent_mse = F.mse_loss(out_h["latent_last_frame"], z_target_video[:, -1]).item()
        save_video(
            video_h, os.path.join(args.output_dir, "heuristic_video.mp4"))
        save_frame(
            video_h[:, -1],
            os.path.join(args.output_dir, "heuristic_last_frame.png"),
        )
        print(
            f"[null-inv] HEURISTIC (UniPC, random noise, deployment-fidelity): "
            f"ssim_last={heur_ssim:.4f} latent_mse_last={heur_latent_mse:.5f}"
        )
        print(
            f"[null-inv] Δ(null-inv − heuristic, last frame): "
            f"ssim={recon_ssim_last - heur_ssim:+.4f} "
            f"latent_mse={last_latent_mse - heur_latent_mse:+.5f}"
        )


def run_positive_inversion(
    args, pipe, img0_pil, imgT_pil, imgT_pixels, target_clip_pixels, oh, ow,
):
    """Top-level driver for --mode positive_inversion. Mirrors
    run_null_inversion exactly except the per-step optimization and the
    final reconstruction replay operate on positive context C_t (warm
    started from T5(heuristic)) with ∅ frozen at T5("").

    Outputs (all in args.output_dir):
       - vae_roundtrip.mp4         : decode(encode(reference)). UPPER BOUND.
       - ddim_inv_only.mp4         : Euler replay from inversion endpoint
                                     with frozen heuristic C and frozen ∅.
                                     LOWER BOUND.
       - reconstruction.mp4        : Euler replay with optimized {C_t}.
       - inversion_trajectory.pt   : pivot latents from the w=1 inversion.
       - positive_embeddings.pt    : optimized C_t (one per sampling step)."""
    device = pipe.device

    # ---- T5 encodings ----
    pipe.text_encoder.model.to(device)
    with torch.no_grad():
        context_pos = pipe.encode_prompt(args.heuristic).detach().float()
        null_init = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    print(
        f"[pos-inv] T5(heuristic) shape={tuple(context_pos.shape)}; "
        f"T5('') shape={tuple(null_init.shape)}"
    )

    # Optionally force the positive-init to have exactly L_pos tokens.
    if getattr(args, "L_pos", 0) and args.L_pos > 0:
        L_target = int(args.L_pos)
        L_cur = context_pos.shape[0]
        if L_cur > L_target:
            context_pos = context_pos[:L_target].contiguous()
        elif L_cur < L_target:
            pad = torch.zeros(
                L_target - L_cur, context_pos.shape[1],
                device=context_pos.device, dtype=context_pos.dtype,
            )
            context_pos = torch.cat([context_pos, pad], dim=0).contiguous()
        print(
            f"[pos-inv] forcing positive init to L_pos={L_target} -> "
            f"shape={tuple(context_pos.shape)}"
        )

    # ---- Geometry ----
    sigmas, mask2_full, seq_len, (z_dim, F_z, H_z, W_z) = _build_inversion_geometry(
        pipe, oh, ow, args.frames, args.sampling_steps, args.shift, device)
    print(
        f"[pos-inv] latent geom z=[{z_dim}, {F_z}, {H_z}, {W_z}] seq_len={seq_len}; "
        f"sigmas: {sigmas[0].item():.4f} → {sigmas[-1].item():.4f} (N={len(sigmas) - 1})"
    )

    # ---- Encode I_0 and the real reference clip ----
    img0_pixels = to_tensor_signed(img0_pil, device)
    print(
        f"[pos-inv] real reference clip: shape={tuple(target_clip_pixels.shape)}"
    )
    with torch.no_grad():
        z_I0_singleframe = pipe.vae.encode([img0_pixels.unsqueeze(1)])[0]
        z_I0_full = z_I0_singleframe.expand(-1, F_z, -1, -1).contiguous().float()
        z_target_video = pipe.vae.encode([target_clip_pixels])[0].float()
    print(
        f"[pos-inv] z_I0 shape={tuple(z_I0_full.shape)}, "
        f"z_target_video shape={tuple(z_target_video.shape)}"
    )

    # ---- VAE round-trip ceiling ----
    print("[pos-inv] VAE round-trip ceiling: decode(encode(reference))…")
    with torch.no_grad():
        vae_recon = pipe.vae.decode([z_target_video])[0]
    save_video(vae_recon, os.path.join(args.output_dir, "vae_roundtrip.mp4"))
    save_frame(
        vae_recon[:, -1],
        os.path.join(args.output_dir, "vae_roundtrip_last_frame.png"),
    )
    with torch.no_grad():
        ssim_per_frame_vae = torch.tensor([
            L.ssim(vae_recon[:, k], target_clip_pixels[:, k]).item()
            for k in range(target_clip_pixels.shape[1])
        ])
        vae_ssim_avg = ssim_per_frame_vae.mean().item()
        vae_ssim_last = ssim_per_frame_vae[-1].item()
    print(
        f"[pos-inv] VAE-CEILING: ssim_avg={vae_ssim_avg:.4f} "
        f"ssim_last={vae_ssim_last:.4f}"
    )

    # ---- DDIM-style inversion at w=1 (with heuristic positive, like null-inv) ----
    pipe.model.to(device)
    traj = compute_inversion_trajectory(
        pipe=pipe,
        z_target_video=z_target_video,
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        context_list=[context_pos],
        inversion_guide_scale=args.inversion_guide_scale,
        context_null_list=[null_init],
        param_dtype=pipe.param_dtype,
    )

    with torch.no_grad():
        z0 = traj[0]
        z0_mean = z0.mean().item()
        z0_std = z0.std().item()
    print(
        f"[pos-inv] Inversion endpoint stats: mean={z0_mean:+.3e} "
        f"std={z0_std:.3f} (Gaussian → 0, 1)"
    )

    torch.save(
        {
            "sigmas": sigmas.detach().cpu(),
            "trajectory": [t.detach().cpu() for t in traj],
            "heuristic": args.heuristic,
            "inversion_guide_scale": args.inversion_guide_scale,
            "start_index": args.start_index,
            "delta_index": args.delta_index,
        },
        os.path.join(args.output_dir, "inversion_trajectory.pt"),
    )

    # ---- Lower bound: Euler replay with frozen heuristic C and frozen ∅ ----
    # Identical to null_inversion's ddim_inv_only.mp4 — they're the same
    # baseline because both use the heuristic C with no per-step optimization.
    print("[pos-inv] Lower bound: Euler replay with frozen heuristic C and ∅=T5(\"\")…")
    n_steps = len(sigmas) - 1
    z_ddim_only = regenerate_with_null_embeds(
        pipe=pipe,
        z_init=traj[0],
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        context_list=[context_pos],
        null_embeds=[null_init] * n_steps,
        guide_scale=args.guide_scale,
        param_dtype=pipe.param_dtype,
    )
    with torch.no_grad():
        video_ddim_only = pipe.vae.decode([z_ddim_only])[0]
    save_video(
        video_ddim_only,
        os.path.join(args.output_dir, "ddim_inv_only.mp4"),
    )
    save_frame(
        video_ddim_only[:, -1],
        os.path.join(args.output_dir, "ddim_inv_only_last_frame.png"),
    )
    with torch.no_grad():
        ssim_per_frame_ddim = torch.tensor([
            L.ssim(video_ddim_only[:, k], target_clip_pixels[:, k]).item()
            for k in range(target_clip_pixels.shape[1])
        ])
        ddim_ssim_avg = ssim_per_frame_ddim.mean().item()
        ddim_ssim_last = ssim_per_frame_ddim[-1].item()
    print(
        f"[pos-inv] DDIM-INV-ONLY: ssim_avg={ddim_ssim_avg:.4f} "
        f"ssim_last={ddim_ssim_last:.4f}"
    )

    # ---- Per-step POSITIVE optimization (the role-swapped piece) ----
    pos_embeds, _ = optimize_positive_embeddings(
        pipe=pipe,
        traj=traj,
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        null_context_list=[null_init],
        positive_init=context_pos,
        inner_iters=args.null_inner_iters,
        pos_lr=args.null_lr,
        guide_scale=args.guide_scale,
        param_dtype=pipe.param_dtype,
        log_dir=args.output_dir,
    )

    torch.save(
        {
            "positive_embeddings": [e.detach().cpu() for e in pos_embeds],
            "null_init": null_init.detach().cpu(),
            "sigmas": sigmas.detach().cpu(),
            "guide_scale": args.guide_scale,
            "heuristic": args.heuristic,
            "z_init": traj[0].detach().cpu(),
        },
        os.path.join(args.output_dir, "positive_embeddings.pt"),
    )

    # ---- Reconstruction: replay with optimized {C_t}, frozen ∅ ----
    print("[pos-inv] Replaying CFG sampler with optimized positive embeddings…")
    z_recon = regenerate_with_positive_embeds(
        pipe=pipe,
        z_init=traj[0],
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        null_context_list=[null_init],
        positive_embeds=pos_embeds,
        guide_scale=args.guide_scale,
        param_dtype=pipe.param_dtype,
    )
    with torch.no_grad():
        video_recon = pipe.vae.decode([z_recon])[0]
    save_video(
        video_recon, os.path.join(args.output_dir, "reconstruction.mp4"))
    save_frame(
        video_recon[:, -1],
        os.path.join(args.output_dir, "reconstruction_last_frame.png"),
    )

    with torch.no_grad():
        ssim_per_frame_recon = torch.tensor([
            L.ssim(video_recon[:, k], target_clip_pixels[:, k]).item()
            for k in range(target_clip_pixels.shape[1])
        ])
        recon_ssim_avg = ssim_per_frame_recon.mean().item()
        recon_ssim_last = ssim_per_frame_recon[-1].item()
        last_latent_mse = F.mse_loss(z_recon[:, -1], z_target_video[:, -1]).item()
        full_latent_mse = F.mse_loss(z_recon, z_target_video).item()
    print(
        f"[pos-inv] RECON: ssim_avg={recon_ssim_avg:.4f} "
        f"ssim_last={recon_ssim_last:.4f} "
        f"latent_mse_last={last_latent_mse:.5f} "
        f"latent_mse_full={full_latent_mse:.5f}"
    )
    print(
        f"[pos-inv] CAPACITY-STUDY SUMMARY (ssim_avg over all {target_clip_pixels.shape[1]} frames):\n"
        f"             VAE-CEILING (upper):   {vae_ssim_avg:.4f}\n"
        f"             POS-TEXT-INV result:   {recon_ssim_avg:.4f}\n"
        f"             DDIM-INV-ONLY (lower): {ddim_ssim_avg:.4f}\n"
        f"           Closer to VAE-CEILING => per-step positive C_t can express this clip;\n"
        f"           closer to DDIM-INV-ONLY => positive-text opt is not helping."
    )


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------


def save_frame(t: torch.Tensor, path: str):
    """Save a [3, H, W] tensor in [-1, 1] as a PNG."""
    img = (t.detach().float().clamp(-1, 1) + 1) / 2
    img = (img * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(path)


def save_video(video: torch.Tensor, path: str, fps: int = 16):
    """Save a [3, F, H, W] tensor in [-1, 1] as MP4 if imageio is available;
    otherwise fall back to a frame strip PNG."""
    video = (video.detach().float().clamp(-1, 1) + 1) / 2
    video = (
        (video * 255).round().byte().permute(1, 2, 3, 0).cpu().numpy()
    )  # [F, H, W, 3]
    try:
        import imageio

        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as w:
            for frame in video:
                w.append_data(frame)
    except Exception as e:
        print(
            f"[search] imageio video write failed ({e}); writing frame strip PNG instead."
        )
        import numpy as np

        strip = np.concatenate(list(video), axis=1)  # stack horizontally
        Image.fromarray(strip).save(path.replace(".mp4", "_strip.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_pipeline(args) -> WanTI2V:
    """Construct the real Wan TI2V-5B pipeline from a checkpoint directory.
    Split out so a mock pipeline can be injected for validation."""
    print("[search] Loading WanTI2V-5B…")
    pipe = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    print(f"[search] Loaded. device={pipe.device}")
    return pipe


def run_search(args, pipe):
    """Run the full optimization pipeline against a pre-built `pipe`.
    Used by `main()` and by `scripts/validate_phase2_mock.py`."""
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "loss_log.csv")
    cfg_path = os.path.join(args.output_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    device = pipe.device

    # Seed all PRNGs so any incidental RNG draws beyond the explicit
    # initial-noise seed are also pinned across opt steps.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[search] mode={args.mode}")
    if args.mode == "embed":
        print(
            f"          - learnable: text embedding [L_opt={args.L_opt} x 4096], "
            "shared across all DiT layers AND all sampling steps"
        )
        print(
            f"          - initial noise: FIXED (seed={args.seed}, regenerated each step)"
        )
    elif args.mode == "noise":
        print(f"          - learnable: initial latent noise (seed-init at {args.seed})")
        print(f"          - text context: FROZEN heuristic = {args.heuristic!r}")
    elif args.mode == "null_inversion":
        print(
            f"          - learnable: per-timestep null embedding ∅_t "
            f"(N={args.sampling_steps} steps × {args.null_inner_iters} inner iters)"
        )
        print(f"          - source caption C: T5({args.heuristic!r}) (frozen)")
        print(
            f"          - inversion: w={args.inversion_guide_scale} "
            f"(real reference video as pivot)"
        )
    else:  # positive_inversion
        print(
            f"          - learnable: per-timestep POSITIVE context C_t "
            f"(N={args.sampling_steps} steps × {args.null_inner_iters} inner iters), "
            f"warm-started from T5({args.heuristic!r})"
        )
        print(f"          - null embedding ∅: T5(\"\") (frozen)")
        print(
            f"          - inversion: w={args.inversion_guide_scale} "
            f"(real reference video as pivot)"
        )
    print(f"          - classifier-free guidance: guide_scale={args.guide_scale}")

    # ---- Reference video clip ----
    clip_pil = load_reference_clip(
        args.reference_video, args.start_index, args.delta_index)
    clip_pil, (oh, ow) = fit_clip_to_pipeline(
        clip_pil, pipe.vae_stride, pipe.patch_size, args.max_area)
    img0_pil = clip_pil[0]
    imgT_pil = clip_pil[-1]
    print(
        f"[search] Loaded reference clip: {len(clip_pil)} frames "
        f"({args.reference_video!r}[{args.start_index}:"
        f"{args.start_index + args.delta_index + 1}]); resized to "
        f"(H, W) = ({oh}, {ow})."
    )
    img0_pil.save(os.path.join(args.output_dir, "I_0.png"))
    imgT_pil.save(os.path.join(args.output_dir, "I_T_target.png"))

    target_clip_pixels = clip_to_tensor_signed(clip_pil, device)  # [3, F, H, W]
    imgT_pixels = target_clip_pixels[:, -1]  # [3, H, W]
    save_video(
        target_clip_pixels,
        os.path.join(args.output_dir, "reference_clip.mp4"),
    )

    # Null-text inversion is structurally different (no Adam-on-embedding loop,
    # no per-iteration loss CSV in the same shape) — dispatch out before the
    # naive-search machinery sets up its z_T target / learnable variable.
    if args.mode == "null_inversion":
        run_null_inversion(
            args, pipe, img0_pil, imgT_pil, imgT_pixels,
            target_clip_pixels, oh, ow,
        )
        print(f"[search] Done. Outputs in {args.output_dir}")
        return

    if args.mode == "positive_inversion":
        run_positive_inversion(
            args, pipe, img0_pil, imgT_pil, imgT_pixels,
            target_clip_pixels, oh, ow,
        )
        print(f"[search] Done. Outputs in {args.output_dir}")
        return

    # Encode I_T* as a single-frame "video" → [z_dim, 1, H_lat, W_lat] → squeeze F.
    with torch.no_grad():
        imgT_t = imgT_pixels.unsqueeze(1)  # [3, 1, H, W]
        z_T = pipe.vae.encode([imgT_t])[0]  # [z_dim, 1, H_lat, W_lat]
        z_T = z_T.squeeze(1)  # [z_dim, H_lat, W_lat]

    # ---- Negative (unconditional) context ----
    # When CFG is on we want a real T5("") encoding so the unconditional pass
    # matches the model's training distribution; with CFG off, a zero
    # placeholder is fine (and the unconditional forward is short-circuited).
    use_cfg = abs(args.guide_scale - 1.0) > 1e-6 or abs(args.guide_scale_val - 1.0) > 1e-6
    if use_cfg:
        with torch.no_grad():
            context_null = pipe.encode_prompt("").detach()
    else:
        context_null = torch.zeros(1, 4096, device=device)

    # ---- Build the learnable variable + the (possibly fixed) context ----
    if args.mode == "embed":
        e_init = init_context(pipe, args.init, args.heuristic, args.L_opt)
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        print(
            f"[search] embed init={args.init}; shape={tuple(e_init.shape)} "
            f"mean={e_init.mean().item():+.3e} std={e_init.std().item():.3e}"
        )
        torch.save(
            {"e": e_init.detach().cpu(), "init": args.init, "heuristic": args.heuristic},
            os.path.join(args.output_dir, "embedding_init.pt"),
        )
        learnable = e_init.clone().detach().requires_grad_(True)
        context_for_pipe = learnable  # learnable text embedding
        noise_for_pipe = None  # let i2v_diff seed-generate the noise
    else:  # noise
        # Heuristic context, frozen. T5 only needed once.
        with torch.no_grad():
            context_for_pipe = pipe.encode_prompt(args.heuristic).detach()
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        # Initial noise from the seed: same starting point as embed mode.
        F_z = (args.frames - 1) // pipe.vae_stride[0] + 1
        H_z = oh // pipe.vae_stride[1]
        W_z = ow // pipe.vae_stride[2]
        z_dim = pipe.vae.model.z_dim
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(int(args.seed))
        noise_init = torch.randn(z_dim, F_z, H_z, W_z, generator=seed_g, device=device)
        print(
            f"[search] noise init: shape={tuple(noise_init.shape)} "
            f"(z_dim={z_dim}, F_z={F_z}, H={H_z}, W={W_z}); context = T5({args.heuristic!r})"
        )
        torch.save(
            {"noise": noise_init.detach().cpu(), "heuristic": args.heuristic},
            os.path.join(args.output_dir, "noise_init.pt"),
        )
        learnable = noise_init.clone().detach().requires_grad_(True)
        noise_for_pipe = learnable

    li = LossInputs(pipe=pipe, z_target=z_T, img_target_pixels=imgT_pixels)

    # ---- Optimizer ----
    optim = torch.optim.Adam([learnable], lr=args.lr)
    assert learnable.requires_grad

    decode_for_loss = args.loss in ("lpips", "ssim")

    # ---- CSV header ----
    metric_keys = [
        "step", "loss", "loss_type", "latent_mse", "ssim", "lpips",
        "param_norm", "grad_norm", "wall_s",
    ]
    with open(log_path, "w") as f:
        f.write(",".join(metric_keys) + "\n")

    # ---- Optimization loop ----
    print(
        f"[search] Starting Adam: iters={args.num_iters} lr={args.lr} loss={args.loss}"
    )
    t_start = time.time()
    for step in range(args.num_iters):
        # Periodically also decode for visualization, even when loss is latent.
        decode_now = (
            decode_for_loss
            or (step % args.log_every == 0)
            or (step == args.num_iters - 1)
        )

        out = pipe.i2v_diff(
            img=img0_pil,
            context=context_for_pipe,
            context_null=context_null,
            max_area=args.max_area,
            frame_num=args.frames,
            shift=args.shift,
            sample_solver="unipc",
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            use_gradient_checkpointing=True,
            decode_video=decode_now,
            return_last_frame_only=True,
            noise=noise_for_pipe,
        )

        loss, metrics = compute_loss(out, args.loss, li)

        optim.zero_grad(set_to_none=True)
        loss.backward()

        gnorm = learnable.grad.norm().item() if learnable.grad is not None else 0.0
        pnorm = learnable.detach().norm().item()
        optim.step()

        elapsed = time.time() - t_start
        row = {
            "step": step,
            "loss": float(loss.item()),
            "loss_type": args.loss,
            "latent_mse": metrics.get("latent_mse", float("nan")),
            "ssim": metrics.get("ssim", float("nan")),
            "lpips": metrics.get("lpips", float("nan")),
            "param_norm": pnorm,
            "grad_norm": gnorm,
            "wall_s": elapsed,
        }
        with open(log_path, "a") as f:
            f.write(",".join(str(row[k]) for k in metric_keys) + "\n")

        if step % args.log_every == 0 or step == args.num_iters - 1:
            print(
                f"[search] step={step:4d} loss={loss.item():.5f} "
                f"latent_mse={metrics['latent_mse']:.4f} "
                f"ssim={metrics.get('ssim', float('nan')):.4f} "
                f"lpips={metrics.get('lpips', float('nan')):.4f} "
                f"|p|={pnorm:.2e} |g|={gnorm:.2e} t={elapsed:6.1f}s"
            )
            if "last_frame" in out:
                save_frame(
                    out["last_frame"],
                    os.path.join(args.output_dir, f"step_{step:05d}_decoded.png"),
                )

        # Free memory before next iteration.
        del out, loss
        torch.cuda.empty_cache()

    # ---- Save final learnable ----
    final_blob = {
        "mode": args.mode,
        "heuristic": args.heuristic,
        "loss": args.loss,
        "num_iters": args.num_iters,
        "lr": args.lr,
    }
    if args.mode == "embed":
        final_blob["e"] = learnable.detach().cpu()
        final_blob["init"] = args.init
        torch.save(final_blob, os.path.join(args.output_dir, "embedding_final.pt"))
    else:
        final_blob["noise"] = learnable.detach().cpu()
        torch.save(final_blob, os.path.join(args.output_dir, "noise_final.pt"))

    # ---- Validation: regenerate at full quality with optimized variable ----
    if args.validate:
        opt_label = "opt-embedding" if args.mode == "embed" else "opt-noise"
        print(f"[search] Validating: full-quality regen with {opt_label}…")
        val_context = context_for_pipe.detach() if args.mode == "embed" else context_for_pipe
        val_noise = learnable.detach() if args.mode == "noise" else None
        with torch.no_grad():
            out = pipe.i2v_diff(
                img=img0_pil,
                context=val_context,
                context_null=context_null,
                max_area=args.max_area,
                frame_num=args.frames,
                shift=args.shift,
                sampling_steps=args.sampling_steps_val,
                guide_scale=args.guide_scale_val,
                seed=args.seed,
                use_gradient_checkpointing=False,
                decode_video=True,
                return_last_frame_only=False,
                noise=val_noise,
            )
        video = out["video"]  # [3, F, H, W]
        save_video(video, os.path.join(args.output_dir, "final_video.mp4"))
        save_frame(video[:, -1], os.path.join(args.output_dir, "final_decoded.png"))

        # Final loss with optimized variable at full quality.
        with torch.no_grad():
            final_pixel_ssim = L.ssim(video[:, -1], imgT_pixels).item()
            final_latent_mse = F.mse_loss(out["latent_last_frame"], z_T).item()
        print(
            f"[search] FINAL ({opt_label} @ {args.sampling_steps_val} steps, CFG={args.guide_scale_val}): "
            f"ssim={final_pixel_ssim:.4f} latent_mse={final_latent_mse:.5f}"
        )

        # ---- Heuristic baseline at full quality (for the 2x2 outcome matrix) ----
        # Always runs with seed-generated noise + heuristic prompt; this is the
        # reference point both modes are trying to beat.
        if args.heuristic:
            print(
                "[search] Heuristic baseline: same I_0 + heuristic prompt at full quality…"
            )
            with torch.no_grad():
                ctx_heur = pipe.encode_prompt(args.heuristic).detach()
                out_h = pipe.i2v_diff(
                    img=img0_pil,
                    context=ctx_heur,
                    context_null=context_null,
                    max_area=args.max_area,
                    frame_num=args.frames,
                    shift=args.shift,
                    sampling_steps=args.sampling_steps_val,
                    guide_scale=args.guide_scale_val,
                    seed=args.seed,
                    use_gradient_checkpointing=False,
                    decode_video=True,
                    return_last_frame_only=False,
                )
                video_h = out_h["video"]
                save_video(
                    video_h, os.path.join(args.output_dir, "heuristic_video.mp4")
                )
                save_frame(
                    video_h[:, -1],
                    os.path.join(args.output_dir, "heuristic_decoded.png"),
                )
                heur_ssim = L.ssim(video_h[:, -1], imgT_pixels).item()
                heur_latent_mse = F.mse_loss(out_h["latent_last_frame"], z_T).item()
            print(
                f"[search] HEURISTIC: ssim={heur_ssim:.4f} latent_mse={heur_latent_mse:.5f}"
            )
            print(
                f"[search] Δ(opt − heuristic): ssim={final_pixel_ssim - heur_ssim:+.4f} "
                f"latent_mse={final_latent_mse - heur_latent_mse:+.5f}"
            )

    print(f"[search] Done. Outputs in {args.output_dir}")


def main():
    args = parse_args()
    pipe = build_pipeline(args)
    run_search(args, pipe)


if __name__ == "__main__":
    main()
