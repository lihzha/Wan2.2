"""Sample videos from the trained trajectory adaptor and compare to ground truth.

For each evaluation triplet:
  1. Load actions, z_I0 (precomputed by scripts/precompute_features.py).
  2. Run the adaptor → C_traj [1, N, L, D].
  3. Replay Wan's CFG-guided Euler sampler with C_t from the adaptor as the
     positive context and T5("") as the null context. Frame 0 is pinned to
     z_I0 at every step (same i2v latent pin Wan was trained with).
  4. Decode → save `sample.mp4` and metrics.

Optional companion videos:
  - `ground_truth.mp4`         — VAE-decoded z_video (upper bound).
  - `oracle.mp4`               — replay with the saved per-step inversion
                                 embeddings (runs/batch_inv_positive/
                                 triplet_<i>/positive_embeddings.pt).
                                 The adaptor's reconstruction quality
                                 cannot exceed this.
  - `null_only.mp4`            — replay with T5("") in *both* slots, no
                                 CFG steering — generic motion baseline.

Sampler note: this runs Euler with the FlowUniPCMultistepScheduler sigmas
(same N+1 sigma sequence the adaptor was trained on), matching what
`embedding_search.py:regenerate_with_positive_embeds` does. The deployed
UniPC sampler would interpolate between these sigmas differently, so the
adaptor's σ-indexed warm-start no longer matches. Stay on Euler for
within-distribution evaluation; revisit UniPC only after M2 passes.

Usage:
  python scripts/eval_adaptor.py \\
      --triplets_root data/triplets \\
      --adaptor_init runs/batch_inv_positive/factorization/adaptor_init.pt \\
      --ckpt_path runs/adaptor_overfit_t0/ckpt_latest.pt \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --eval_names 0 \\
      --include_oracle --include_null \\
      --output_dir runs/eval_overfit_t0

Pass `--ckpt_path none` to evaluate the adaptor at *init* (the M0 check —
output trajectory should be the shared μ-only video).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import torch
import torch.nn.functional as F

import embedding_search as ES
import embedding_search_losses as L
from models.trajectory_adaptor import TrajectoryAdaptor
from models.z_init_predictor import ZInitPredictor
from wan.configs.wan_ti2v_5B import ti2v_5B
from wan.textimage2video import WanTI2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets_root", default="data/triplets")
    p.add_argument(
        "--adaptor_init",
        default="runs/batch_inv_positive/factorization/adaptor_init.pt",
        help="Used to instantiate the adaptor module before loading "
        "--ckpt_path. Required even when evaluating a trained checkpoint.",
    )
    p.add_argument(
        "--ckpt_path", default="none",
        help="Adaptor checkpoint produced by train_adaptor.py "
        "(ckpt_step<N>.pt / ckpt_latest.pt). Pass 'none' to evaluate at "
        "init — the M0 check (output ≈ μ).",
    )
    p.add_argument("--ckpt_dir", required=True,
                   help="Wan2.2 TI2V-5B checkpoint dir (DiT + T5 + VAE).")
    p.add_argument(
        "--eval_names", nargs="+", required=True,
        help="Triplet folder basenames to evaluate (the held-out / val set), "
        "e.g. '45 46 47 48 49'. Tagged split='val' in the summary.",
    )
    p.add_argument("--output_dir", required=True)

    # Optional: ALSO evaluate a handful of TRAINING episodes in the same run
    # (same loaded pipeline), so the fit-vs-generalization gap is visible in
    # one summary.csv (split column = train|val). See §3.6 Diagnostic A.
    p.add_argument(
        "--train_triplets_root", default=None,
        help="If set, also eval some training episodes from this dir.",
    )
    p.add_argument(
        "--train_eval_names", nargs="*", default=None,
        help="Explicit training episode names to eval. If omitted, the first "
        "--n_train_eval subdirs of --train_triplets_root are used.",
    )
    p.add_argument("--n_train_eval", type=int, default=5,
                   help="How many training episodes to eval when "
                   "--train_eval_names is not given.")

    # Sampler knobs (must match training so σ-indexing aligns with μ, β).
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--sigma_schedule_path", default=None,
                   help="Optional .pt containing a custom sigmas tensor "
                   "[N+1], or a dict with key 'sigmas'. Overrides "
                   "--sampling_steps/--shift for the actual rollout grid.")
    p.add_argument("--guide_scale", type=float, default=-1.0,
                   help="CFG scale for the Euler replay. Default -1 = AUTO: use "
                   "the checkpoint's train_guide_scale so eval matches how the "
                   "model was trained (the whole point — a mismatch breaks "
                   "reconstruction; see §3.13). Falls back to 5.0 if the ckpt "
                   "predates the flag / at init. Set a positive value to force.")
    p.add_argument("--seed", type=int, default=0,
                   help="Fixed initial-noise seed so different adaptors / "
                   "checkpoints can be A/B'd visually.")
    p.add_argument("--sample_z_init_path", default=None,
                   help="Optional .pt containing an exact [C,F,H,W] z_init "
                   "or a dict with z_init. If set, also replay the adaptor "
                   "from this tensor and save sample_fixed_zinit.mp4.")
    p.add_argument("--include_pred_zinit", action="store_true",
                   help="If the checkpoint contains a z_init predictor, also "
                   "render sample_pred_zinit.mp4 from its predicted initial latent.")
    p.add_argument("--max_area", type=int, default=230400,
                   help="Pixel-area budget. Must match precompute / inversion "
                   "so the precomputed latents land on the right grid.")
    p.add_argument(
        "--include_oracle", action="store_true",
        help="Also render with the saved per-step inversion embeddings "
        "(--oracle_root/{triplet_<name>,<name>}/positive_embeddings.pt). "
        "Upper bound for what the adaptor can match.",
    )
    p.add_argument(
        "--oracle_root", default="runs/batch_inv_positive",
        help="Root directory containing per-triplet positive_embeddings.pt. "
        "For DROID, use runs/droid_inv/train or runs/droid_inv/val.",
    )
    p.add_argument(
        "--include_adaptor_oracle_zinit", action="store_true",
        help="If an oracle positive_embeddings.pt with z_init is found, also "
        "render the adaptor contexts from that saved inversion initial latent. "
        "This diagnoses random-noise-basin mismatch separately from bad "
        "adaptor embeddings.",
    )
    p.add_argument(
        "--include_oracle_seed", action="store_true",
        help="If oracle embeddings are found, also replay them from eval's "
        "seeded random noise instead of the saved inversion z_init. This "
        "tests whether the optimized embeddings are tied to their z_init.",
    )
    p.add_argument(
        "--include_null", action="store_true",
        help="Also render with the null context T5('') in the positive slot "
        "(== unconditional generation, no CFG steering).",
    )
    p.add_argument(
        "--include_groundtruth", action="store_true", default=True,
        help="Save VAE.decode(z_video) as ground_truth.mp4. Upper bound.",
    )

    # Adaptor architecture — only used when --ckpt_path none (init / M0 mode);
    # when a checkpoint is given, its saved training args take precedence.
    p.add_argument("--action_dim", type=int, default=8,
                   help="Action vector width. DROID = 7, old triplets = 8. "
                   "Only consulted in init mode; a checkpoint overrides it.")
    p.add_argument("--action_len", type=int, default=32)
    p.add_argument("--action_repr", choices=["delta", "raw"], default="delta")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--arch_fusion", choices=["concat", "cross_attn"], default="concat")
    p.add_argument("--arch_head", choices=["rank1", "rankk", "perstep"], default="rank1")
    p.add_argument("--rank_k", type=int, default=4)
    p.add_argument("--n_xattn_layers", type=int, default=2)
    p.add_argument("--step_emb_dim", type=int, default=128)

    return p.parse_args()


def resolve_oracle_path(oracle_root: str, name: str):
    """Find positive_embeddings.pt for both old triplet_* and DROID layouts."""
    candidates = []
    if name.startswith("triplet_"):
        bare = name[len("triplet_"):]
        candidates.extend([
            os.path.join(oracle_root, name, "positive_embeddings.pt"),
            os.path.join(oracle_root, bare, "positive_embeddings.pt"),
        ])
    else:
        candidates.extend([
            os.path.join(oracle_root, f"triplet_{name}", "positive_embeddings.pt"),
            os.path.join(oracle_root, name, "positive_embeddings.pt"),
        ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_z_init_tensor(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, torch.Tensor):
        z = d.float()
    elif "z_init" in d:
        z = d["z_init"].float()
    else:
        raise KeyError(f"{path} has neither tensor content nor z_init")
    if z.dim() != 4:
        raise ValueError(f"expected z_init [C,F,H,W], got {tuple(z.shape)}")
    return z


def load_sigma_schedule(path: str, device) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    sigmas = d["sigmas"] if isinstance(d, dict) else d
    sigmas = sigmas.float().to(device)
    if sigmas.dim() != 1 or sigmas.numel() < 2:
        raise ValueError(f"expected sigmas [N+1], got {tuple(sigmas.shape)}")
    return sigmas


# ---------------------------------------------------------------------------
# Adaptor checkpoint loader
# ---------------------------------------------------------------------------


def load_adaptor(adaptor_init: str, ckpt_path: str, device, args):
    """Instantiate from adaptor_init.pt, optionally load a checkpoint.

    A checkpoint's saved training args define the architecture (action_dim
    etc.). In init mode (--ckpt_path none) there are no saved args, so the
    CLI --action_dim/--action_len/--action_repr/--hidden are used instead —
    these MUST match the dataset (DROID = 7, old triplets = 8).
    """
    if ckpt_path != "none":
        if not os.path.exists(ckpt_path):
            # Fail loudly: silently falling back to init produced confusing
            # results before (wrong action_dim, "M0 mode" surprise).
            raise FileNotFoundError(
                f"--ckpt_path {ckpt_path!r} not found. Pass 'none' to evaluate "
                f"the adaptor at init, or point to an existing checkpoint "
                f"(saved every --ckpt_interval steps)."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        train_args = ckpt.get("args", {})
        g = lambda k, d: train_args.get(k, d)   # saved args win; CLI is fallback
        adaptor = TrajectoryAdaptor(
            adaptor_init_path=adaptor_init,
            hidden=g("hidden", args.hidden),
            action_dim=g("action_dim", args.action_dim),
            action_len=g("action_len", args.action_len),
            action_repr=g("action_repr", args.action_repr),
            b_head_init_scale=g("b_head_init_scale", 1e-4),
            fusion=g("arch_fusion", args.arch_fusion),
            head=g("arch_head", args.arch_head),
            rank_k=g("rank_k", args.rank_k),
            n_xattn_layers=g("n_xattn_layers", args.n_xattn_layers),
            step_emb_dim=g("step_emb_dim", args.step_emb_dim),
        )
        missing, unexpected = adaptor.load_state_dict(
            ckpt["adaptor"], strict=False)
        if missing or unexpected:
            print(f"[eval] state_dict mismatch: missing={missing} unexpected={unexpected}")
        print(f"[eval] loaded adaptor from {ckpt_path} (step {ckpt.get('step','?')}, "
              f"action_dim={g('action_dim', args.action_dim)}, "
              f"fusion={g('arch_fusion', args.arch_fusion)}, head={g('arch_head', args.arch_head)}, "
              f"train_guide_scale={g('train_guide_scale', None)})")
        z_init_state = ckpt.get("z_init_predictor", None)
        if z_init_state is not None:
            print(f"[eval] checkpoint contains z_init_predictor "
                  f"shape={ckpt.get('z_init_shape', 'unknown')}")
        return adaptor.to(device).eval(), train_args, {
            "state": z_init_state,
            "shape": ckpt.get("z_init_shape", None),
        }
    else:
        adaptor = TrajectoryAdaptor(
            adaptor_init_path=adaptor_init,
            hidden=args.hidden,
            action_dim=args.action_dim,
            action_len=args.action_len,
            action_repr=args.action_repr,
            fusion=args.arch_fusion,
            head=args.arch_head,
            rank_k=args.rank_k,
            n_xattn_layers=args.n_xattn_layers,
            step_emb_dim=args.step_emb_dim,
        )
        print(f"[eval] using ADAPTOR AT INIT (no checkpoint — M0 mode), "
              f"action_dim={args.action_dim} fusion={args.arch_fusion} head={args.arch_head}")
        return adaptor.to(device).eval(), {}, {"state": None, "shape": None}


# ---------------------------------------------------------------------------
# Sampler (Euler with CFG, matching embedding_search.regenerate_with_positive_embeds)
# ---------------------------------------------------------------------------


@torch.no_grad()
def replay_with_per_step_context(
    pipe: WanTI2V,
    pos_embeds: list[torch.Tensor],     # len N, each [L, D]
    null_context: torch.Tensor,         # [L_null, D]
    z_init: torch.Tensor,                # [C, F_z, H_z, W_z] (noise; sigma_max-scaled)
    z_I0_full: torch.Tensor,             # [C, F_z, H_z, W_z]
    mask2_full: torch.Tensor,            # [C, F_z, H_z, W_z]
    seq_len: int,
    sigmas: torch.Tensor,                # [N+1]
    guide_scale: float,
):
    """Replay Wan's CFG-guided Euler sampler using a list of per-step
    positive contexts. Mirrors embedding_search.regenerate_with_positive_embeds
    (which is also Euler + per-step C_i) so the σ-trajectory matches the
    one the adaptor was trained on."""
    N = len(sigmas) - 1
    mask2_zero = mask2_full[0]
    param_dtype = pipe.param_dtype

    z = ES._apply_first_frame_pin(z_init.detach().float(), z_I0_full, mask2_full)
    for i in range(N):
        sigma_i = sigmas[i].item()
        sigma_ip1 = sigmas[i + 1].item()
        v_cond = ES._dit_velocity(
            pipe, z, sigma_i, [pos_embeds[i]], seq_len,
            mask2_zero, param_dtype)
        v_uncond = ES._dit_velocity(
            pipe, z, sigma_i, [null_context], seq_len,
            mask2_zero, param_dtype)
        v_cfg = v_uncond + guide_scale * (v_cond - v_uncond)
        z = ES._apply_first_frame_pin(
            z + (sigma_ip1 - sigma_i) * v_cfg, z_I0_full, mask2_full)
    return z


# ---------------------------------------------------------------------------
# Per-triplet evaluation
# ---------------------------------------------------------------------------


def load_ground_truth_pixels(pipe: WanTI2V, triplet_dir: str, max_area: int):
    """Load the real reference clip at the same pixel grid the latents were
    encoded at, so the SSIM comparison is apples-to-apples.

    Two layouts are supported:
      * DROID cache: a meta.json records {video_path, start_frame, frames,
        max_area}. We read those exact frames from the source video.
      * Old triplets: video.mp4 + two frame_<NNN>.png markers in-dir.
    Returns None if neither is available (caller falls back to the VAE
    round-trip of z_video as the reference)."""
    import re, glob
    meta_path = os.path.join(triplet_dir, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        vpath = meta.get("video_path")
        if vpath and os.path.exists(vpath):
            start = int(meta["start_frame"])
            delta = int(meta["frames"]) - 1
            area = int(meta.get("max_area", max_area))
            clip_pil = ES.load_reference_clip(vpath, start, delta)
            clip_pil, _ = ES.fit_clip_to_pipeline(
                clip_pil, pipe.vae_stride, pipe.patch_size, area)
            return ES.clip_to_tensor_signed(clip_pil, pipe.device)
        return None  # meta but source video not reachable from here

    nums = []
    for f in glob.glob(os.path.join(triplet_dir, "frame_*.png")):
        m = re.search(r"frame_(\d+)", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    if len(nums) < 2 or not os.path.exists(os.path.join(triplet_dir, "video.mp4")):
        return None
    nums.sort()
    start, end = nums[0], nums[-1]
    delta = (end - start) // 4 * 4
    clip_pil = ES.load_reference_clip(
        os.path.join(triplet_dir, "video.mp4"), start, delta)
    clip_pil, _ = ES.fit_clip_to_pipeline(
        clip_pil, pipe.vae_stride, pipe.patch_size, max_area)
    return ES.clip_to_tensor_signed(clip_pil, pipe.device)   # [3, F, H, W]


def per_frame_ssim(video_pred: torch.Tensor, video_gt: torch.Tensor) -> torch.Tensor:
    """Both [3, F, H, W] in [-1, 1]. Returns per-frame SSIM tensor [F]."""
    Fn = video_pred.shape[1]
    return torch.tensor([
        L.ssim(video_pred[:, k], video_gt[:, k]).item() for k in range(Fn)
    ])


def eval_triplet(
    pipe: WanTI2V,
    adaptor: TrajectoryAdaptor,
    name: str,
    triplet_dir: str,
    null_context: torch.Tensor,
    sigmas: torch.Tensor,
    args,
    out_dir: str,
    z_init_meta: dict | None = None,
    train_args: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    device = pipe.device

    # ---- load precomputed inputs ----
    actions = torch.from_numpy(
        np.load(os.path.join(triplet_dir, "actions.npy"))).float().unsqueeze(0).to(device)
    z_I0 = torch.load(
        os.path.join(triplet_dir, "z_I0.pt"),
        map_location=device, weights_only=False).float().unsqueeze(0)
    z_video = torch.load(
        os.path.join(triplet_dir, "z_video.pt"),
        map_location=device, weights_only=False).float()                # [C, F_z, H_z, W_z]

    C, F_z, H_z, W_z = z_video.shape
    z_I0_full = z_I0.squeeze(0).expand(-1, F_z, -1, -1).contiguous().float()
    mask2_full = torch.ones(C, F_z, H_z, W_z, device=device, dtype=torch.float32)
    mask2_full[:, 0] = 0.0
    seq_len = (F_z * H_z * W_z) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int((seq_len + pipe.sp_size - 1) // pipe.sp_size) * pipe.sp_size

    # ---- adaptor forward → per-step context list ----
    adaptor.eval()
    with torch.no_grad():
        C_traj = adaptor(actions, z_I0)                                  # [1, N, L, D]
    pos_embeds = [C_traj[0, i].detach() for i in range(C_traj.shape[1])]

    # ---- diagnostic: residual magnitude (head-agnostic) ----
    with torch.no_grad():
        diag = adaptor.initial_factors(actions, z_I0)
    if "gamma" in diag:
        g = diag["gamma"]                       # [B, K]  (rank1 / rankk)
        print(f"[eval] {name}: |γ|_mean={g.abs().mean().item():.4e} "
              f"(K={g.shape[-1]}), γ={g.flatten().tolist()}")
    else:
        rn = diag["residual_norm"]              # [B, N]  (perstep)
        print(f"[eval] {name}: residual_norm mean={rn.mean().item():.4e} "
              f"min={rn.min().item():.4e} max={rn.max().item():.4e}")

    # ---- initial noise (fixed seed for A/B reproducibility) ----
    z_dim = pipe.vae.model.z_dim
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(int(args.seed))
    noise = torch.randn(
        z_dim, F_z, H_z, W_z, dtype=torch.float32,
        generator=seed_g, device=device)

    # ---- optional saved inversion embedding + z_init diagnostics ----
    oracle_data = None
    if args.include_oracle or args.include_adaptor_oracle_zinit or args.include_oracle_seed:
        oracle_path = resolve_oracle_path(args.oracle_root, name)
        if oracle_path is not None:
            oracle_data = torch.load(oracle_path, map_location=device, weights_only=False)
            print(f"[eval] {name}: oracle source {oracle_path}")
        else:
            print(f"[eval] {name}: WARN oracle requested but no positive_embeddings.pt "
                  f"found under {args.oracle_root}")

    # ---- always: VAE roundtrip of ground truth, for upper-bound reference ----
    metrics = {"triplet": name}
    if oracle_data is not None and "positive_embeddings" in oracle_data:
        oracle_pe = oracle_data["positive_embeddings"]
        oracle_ctx = (
            torch.stack([e.float() for e in oracle_pe], dim=0)
            if isinstance(oracle_pe, list) else oracle_pe.float()
        ).to(device)
        if tuple(oracle_ctx.shape) == tuple(C_traj[0].shape):
            ctx_delta = C_traj[0].detach().float() - oracle_ctx
            metrics["ctx_oracle_mse"] = float(ctx_delta.pow(2).mean().item())
            metrics["ctx_oracle_max_abs"] = float(ctx_delta.abs().max().item())
            print(f"[eval] {name}: ctx_vs_oracle mse={metrics['ctx_oracle_mse']:.6e} "
                  f"max_abs={metrics['ctx_oracle_max_abs']:.6e}")
        else:
            print(f"[eval] {name}: WARN oracle context shape {tuple(oracle_ctx.shape)} "
                  f"!= adaptor context shape {tuple(C_traj[0].shape)}")
    with torch.no_grad():
        gt_pixels = pipe.vae.decode([z_video])[0]
    if args.include_groundtruth:
        ES.save_video(gt_pixels, os.path.join(out_dir, "ground_truth.mp4"))

    # Need real reference frames for SSIM scoring. If the source video isn't
    # reachable (e.g. a moved DROID cache), fall back to the VAE round-trip
    # of z_video — SSIM then measures against the ceiling rather than raw
    # pixels, which is still a valid relative comparison across methods.
    ref_pixels = load_ground_truth_pixels(pipe, triplet_dir, args.max_area)
    if ref_pixels is None:
        print(f"[eval] {name}: real reference frames unavailable; "
              "scoring SSIM against VAE round-trip of z_video instead")
        ref_pixels = gt_pixels
    ES.save_video(ref_pixels, os.path.join(out_dir, "reference_clip.mp4"))

    # ---- adaptor sample ----
    t0 = time.time()
    z_pred = replay_with_per_step_context(
        pipe=pipe,
        pos_embeds=pos_embeds,
        null_context=null_context,
        z_init=noise,
        z_I0_full=z_I0_full,
        mask2_full=mask2_full,
        seq_len=seq_len,
        sigmas=sigmas,
        guide_scale=args.guide_scale,
    )
    with torch.no_grad():
        video_pred = pipe.vae.decode([z_pred])[0]                        # [3, F, H, W]
    ES.save_video(video_pred, os.path.join(out_dir, "sample.mp4"))
    ssim_pred = per_frame_ssim(video_pred, ref_pixels)
    metrics["sample_ssim_avg"] = float(ssim_pred.mean())
    metrics["sample_ssim_last"] = float(ssim_pred[-1])
    metrics["sample_latent_mse_full"] = float(F.mse_loss(z_pred, z_video).item())
    print(f"[eval] {name}: adaptor   ssim_avg={metrics['sample_ssim_avg']:.4f} "
          f"latent_mse={metrics['sample_latent_mse_full']:.5f}  "
          f"({time.time() - t0:.1f}s)")

    # ---- adaptor sample from an explicitly supplied initial latent ----
    if args.sample_z_init_path:
        z_fixed_init = load_z_init_tensor(args.sample_z_init_path).to(device)
        if tuple(z_fixed_init.shape) != tuple(z_video.shape):
            raise ValueError(
                f"--sample_z_init_path shape {tuple(z_fixed_init.shape)} does not "
                f"match z_video {tuple(z_video.shape)}")
        t0 = time.time()
        z_fixed = replay_with_per_step_context(
            pipe=pipe,
            pos_embeds=pos_embeds,
            null_context=null_context,
            z_init=z_fixed_init.float(),
            z_I0_full=z_I0_full,
            mask2_full=mask2_full,
            seq_len=seq_len,
            sigmas=sigmas,
            guide_scale=args.guide_scale,
        )
        with torch.no_grad():
            video_fixed = pipe.vae.decode([z_fixed])[0]
        ES.save_video(video_fixed, os.path.join(out_dir, "sample_fixed_zinit.mp4"))
        ssim_fixed = per_frame_ssim(video_fixed, ref_pixels)
        metrics["sample_fixed_zinit_ssim_avg"] = float(ssim_fixed.mean())
        metrics["sample_fixed_zinit_ssim_last"] = float(ssim_fixed[-1])
        metrics["sample_fixed_zinit_latent_mse_full"] = float(
            F.mse_loss(z_fixed, z_video).item())
        print(f"[eval] {name}: adaptor@fixed_zinit "
              f"ssim_avg={metrics['sample_fixed_zinit_ssim_avg']:.4f} "
              f"latent_mse={metrics['sample_fixed_zinit_latent_mse_full']:.5f} "
              f"({time.time() - t0:.1f}s)")

    # ---- adaptor sample from a learned initial latent predictor ----
    z_meta = z_init_meta or {}
    if args.include_pred_zinit:
        z_state = z_meta.get("state")
        if z_state is None:
            print(f"[eval] {name}: WARN --include_pred_zinit requested but "
                  "checkpoint has no z_init_predictor")
        else:
            ta = train_args or {}
            z_pred_module = ZInitPredictor(
                latent_shape=z_video.shape,
                hidden=ta.get("z_init_hidden", 512),
                action_dim=ta.get("action_dim", args.action_dim),
                action_len=ta.get("action_len", args.action_len),
                action_repr=ta.get("action_repr", args.action_repr),
            ).to(device).eval()
            missing, unexpected = z_pred_module.load_state_dict(z_state, strict=False)
            if missing or unexpected:
                print(f"[eval] z_init_predictor mismatch: "
                      f"missing={missing} unexpected={unexpected}")
            with torch.no_grad():
                z_pred_init = z_pred_module(actions, z_I0, noise.unsqueeze(0))[0].float()
            torch.save(z_pred_init.detach().cpu(), os.path.join(out_dir, "pred_z_init.pt"))
            if oracle_data is not None and oracle_data.get("z_init", None) is not None:
                z_oracle_init = oracle_data["z_init"].to(device).float()
                metrics["pred_zinit_oracle_mse"] = float(
                    F.mse_loss(z_pred_init * mask2_full, z_oracle_init * mask2_full).item())
                metrics["pred_zinit_oracle_rmse"] = float(metrics["pred_zinit_oracle_mse"] ** 0.5)
                print(f"[eval] {name}: pred_zinit_vs_oracle "
                      f"mse={metrics['pred_zinit_oracle_mse']:.6f} "
                      f"rmse={metrics['pred_zinit_oracle_rmse']:.4f}")
            t0 = time.time()
            z_pred_from_pred_init = replay_with_per_step_context(
                pipe=pipe,
                pos_embeds=pos_embeds,
                null_context=null_context,
                z_init=z_pred_init,
                z_I0_full=z_I0_full,
                mask2_full=mask2_full,
                seq_len=seq_len,
                sigmas=sigmas,
                guide_scale=args.guide_scale,
            )
            with torch.no_grad():
                video_pred_zinit = pipe.vae.decode([z_pred_from_pred_init])[0]
            ES.save_video(video_pred_zinit, os.path.join(out_dir, "sample_pred_zinit.mp4"))
            ssim_pred_zinit = per_frame_ssim(video_pred_zinit, ref_pixels)
            metrics["sample_pred_zinit_ssim_avg"] = float(ssim_pred_zinit.mean())
            metrics["sample_pred_zinit_ssim_last"] = float(ssim_pred_zinit[-1])
            metrics["sample_pred_zinit_latent_mse_full"] = float(
                F.mse_loss(z_pred_from_pred_init, z_video).item())
            print(f"[eval] {name}: adaptor@pred_zinit "
                  f"ssim_avg={metrics['sample_pred_zinit_ssim_avg']:.4f} "
                  f"latent_mse={metrics['sample_pred_zinit_latent_mse_full']:.5f} "
                  f"({time.time() - t0:.1f}s)")

    # ---- adaptor contexts from the saved inversion endpoint ----
    if args.include_adaptor_oracle_zinit and oracle_data is not None:
        z_adapt_init = oracle_data.get("z_init", None)
        if z_adapt_init is None:
            print(f"[eval] {name}: WARN oracle file has no z_init; "
                  "cannot render sample_oracle_zinit.mp4")
        else:
            z_adapt_orc = replay_with_per_step_context(
                pipe=pipe,
                pos_embeds=pos_embeds,
                null_context=null_context,
                z_init=z_adapt_init.to(device).float(),
                z_I0_full=z_I0_full,
                mask2_full=mask2_full,
                seq_len=seq_len,
                sigmas=sigmas,
                guide_scale=args.guide_scale,
            )
            with torch.no_grad():
                video_adapt_orc = pipe.vae.decode([z_adapt_orc])[0]
            ES.save_video(video_adapt_orc, os.path.join(out_dir, "sample_oracle_zinit.mp4"))
            ssim_adapt_orc = per_frame_ssim(video_adapt_orc, ref_pixels)
            metrics["sample_oracle_zinit_ssim_avg"] = float(ssim_adapt_orc.mean())
            metrics["sample_oracle_zinit_ssim_last"] = float(ssim_adapt_orc[-1])
            metrics["sample_oracle_zinit_latent_mse_full"] = float(
                F.mse_loss(z_adapt_orc, z_video).item())
            print(f"[eval] {name}: adaptor@zinit "
                  f"ssim_avg={metrics['sample_oracle_zinit_ssim_avg']:.4f} "
                  f"latent_mse={metrics['sample_oracle_zinit_latent_mse_full']:.5f}")

    # ---- oracle (saved per-step inversion) ----
    if args.include_oracle:
        if oracle_data is not None:
            oracle_embeds = [e.float() for e in oracle_data["positive_embeddings"]]
            # IMPORTANT: the saved per-step embeddings were optimized to
            # reconstruct the clip from ONE specific initial noise — the
            # inversion endpoint z_init = traj[0], stored alongside them.
            # They are tied to that noise basin (§1.5). Replaying them from
            # fresh random noise drives the trajectory out of basin and the
            # decode blows out (pure white). To reproduce the inversion's own
            # reconstruction.mp4 (the true per-clip ceiling) we must start
            # from the saved z_init. The adaptor, trained with stochastic ε,
            # is noise-robust and correctly samples from fresh noise above.
            z_orc_init = noise
            if "z_init" in oracle_data and oracle_data["z_init"] is not None:
                z_orc_init = oracle_data["z_init"].to(device).float()
            else:
                print(f"[eval] {name}: WARN oracle file has no z_init; "
                      "replaying from random noise (will likely diverge)")
            z_orc = replay_with_per_step_context(
                pipe=pipe, pos_embeds=oracle_embeds,
                null_context=null_context, z_init=z_orc_init,
                z_I0_full=z_I0_full, mask2_full=mask2_full,
                seq_len=seq_len, sigmas=sigmas, guide_scale=args.guide_scale,
            )
            with torch.no_grad():
                video_orc = pipe.vae.decode([z_orc])[0]
            ES.save_video(video_orc, os.path.join(out_dir, "oracle.mp4"))
            ssim_orc = per_frame_ssim(video_orc, ref_pixels)
            metrics["oracle_ssim_avg"] = float(ssim_orc.mean())
            metrics["oracle_ssim_last"] = float(ssim_orc[-1])
            metrics["oracle_latent_mse_full"] = float(F.mse_loss(z_orc, z_video).item())
            print(f"[eval] {name}: oracle    ssim_avg={metrics['oracle_ssim_avg']:.4f} "
                  f"latent_mse={metrics['oracle_latent_mse_full']:.5f}")
        else:
            print(f"[eval] {name}: WARN oracle render skipped")

    # ---- oracle embeddings from eval seed-0 noise ----
    if args.include_oracle_seed and oracle_data is not None:
        oracle_embeds = [e.float() for e in oracle_data["positive_embeddings"]]
        z_orc_seed = replay_with_per_step_context(
            pipe=pipe, pos_embeds=oracle_embeds,
            null_context=null_context, z_init=noise,
            z_I0_full=z_I0_full, mask2_full=mask2_full,
            seq_len=seq_len, sigmas=sigmas, guide_scale=args.guide_scale,
        )
        with torch.no_grad():
            video_orc_seed = pipe.vae.decode([z_orc_seed])[0]
        ES.save_video(video_orc_seed, os.path.join(out_dir, "oracle_seed0.mp4"))
        ssim_orc_seed = per_frame_ssim(video_orc_seed, ref_pixels)
        metrics["oracle_seed0_ssim_avg"] = float(ssim_orc_seed.mean())
        metrics["oracle_seed0_ssim_last"] = float(ssim_orc_seed[-1])
        metrics["oracle_seed0_latent_mse_full"] = float(F.mse_loss(z_orc_seed, z_video).item())
        print(f"[eval] {name}: oracle@seed0 "
              f"ssim_avg={metrics['oracle_seed0_ssim_avg']:.4f} "
              f"latent_mse={metrics['oracle_seed0_latent_mse_full']:.5f}")

    # ---- null baseline (T5("") in both slots) ----
    if args.include_null:
        N = len(sigmas) - 1
        null_embeds = [null_context.float()] * N
        z_null = replay_with_per_step_context(
            pipe=pipe, pos_embeds=null_embeds,
            null_context=null_context, z_init=noise,
            z_I0_full=z_I0_full, mask2_full=mask2_full,
            seq_len=seq_len, sigmas=sigmas, guide_scale=args.guide_scale,
        )
        with torch.no_grad():
            video_null = pipe.vae.decode([z_null])[0]
        ES.save_video(video_null, os.path.join(out_dir, "null_only.mp4"))
        ssim_null = per_frame_ssim(video_null, ref_pixels)
        metrics["null_ssim_avg"] = float(ssim_null.mean())
        metrics["null_ssim_last"] = float(ssim_null[-1])
        metrics["null_latent_mse_full"] = float(F.mse_loss(z_null, z_video).item())
        print(f"[eval] {name}: null      ssim_avg={metrics['null_ssim_avg']:.4f} "
              f"latent_mse={metrics['null_latent_mse_full']:.5f}")

    # ---- VAE roundtrip ceiling SSIM (decode→encode reference) ----
    ssim_vae = per_frame_ssim(gt_pixels, ref_pixels)
    metrics["vae_ceil_ssim_avg"] = float(ssim_vae.mean())
    metrics["vae_ceil_ssim_last"] = float(ssim_vae[-1])
    print(f"[eval] {name}: VAE-ceil  ssim_avg={metrics['vae_ceil_ssim_avg']:.4f}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    if not os.path.isdir(args.triplets_root):
        raise FileNotFoundError(f"--triplets_root not found: {args.triplets_root}")
    if args.train_triplets_root is not None and not os.path.isdir(args.train_triplets_root):
        raise FileNotFoundError(
            f"--train_triplets_root not found: {args.train_triplets_root}")
    if not os.path.isfile(args.adaptor_init):
        raise FileNotFoundError(f"--adaptor_init not found: {args.adaptor_init}")
    if args.ckpt_path != "none" and not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(
            f"--ckpt_path {args.ckpt_path!r} not found. Pass 'none' to evaluate init.")
    if args.sample_z_init_path is not None and not os.path.isfile(args.sample_z_init_path):
        raise FileNotFoundError(f"--sample_z_init_path not found: {args.sample_z_init_path}")
    if args.sigma_schedule_path is not None and not os.path.isfile(args.sigma_schedule_path):
        raise FileNotFoundError(f"--sigma_schedule_path not found: {args.sigma_schedule_path}")
    if not os.path.isdir(args.ckpt_dir):
        raise FileNotFoundError(f"--ckpt_dir not found: {args.ckpt_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Wan pipeline ----
    print(f"[eval] loading Wan TI2V-5B from {args.ckpt_dir}…")
    pipe = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    pipe.model.eval().requires_grad_(False)
    pipe.vae.model.eval().requires_grad_(False)
    pipe.text_encoder.model.eval().requires_grad_(False)
    pipe.model.to(pipe.device)
    print(f"[eval] loaded. device={pipe.device}")

    # ---- T5("") once, then T5 to CPU ----
    pipe.text_encoder.model.to(pipe.device)
    with torch.no_grad():
        null_context = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    print(f"[eval] T5('') shape={tuple(null_context.shape)}")

    # ---- sigma schedule (must match training) ----
    if args.sigma_schedule_path is not None:
        sigmas = load_sigma_schedule(args.sigma_schedule_path, pipe.device)
        if sigmas.numel() != args.sampling_steps + 1:
            raise ValueError(
                f"--sigma_schedule_path length {sigmas.numel()} does not match "
                f"--sampling_steps + 1 = {args.sampling_steps + 1}")
    else:
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=pipe.num_train_timesteps,
            shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(args.sampling_steps, device=pipe.device, shift=args.shift)
        sigmas = scheduler.sigmas.to(pipe.device).float()
    print(f"[eval] sigmas: {sigmas[0].item():.4f} → {sigmas[-1].item():.4f} "
          f"(N+1={len(sigmas)})")

    # ---- adaptor ----
    adaptor, train_args, z_init_meta = load_adaptor(
        args.adaptor_init, args.ckpt_path, pipe.device, args)
    print(f"[eval] adaptor: N={adaptor.num_steps} L={adaptor.L} D={adaptor.D} "
          f"params={sum(p.numel() for p in adaptor.parameters())/1e6:.2f}M")

    # ---- resolve eval guide_scale (AUTO = match training) ----
    if args.guide_scale < 0:
        args.guide_scale = float(train_args.get("train_guide_scale", 5.0))
        print(f"[eval] guide_scale=AUTO -> {args.guide_scale} "
              f"(from checkpoint's train_guide_scale)")
    else:
        tw = train_args.get("train_guide_scale", None)
        if tw is not None and abs(tw - args.guide_scale) > 1e-6:
            print(f"[eval] WARNING: eval guide_scale={args.guide_scale} != "
                  f"train_guide_scale={tw}. A mismatch usually breaks "
                  f"reconstruction (see §3.13).")

    # ---- build the list of (split, name, triplet_dir) to evaluate ----
    def _resolve(root, n):
        tdir = os.path.join(root, n)
        if os.path.isdir(tdir):
            return tdir
        alt = n[len("triplet_"):] if n.startswith("triplet_") else n
        alt = os.path.join(root, alt)
        return alt if os.path.isdir(alt) else None

    def _sorted_subdir_names(root):
        import glob as _glob
        names = [os.path.basename(d) for d in _glob.glob(os.path.join(root, "*"))
                 if os.path.isdir(d)]

        def _key(n):
            return (0, int(n)) if n.isdigit() else (1, n)
        names.sort(key=_key)
        return names

    groups = [("val", args.triplets_root, args.eval_names)]
    if args.train_triplets_root:
        if args.train_eval_names:
            tnames = args.train_eval_names
        else:
            tnames = _sorted_subdir_names(args.train_triplets_root)[: args.n_train_eval]
        groups.append(("train", args.train_triplets_root, tnames))
        print(f"[eval] also evaluating {len(tnames)} TRAIN episodes from "
              f"{args.train_triplets_root}: {tnames[:5]}")

    # ---- loop over groups × triplets ----
    summary = []
    for split, root, names in groups:
        for n in names:
            tdir = _resolve(root, n)
            if tdir is None:
                print(f"[eval] WARN {split}/{n}: triplet dir not found, skip")
                continue
            out_dir = os.path.join(args.output_dir, split, f"triplet_{n}")
            metrics = eval_triplet(
                pipe=pipe, adaptor=adaptor, name=n, triplet_dir=tdir,
                null_context=null_context, sigmas=sigmas, args=args,
                out_dir=out_dir, z_init_meta=z_init_meta, train_args=train_args,
            )
            metrics["split"] = split
            summary.append(metrics)
            torch.cuda.empty_cache()

    # ---- aggregate CSV ----
    if summary:
        keys = list(summary[0].keys())
        for s in summary:
            for k in s:
                if k not in keys:
                    keys.append(k)
        csv_path = os.path.join(args.output_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summary:
                w.writerow(s)
        # Print one-line summary per triplet, grouped by split, with means.
        print("\n[eval] === summary ===")
        for split in ("train", "val"):
            rows = [s for s in summary if s.get("split") == split]
            if not rows:
                continue
            print(f"  [{split}]")
            for s in rows:
                print(f"    {s['triplet']:>8}: "
                      f"adaptor={s.get('sample_ssim_avg', float('nan')):.4f}  "
                      f"pred_z={s.get('sample_pred_zinit_ssim_avg', float('nan')):.4f}  "
                      f"oracle={s.get('oracle_ssim_avg', float('nan')):.4f}  "
                      f"null={s.get('null_ssim_avg', float('nan')):.4f}  "
                      f"VAE-ceil={s.get('vae_ceil_ssim_avg', float('nan')):.4f}")
            m = [s["sample_ssim_avg"] for s in rows if "sample_ssim_avg" in s]
            if m:
                print(f"    -> {split} adaptor mean SSIM = {np.mean(m):.4f} "
                      f"(n={len(m)})")
        # Fit-vs-generalization gap, if both splits present.
        tr = [s["sample_ssim_avg"] for s in summary
              if s.get("split") == "train" and "sample_ssim_avg" in s]
        va = [s["sample_ssim_avg"] for s in summary
              if s.get("split") == "val" and "sample_ssim_avg" in s]
        if tr and va:
            print(f"\n[eval] train SSIM {np.mean(tr):.4f}  vs  val SSIM "
                  f"{np.mean(va):.4f}  (gap {np.mean(tr) - np.mean(va):+.4f})")
        print(f"[eval] CSV: {csv_path}")

    print(f"[eval] done. outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
