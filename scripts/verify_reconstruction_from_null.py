"""Reproduce reconstruction.mp4 from ONLY I_0 + null_embeddings.pt.

Loads no GT-video frames, no inversion_trajectory.pt, no scheduler RNG.
Verifies that (model + I_0 + null_embeddings.pt) is sufficient to regenerate
the saved reconstruction.mp4 deterministically.

Usage:
  python scripts/verify_reconstruction_from_null.py \
      --run_dir runs/capacity_001 --ckpt_dir Wan2.2-TI2V-5B
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import embedding_search_losses as L  # noqa: E402

from wan.configs.wan_ti2v_5B import ti2v_5B  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402

from embedding_search import (  # noqa: E402
    fit_image_to_pipeline,
    regenerate_with_null_embeds,
    save_frame,
    save_video,
    to_tensor_signed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Output dir of the original null_inversion run "
                        "(e.g. runs/capacity_001).")
    p.add_argument("--ckpt_dir", required=True,
                   help="Path to Wan2.2-TI2V-5B checkpoint directory.")
    p.add_argument("--max_area", type=int, default=230400,
                   help="Must match the original run's --max_area.")
    p.add_argument("--out_name", default=None,
                   help="Output mp4 filename. Defaults to "
                        "'reconstruction_replay.mp4' (inverted z_init) or "
                        "'random_init_seed{S}.mp4' (--random_noise).")
    p.add_argument("--random_noise", action="store_true",
                   help="Ignore the inverted z_init in null_embeddings.pt and "
                        "start from a freshly sampled Gaussian noise of the "
                        "same shape. Reuses the optimized null_embeds, "
                        "context_pos, sigmas, and guide_scale.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed used when --random_noise is set.")
    return p.parse_args()


def main():
    args = parse_args()

    print("[verify] Loading WanTI2V-5B …")
    pipe = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    device = pipe.device
    print(f"[verify] Loaded. device={device}")

    # ---- ONLY I_0 from the run dir ----
    i0_path = os.path.join(args.run_dir, "I_0.png")
    img0 = Image.open(i0_path).convert("RGB")
    img0_fit, (oh, ow) = fit_image_to_pipeline(
        img0, pipe.vae_stride, pipe.patch_size, max_area=args.max_area)
    img0_pixels = to_tensor_signed(img0_fit, device)        # [3,H,W] in [-1,1]
    with torch.no_grad():
        z_I0 = pipe.vae.encode([img0_pixels.unsqueeze(1)])[0]   # [C,1,H_z,W_z]

    # ---- ONLY null_embeddings.pt ----
    nb_path = os.path.join(args.run_dir, "null_embeddings.pt")
    nb = torch.load(nb_path, map_location=device)
    z_init      = nb["z_init"].to(device).float()
    null_embeds = [e.to(device).float() for e in nb["null_embeddings"]]
    context_pos = nb["context_pos"].to(device).float()
    sigmas      = nb["sigmas"].to(device).float()
    guide_scale = float(nb["guide_scale"])
    print(
        f"[verify] z_init shape={tuple(z_init.shape)} "
        f"null_embeds N={len(null_embeds)} (each {tuple(null_embeds[0].shape)}) "
        f"sigmas N+1={sigmas.numel()} w={guide_scale}"
    )

    # ---- Optionally swap the inverted z_init for fresh Gaussian noise ----
    # Matches the deployment sampler convention: plain randn() of the latent
    # shape, dtype=float32, no sigma scaling (see WanTI2V.generate at
    # wan/textimage2video.py:489-495). All other replay inputs (null_embeds,
    # context_pos, sigmas, guide_scale) are kept as-is.
    if args.random_noise:
        gen = torch.Generator(device=device).manual_seed(args.seed)
        z_init = torch.randn(
            z_init.shape, dtype=torch.float32, device=device, generator=gen)
        print(
            f"[verify] --random_noise: replaced inverted z_init with "
            f"randn(seed={args.seed}); other replay inputs unchanged."
        )

    # ---- Shape-only derivations (no GT pixel content) ----
    C_z, F_z, H_z, W_z = z_init.shape
    mask2_full = torch.ones_like(z_init)
    mask2_full[:, 0] = 0.0
    z_I0_full = z_I0.expand(-1, F_z, -1, -1).contiguous().float()
    seq_len = (F_z * H_z * W_z) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = ((seq_len + pipe.sp_size - 1) // pipe.sp_size) * pipe.sp_size
    print(
        f"[verify] derived geom: F_z={F_z} H_z={H_z} W_z={W_z} seq_len={seq_len}"
    )

    if (z_I0.shape[-2], z_I0.shape[-1]) != (H_z, W_z):
        raise SystemExit(
            f"[verify] FAIL: I_0 encodes to {tuple(z_I0.shape)} but z_init has "
            f"H_z={H_z} W_z={W_z}. Likely a --max_area mismatch with the "
            f"original run."
        )

    # ---- Deterministic Euler replay with CFG ----
    pipe.model.to(device)
    print("[verify] Running Euler replay …")
    z_recon = regenerate_with_null_embeds(
        pipe=pipe, z_init=z_init,
        z_I0_full=z_I0_full, mask2_full=mask2_full, seq_len=seq_len,
        sigmas=sigmas, context_list=[context_pos],
        null_embeds=null_embeds, guide_scale=guide_scale,
        param_dtype=pipe.param_dtype,
    )
    with torch.no_grad():
        video_replay = pipe.vae.decode([z_recon])[0]            # [3,F,H,W] in [-1,1]

    if args.out_name is not None:
        out_name = args.out_name
    elif args.random_noise:
        out_name = f"random_init_seed{args.seed}.mp4"
    else:
        out_name = "reconstruction_replay.mp4"
    out_mp4 = os.path.join(args.run_dir, out_name)
    out_png = os.path.join(args.run_dir, out_name.replace(".mp4", "_last_frame.png"))
    latent_pt = os.path.join(args.run_dir, out_name.replace(".mp4", "_latent.pt"))
    save_video(video_replay, out_mp4)
    save_frame(video_replay[:, -1], out_png)
    torch.save(z_recon.detach().cpu(), latent_pt)
    print(f"[verify] wrote: {out_mp4}")

    # ---- Compare against the saved reconstruction.mp4 (sanity only) ----
    # Only meaningful when replaying from the inverted z_init; with random
    # noise the output is a fresh sample, not a reconstruction.
    saved_mp4 = os.path.join(args.run_dir, "reconstruction.mp4")
    if args.random_noise:
        print("[verify] --random_noise: skipping SSIM vs reconstruction.mp4 "
              "(not a reconstruction).")
    elif os.path.exists(saved_mp4):
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(saved_mp4)
        saved = vr.get_batch(range(len(vr))).float() / 127.5 - 1.0  # [F,H,W,3]
        saved = saved.permute(3, 0, 1, 2).to(device)                # [3,F,H,W]
        F_replay = video_replay.shape[1]
        F_saved = saved.shape[1]
        if F_saved != F_replay:
            print(f"[verify] WARN: saved has {F_saved} frames, replay has {F_replay}; "
                  f"comparing first min({F_saved},{F_replay})")
        K = min(F_saved, F_replay)
        ssims = torch.tensor([
            L.ssim(video_replay[:, k].cpu(), saved[:, k].cpu()).item()
            for k in range(K)
        ])
        print(
            f"[verify] replay vs saved reconstruction.mp4: "
            f"ssim_avg={ssims.mean().item():.4f} "
            f"ssim_min={ssims.min().item():.4f} "
            f"(<1.0 expected: lossy mp4 codec on both sides)"
        )
    else:
        print(f"[verify] no saved {saved_mp4}; skipping sanity SSIM.")

    print("[verify] DONE.")


if __name__ == "__main__":
    main()
