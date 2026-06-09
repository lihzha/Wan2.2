"""One-time preprocessing: VAE-encode I_0 and the 33-frame reference clip
for every triplet, and cache them to disk so training never touches Wan's
VAE again.

For each triplet under <triplets_root>/<name>/, reads:
    actions.npy        # [32, 8]
    frame_<NNN>.png    # two of them; smaller NNN = start_index
    video.mp4          # full reference video

Writes:
    z_I0.pt            # [48, 1, H_z, W_z]   VAE-encoded first frame
    z_video.pt         # [48, 9, H_z, W_z]   VAE-encoded 33-frame clip
    meta.json          # start_index, delta_index, oh, ow, frames

Usage:
  python scripts/precompute_features.py \\
      --triplets_root data/triplets \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --max_area 230400

The (start_index, delta_index, max_area) defaults match the values used
during positive_inversion (see runs/batch_inv_positive/triplet_*/config.json),
so the precomputed VAE features land on the same pixel grid the saved
{C_t} embeddings were optimized against.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Reuse the same image-fit helpers used by the inversion script so the
# precomputed latents match the pixel grid of the optimized embeddings.
from embedding_search import (  # noqa: E402
    fit_clip_to_pipeline,
    load_reference_clip,
    to_tensor_signed,
)
from wan.configs.wan_ti2v_5B import ti2v_5B  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def parse_triplet_frames(triplet_dir):
    """Mirror scripts/batch_inversion.py:parse_triplet_frames."""
    files = glob.glob(os.path.join(triplet_dir, "frame_*.png"))
    nums = []
    for f in files:
        m = re.search(r"frame_(\d+)", os.path.basename(f))
        if m:
            nums.append(int(m.group(1)))
    if len(nums) < 2:
        raise RuntimeError(
            f"{triplet_dir}: need at least 2 frame_*.png files; found {len(nums)}."
        )
    nums.sort()
    start, end = nums[0], nums[-1]
    delta = end - start
    if delta < 4:
        raise RuntimeError(f"{triplet_dir}: delta={delta} too small (need >=4).")
    delta = (delta // 4) * 4
    return start, delta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets_root", default="data/triplets")
    p.add_argument("--ckpt_dir", required=True,
                   help="Directory with Wan2.2_VAE.pth (i.e. the TI2V-5B ckpt dir).")
    p.add_argument(
        "--max_area", type=int, default=230400,
        help="Same pixel-area budget used during positive_inversion. "
        "Must match so the precomputed latents land on the same grid as "
        "the saved {C_t} embeddings (230400 = 480x480).",
    )
    p.add_argument(
        "--delta_index_default", type=int, default=32,
        help="Override the auto-parsed delta_index if frame_*.png filenames "
        "do not match the inversion's delta_index. Default 32 (= 33-frame clip).",
    )
    p.add_argument("--force", action="store_true",
                   help="Re-encode even if z_I0.pt / z_video.pt already exist.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    config = ti2v_5B
    print(f"[precompute] loading VAE from {args.ckpt_dir}/{config.vae_checkpoint}…")
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.ckpt_dir, config.vae_checkpoint),
        device=device,
    )
    vae_stride = config.vae_stride
    patch_size = config.patch_size

    triplet_dirs = [d for d in glob.glob(os.path.join(args.triplets_root, "*"))
                    if os.path.isdir(d)]

    def _key(d):
        name = os.path.basename(d)
        return (0, int(name)) if name.isdigit() else (1, name)
    triplet_dirs.sort(key=_key)
    if args.limit:
        triplet_dirs = triplet_dirs[: args.limit]

    print(f"[precompute] processing {len(triplet_dirs)} triplets")

    for idx, tdir in enumerate(triplet_dirs):
        name = os.path.basename(tdir)
        z_I0_path = os.path.join(tdir, "z_I0.pt")
        z_video_path = os.path.join(tdir, "z_video.pt")
        meta_path = os.path.join(tdir, "meta.json")
        if (not args.force and os.path.exists(z_I0_path) and
                os.path.exists(z_video_path) and os.path.exists(meta_path)):
            print(f"[precompute] [{idx+1}/{len(triplet_dirs)}] {name}: cached, skip")
            continue

        video_path = os.path.join(tdir, "video.mp4")
        if not os.path.exists(video_path):
            print(f"[precompute] [{idx+1}/{len(triplet_dirs)}] {name}: no video.mp4, skip")
            continue

        try:
            start, delta = parse_triplet_frames(tdir)
        except Exception as e:
            print(f"[precompute] [{idx+1}/{len(triplet_dirs)}] {name}: cannot parse "
                  f"frame indices ({e}), skip")
            continue

        if delta != args.delta_index_default:
            print(f"[precompute] [{idx+1}/{len(triplet_dirs)}] {name}: WARN "
                  f"parsed delta={delta} != default {args.delta_index_default}; "
                  "using parsed value")

        # Load 33-frame clip aligned to the same pixel grid as inversion.
        clip_pil = load_reference_clip(video_path, start, delta)
        clip_pil, (oh, ow) = fit_clip_to_pipeline(
            clip_pil, vae_stride, patch_size, args.max_area)
        # clip_to_tensor_signed lives in embedding_search; replicate inline so
        # we don't pull in the whole pipeline import.
        clip_pixels = torch.stack(
            [TF.to_tensor(f).sub_(0.5).div_(0.5) for f in clip_pil], dim=1
        ).to(device)  # [3, F, H, W], [-1, 1]
        img0_pixels = to_tensor_signed(clip_pil[0], device)  # [3, H, W]

        with torch.no_grad():
            z_I0 = vae.encode([img0_pixels.unsqueeze(1)])[0]   # [48, 1, H_z, W_z]
            z_video = vae.encode([clip_pixels])[0]              # [48, F_z, H_z, W_z]

        z_I0 = z_I0.detach().float().cpu()
        z_video = z_video.detach().float().cpu()

        torch.save(z_I0, z_I0_path)
        torch.save(z_video, z_video_path)
        meta = {
            "start_index": int(start),
            "delta_index": int(delta),
            "frames": int(delta) + 1,
            "max_area": int(args.max_area),
            "oh": int(oh),
            "ow": int(ow),
            "z_I0_shape": list(z_I0.shape),
            "z_video_shape": list(z_video.shape),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[precompute] [{idx+1}/{len(triplet_dirs)}] {name}: start={start} "
              f"delta={delta} oh={oh} ow={ow} "
              f"z_I0={tuple(z_I0.shape)} z_video={tuple(z_video.shape)}")

        torch.cuda.empty_cache()

    print("[precompute] done")


if __name__ == "__main__":
    main()
