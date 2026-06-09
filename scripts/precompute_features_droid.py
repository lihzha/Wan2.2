"""DROID → Wan-VAE feature cache for adaptor training.

The DROID `droid_ctrl_world` dataset stores, per episode:
    videos/<split>/<ep>/<view>.mp4         320x192, ~5 fps, ~56 frames, 3 views
    actions/<split>/<ep>.json              {"action": [[7], ...]}  (6 cartesian + 1 gripper)
    annotation/<split>/<ep>.json           metadata (video_length, texts, states, ...)
    latent_videos/<split>/<ep>/<view>.pt   [T, 4, 24, 40]  ← a 4-channel image VAE,
                                            NOT Wan's 48-channel VAE; we ignore it
                                            and re-encode with Wan's VAE.

This script re-encodes a clip with Wan's VAE at the native 320x192 grid and
writes the *same per-directory cache format* that scripts/train_adaptor.py's
TripletLatentDataset already consumes, so training needs no new dataloader:

    <out_root>/<split>/<name>/
        z_I0.pt      [48, 1, 12, 20]   Wan VAE encoding of frame `start`
        z_video.pt   [48, F_z, 12, 20] Wan VAE encoding of the clip
        actions.npy  [frames-1, 7]     action rows aligned to the clip transitions
        meta.json    bookkeeping

Resolution note: 320x192 is divisible by Wan's vae_stride*patch (=32) in both
axes, so best_output_size returns it unchanged when `max_area == 320*192`.
Latent grid is (H_z, W_z) = (12, 20). The shared μ/β warm-start in
adaptor_init.pt lives in text-embedding space (R^4096) and is resolution-
independent, so it is reused as-is (see docs/adaptor_design.md §3.10).

Temporal note: DROID is ~5 fps, so a 33-frame clip spans ~6.6 s — much
slower motion than the 60 fps source the original triplets were cut from.
Wan is frozen; the adaptor just learns whatever (action, I_0) → {C_t}
mapping fits. This is a known domain shift, acceptable for a resolution
verification run.

Usage:
  python scripts/precompute_features_droid.py \\
      --droid_root /scratch/gpfs/AM43/yy4041/data/droid_ctrl_world \\
      --split train --limit 300 --view 0 \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --out_root data/droid_cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import numpy as np
import torch

from embedding_search import (  # noqa: E402
    clip_to_tensor_signed,
    fit_clip_to_pipeline,
    to_tensor_signed,
)
from wan.configs.wan_ti2v_5B import ti2v_5B  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--droid_root",
                   default="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--ckpt_dir", required=True,
                   help="Dir with Wan2.2_VAE.pth (TI2V-5B ckpt dir).")
    p.add_argument("--out_root", default="data/droid_cache")
    p.add_argument("--view", type=int, default=0,
                   help="Which camera view (0/1/2) to encode.")
    p.add_argument("--frames", type=int, default=33,
                   help="Clip length; must be 4n+1 (Wan VAE temporal stride 4). "
                   "33 → F_z=9, matching the original triplets.")
    p.add_argument("--start_frame", type=int, default=0,
                   help="First frame of the clip within each episode (I_0).")
    p.add_argument("--limit", type=int, default=300,
                   help="Max episodes to process from this split.")
    p.add_argument("--start_at", type=int, default=0,
                   help="Skip the first N episodes (for sharding).")
    p.add_argument("--max_area", type=int, default=0,
                   help="0 (default) = use the native frame area so no resize "
                   "happens. Set explicitly to force a different budget.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def list_episode_ids(droid_root, split):
    """Episodes that have both an actions json and a videos dir."""
    vid_dir = os.path.join(droid_root, "videos", split)
    ids = []
    for name in os.listdir(vid_dir):
        if name.isdigit() and os.path.isdir(os.path.join(vid_dir, name)):
            ids.append(int(name))
    ids.sort()
    return ids


def main():
    args = parse_args()
    if args.frames < 5 or (args.frames - 1) % 4 != 0:
        raise SystemExit(f"--frames must be 4n+1 (got {args.frames}).")
    device = torch.device(args.device)

    config = ti2v_5B
    print(f"[droid] loading Wan VAE from {args.ckpt_dir}/{config.vae_checkpoint}…")
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.ckpt_dir, config.vae_checkpoint),
        device=device,
    )
    vae_stride, patch_size = config.vae_stride, config.patch_size

    ep_ids = list_episode_ids(args.droid_root, args.split)
    ep_ids = ep_ids[args.start_at:]
    if args.limit:
        ep_ids = ep_ids[: args.limit]
    out_split = os.path.join(args.out_root, args.split)
    os.makedirs(out_split, exist_ok=True)
    print(f"[droid] split={args.split} view={args.view}: {len(ep_ids)} episodes "
          f"(ids {ep_ids[0]}..{ep_ids[-1]}) -> {out_split}")

    n_ok, n_skip = 0, 0
    for i, ep in enumerate(ep_ids):
        name = f"ep{ep}_v{args.view}"
        out_dir = os.path.join(out_split, name)
        done = all(os.path.exists(os.path.join(out_dir, f)) for f in
                   ("z_I0.pt", "z_video.pt", "actions.npy", "meta.json"))
        if done and not args.force:
            n_ok += 1
            continue

        ann_path = os.path.join(args.droid_root, "annotation", args.split, f"{ep}.json")
        act_path = os.path.join(args.droid_root, "actions", args.split, f"{ep}.json")
        vid_path = os.path.join(args.droid_root, "videos", args.split, str(ep),
                                f"{args.view}.mp4")
        if not (os.path.exists(act_path) and os.path.exists(vid_path)):
            n_skip += 1
            continue

        # --- length check ---
        video_length = None
        if os.path.exists(ann_path):
            try:
                video_length = json.load(open(ann_path)).get("video_length")
            except Exception:
                video_length = None

        end = args.start_frame + args.frames
        # --- read clip frames (native 320x192) ---
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(vid_path)
        if len(vr) < end:
            n_skip += 1
            if i < 5 or i % 50 == 0:
                print(f"[droid] {name}: only {len(vr)} frames (< {end}), skip")
            continue
        idxs = list(range(args.start_frame, end))
        frames_np = vr.get_batch(idxs).asnumpy()       # [F, H, W, 3] uint8
        from PIL import Image
        clip_pil = [Image.fromarray(f).convert("RGB") for f in frames_np]

        # --- native area unless overridden ---
        H0, W0 = frames_np.shape[1], frames_np.shape[2]
        max_area = args.max_area if args.max_area > 0 else (H0 * W0)

        clip_pil, (oh, ow) = fit_clip_to_pipeline(
            clip_pil, vae_stride, patch_size, max_area)
        clip_pixels = clip_to_tensor_signed(clip_pil, device)            # [3, F, H, W]
        img0_pixels = to_tensor_signed(clip_pil[0], device)              # [3, H, W]

        # --- actions aligned to the clip transitions ---
        act = np.array(json.load(open(act_path))["action"], dtype=np.float32)  # [T, 7]
        a_start = args.start_frame
        a_end = a_start + (args.frames - 1)
        if a_end > act.shape[0]:
            # pad by repeating the last action (rare: action stream shorter than clip)
            pad = np.repeat(act[-1:], a_end - act.shape[0], axis=0)
            act = np.concatenate([act, pad], axis=0)
        actions_clip = act[a_start:a_end]                                # [frames-1, 7]

        with torch.no_grad():
            z_I0 = vae.encode([img0_pixels.unsqueeze(1)])[0].float().cpu()   # [48,1,H_z,W_z]
            z_video = vae.encode([clip_pixels])[0].float().cpu()             # [48,F_z,H_z,W_z]

        os.makedirs(out_dir, exist_ok=True)
        torch.save(z_I0, os.path.join(out_dir, "z_I0.pt"))
        torch.save(z_video, os.path.join(out_dir, "z_video.pt"))
        np.save(os.path.join(out_dir, "actions.npy"), actions_clip)
        meta = {
            "source": "droid_ctrl_world",
            "episode_id": int(ep),
            "view": int(args.view),
            "split": args.split,
            "start_frame": int(args.start_frame),
            "frames": int(args.frames),
            "video_length": int(video_length) if video_length is not None else None,
            "oh": int(oh), "ow": int(ow),
            "max_area": int(max_area),
            "z_I0_shape": list(z_I0.shape),
            "z_video_shape": list(z_video.shape),
            "action_shape": list(actions_clip.shape),
            "video_path": vid_path,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        n_ok += 1
        if i < 5 or i % 50 == 0:
            print(f"[droid] [{i+1}/{len(ep_ids)}] {name}: oh={oh} ow={ow} "
                  f"z_video={tuple(z_video.shape)} actions={tuple(actions_clip.shape)}")
        torch.cuda.empty_cache()

    print(f"[droid] done. ok/cached={n_ok} skipped={n_skip} -> {out_split}")


if __name__ == "__main__":
    main()
