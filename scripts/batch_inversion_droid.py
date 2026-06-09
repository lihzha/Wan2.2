"""Batch positive_inversion over DROID episodes, reusing one loaded Wan
pipeline. DROID analogue of scripts/batch_inversion.py (which is hardwired
to the data/triplets/ layout).

For each episode <ep> in videos/<split>/, runs
embedding_search.run_search(mode=positive_inversion) on view <view>'s mp4,
producing runs at <output_root>/<split>/ep<ep>_v<view>/ with the usual
positive_embeddings.pt + reconstruction.mp4 + vae_roundtrip.mp4 outputs.

Purpose: re-derive the shared μ/β warm-start (and validate the rank-1
factorization) at the DROID resolution/domain, because the original
adaptor_init.pt — derived from the 480²/60fps triplets — does NOT transfer
(μ-only generation is off-manifold for DROID; see docs/adaptor_design.md
§3.11). After this, run:

    python scripts/cross_video_factorize.py \\
        --runs_root <output_root>/<split> \\
        --mode positive \\
        --out_dir <output_root>/<split>/factorization

then train with --adaptor_init <output_root>/<split>/factorization/adaptor_init.pt.

Usage:
  python scripts/batch_inversion_droid.py \\
      --droid_root /scratch/gpfs/AM43/yy4041/data/droid_ctrl_world \\
      --split train --view 0 --limit 32 \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --output_root runs/droid_inv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from argparse import Namespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch

import embedding_search as ES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--droid_root",
                   default="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_root", default="runs/droid_inv")
    p.add_argument("--limit", type=int, default=32,
                   help="Number of episodes to invert. ~30-50 is enough for a "
                   "stable cross-video factorization (μ is a mean; γ/b come "
                   "from per-video SVD + cross-video alignment).")
    p.add_argument("--start_at", type=int, default=0)
    p.add_argument("--start_frame", type=int, default=0,
                   help="First frame of the clip (I_0) within each episode.")
    p.add_argument("--delta_index", type=int, default=32,
                   help="Clip = frames[start : start+delta+1]; must be 4n.")
    p.add_argument("--max_area", type=int, default=61440,
                   help="61440 = 320*192 native (no resize). Must match the "
                   "training precompute.")
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--null_inner_iters", type=int, default=10)
    p.add_argument("--null_lr", type=float, default=1e-2)
    p.add_argument("--L_pos", type=int, default=1)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--inversion_guide_scale", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def list_episode_ids(droid_root, split):
    vid_dir = os.path.join(droid_root, "videos", split)
    ids = [int(n) for n in os.listdir(vid_dir)
           if n.isdigit() and os.path.isdir(os.path.join(vid_dir, n))]
    ids.sort()
    return ids


def make_base_args(a):
    """Full Namespace with every field embedding_search.run_search reads.
    Mirrors scripts/batch_inversion.py's base; positive_inversion only uses
    a subset, but the rest must exist for config.json dump / dispatch."""
    return Namespace(
        ckpt_dir=a.ckpt_dir,
        mode="positive_inversion",
        init="empty",
        L_opt=16,
        L_pos=a.L_pos,
        loss="lpips",
        num_iters=300,
        lr=1e-2,
        null_inner_iters=a.null_inner_iters,
        null_lr=a.null_lr,
        inversion_guide_scale=a.inversion_guide_scale,
        sampling_steps=a.sampling_steps,
        guide_scale=a.guide_scale,
        max_area=a.max_area,
        shift=a.shift,
        seed=a.seed,
        log_every=25,
        validate=False,
        sampling_steps_val=40,
        guide_scale_val=1.0,
        heuristic="",
    )


def main():
    a = parse_args()
    if a.delta_index < 4 or a.delta_index % 4 != 0:
        raise SystemExit(f"--delta_index must be a positive multiple of 4 (got {a.delta_index}).")
    n_frames = a.delta_index + 1

    ep_ids = list_episode_ids(a.droid_root, a.split)[a.start_at:]
    if a.limit:
        ep_ids = ep_ids[: a.limit]
    out_split = os.path.join(a.output_root, a.split)
    os.makedirs(out_split, exist_ok=True)

    # Plan: which episodes need work (skip ones already inverted, skip too-short).
    import decord
    plan = []
    for ep in ep_ids:
        vid = os.path.join(a.droid_root, "videos", a.split, str(ep), f"{a.view}.mp4")
        out_dir = os.path.join(out_split, f"ep{ep}_v{a.view}")
        if os.path.exists(os.path.join(out_dir, "positive_embeddings.pt")):
            plan.append((ep, vid, out_dir, "skip (done)"))
            continue
        if not os.path.exists(vid):
            plan.append((ep, vid, out_dir, "skip (no mp4)"))
            continue
        try:
            n = len(decord.VideoReader(vid))
        except Exception as e:
            plan.append((ep, vid, out_dir, f"skip ({e})"))
            continue
        if n < a.start_frame + n_frames:
            plan.append((ep, vid, out_dir, f"skip (only {n} frames)"))
            continue
        plan.append((ep, vid, out_dir, "RUN"))

    runs_needed = [r for r in plan if r[-1] == "RUN"]
    print(f"[droid-inv] split={a.split} view={a.view}: {len(plan)} episodes, "
          f"{len(runs_needed)} need work -> {out_split}")
    for ep, _, _, status in plan[:20]:
        print(f"  ep{ep}: {status}")
    if a.dry_run:
        print("[droid-inv] --dry_run: not loading pipeline")
        return
    if not runs_needed:
        print("[droid-inv] nothing to do")
        return

    base = make_base_args(a)
    print(f"[droid-inv] building pipeline once from {a.ckpt_dir}…")
    pipe = ES.build_pipeline(base)

    t_start = time.time()
    manifest = []
    for i, (ep, vid, out_dir, status) in enumerate(plan):
        if status != "RUN":
            continue
        args_i = Namespace(**vars(base))
        args_i.reference_video = vid
        args_i.start_index = a.start_frame
        args_i.delta_index = a.delta_index
        args_i.frames = n_frames
        args_i.output_dir = out_dir
        print(f"\n[droid-inv] === ep{ep} (view {a.view}) === "
              f"({(time.time()-t_start)/60:.1f} min elapsed)")
        try:
            ES.run_search(args_i, pipe)
            manifest.append({"episode": ep, "view": a.view, "output_dir": out_dir,
                             "status": "ok"})
        except Exception as e:
            print(f"[droid-inv] ep{ep} FAILED: {e}")
            traceback.print_exc()
            manifest.append({"episode": ep, "status": "failed", "error": str(e)})
        finally:
            with open(os.path.join(out_split, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2)
            torch.cuda.empty_cache()

    print(f"\n[droid-inv] done in {(time.time()-t_start)/60:.1f} min -> {out_split}")
    print(f"[droid-inv] next: python scripts/cross_video_factorize.py "
          f"--runs_root {out_split} --mode positive "
          f"--out_dir {out_split}/factorization")


if __name__ == "__main__":
    main()
