"""Run positive_inversion (or null_inversion) on every triplet under
data/triplets/<i>/ in sequence, sharing one loaded Wan pipeline so the
per-triplet startup cost is amortized.

For each triplet, the (start_index, delta_index) defaults come from the
two frame_<NNN>.png files in the triplet folder:
    start_index = smaller frame number
    delta_index = (larger - smaller), rounded down to a multiple of 4

If a triplet's run dir already has positive_embeddings.pt (or
null_embeddings.pt for --mode null_inversion), it is skipped — re-runs
are cheap to resume after interruptions or OOMs.

Usage:
  python scripts/batch_inversion.py \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --output_root runs/batch_inv_positive \\
      --mode positive_inversion \\
      --heuristic ""           # empty — purest action-conditioning test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import traceback
from argparse import Namespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import torch

import embedding_search as ES


def parse_triplet_frames(triplet_dir):
    """Return (start_index, delta_index) from the two frame_<NNN>.png
    files in the triplet folder. Rounds delta down to a multiple of 4."""
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
    delta = (delta // 4) * 4  # round down to multiple of 4
    return start, delta


def build_per_triplet_args(parent, triplet_name, video_path, heuristic,
                           start_index, delta_index, output_root):
    """Build an argparse.Namespace that embedding_search.run_search expects."""
    a = Namespace(**vars(parent))
    a.reference_video = video_path
    a.start_index = int(start_index)
    a.delta_index = int(delta_index)
    a.frames = int(delta_index) + 1
    a.heuristic = heuristic
    a.output_dir = os.path.join(output_root, f"triplet_{triplet_name}")
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets_root", default="data/triplets")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_root", default="runs/batch_inv_positive")
    p.add_argument(
        "--mode",
        choices=["positive_inversion", "null_inversion"],
        default="positive_inversion",
    )
    p.add_argument(
        "--heuristic_source",
        choices=["empty", "prompt_txt"],
        default="empty",
        help="'empty' = use \"\" for every triplet (cleanest capacity test). "
        "'prompt_txt' = use the contents of prompt.txt in each triplet "
        "folder (gives the model some semantic warm-start).",
    )
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--null_inner_iters", type=int, default=10)
    p.add_argument("--null_lr", type=float, default=1e-2)
    p.add_argument(
        "--L_pos",
        type=int,
        default=0,
        help="positive_inversion only: force the per-step positive "
        "embedding to have exactly L_pos tokens (0 = natural T5 length).",
    )
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--inversion_guide_scale", type=float, default=1.0)
    p.add_argument("--max_area", type=int, default=480 * 480)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many triplets.")
    p.add_argument("--start_at", type=int, default=0,
                   help="Skip the first N triplets (useful for slurm sharding).")
    p.add_argument("--manifest_suffix", default="",
                   help="Append this suffix to manifest.json (set this to "
                   "the array-task id when sharding so shards don't race).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print planned runs without loading the pipeline.")
    args_cli = p.parse_args()

    # Enumerate triplets (sorted by integer name when possible).
    triplet_dirs = [d for d in glob.glob(os.path.join(args_cli.triplets_root, "*"))
                    if os.path.isdir(d)]

    def _key(d):
        name = os.path.basename(d)
        return (0, int(name)) if name.isdigit() else (1, name)
    triplet_dirs.sort(key=_key)

    triplet_dirs = triplet_dirs[args_cli.start_at:]
    if args_cli.limit:
        triplet_dirs = triplet_dirs[: args_cli.limit]

    os.makedirs(args_cli.output_root, exist_ok=True)
    embed_target = (
        "positive_embeddings.pt" if args_cli.mode == "positive_inversion"
        else "null_embeddings.pt"
    )

    # Plan: figure out which triplets actually need work.
    plan = []
    for tdir in triplet_dirs:
        name = os.path.basename(tdir)
        out_dir = os.path.join(args_cli.output_root, f"triplet_{name}")
        already_done = os.path.exists(os.path.join(out_dir, embed_target))
        video_path = os.path.join(tdir, "video.mp4")
        if not os.path.exists(video_path):
            plan.append((name, tdir, video_path, None, None, "skip (no video.mp4)"))
            continue
        try:
            start, delta = parse_triplet_frames(tdir)
        except Exception as e:
            plan.append((name, tdir, video_path, None, None, f"skip ({e})"))
            continue
        if already_done:
            plan.append((name, tdir, video_path, start, delta, "skip (done)"))
            continue
        plan.append((name, tdir, video_path, start, delta, "RUN"))

    print(f"[batch] enumerated {len(plan)} triplets in {args_cli.triplets_root}:")
    for (name, _, _, start, delta, status) in plan:
        print(f"  triplet {name:>4} : start={start} delta={delta} -> {status}")
    runs_needed = [r for r in plan if r[-1] == "RUN"]
    print(f"[batch] {len(runs_needed)} need work")
    if args_cli.dry_run:
        print("[batch] --dry_run: not loading pipeline")
        return
    if not runs_needed:
        print("[batch] nothing to do")
        return

    # Load the pipeline once.
    base = Namespace(
        ckpt_dir=args_cli.ckpt_dir,
        init="empty",
        L_opt=16,
        L_pos=args_cli.L_pos,
        mode=args_cli.mode,
        loss="lpips",
        num_iters=300,
        lr=1e-2,
        null_inner_iters=args_cli.null_inner_iters,
        null_lr=args_cli.null_lr,
        inversion_guide_scale=args_cli.inversion_guide_scale,
        sampling_steps=args_cli.sampling_steps,
        guide_scale=args_cli.guide_scale,
        max_area=args_cli.max_area,
        shift=args_cli.shift,
        seed=args_cli.seed,
        log_every=25,
        validate=False,
        sampling_steps_val=40,
        guide_scale_val=1.0,
    )
    print(f"[batch] building pipeline once from {args_cli.ckpt_dir}…")
    pipe = ES.build_pipeline(base)

    # Manifest for bookkeeping.
    manifest_path = os.path.join(
        args_cli.output_root, f"manifest{args_cli.manifest_suffix}.json")
    manifest = []
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
        except Exception:
            manifest = []

    t_start = time.time()
    for i, (name, tdir, video_path, start, delta, status) in enumerate(plan):
        if status != "RUN":
            continue
        # Heuristic per triplet
        if args_cli.heuristic_source == "prompt_txt":
            prompt_path = os.path.join(tdir, "prompt.txt")
            heuristic = ""
            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    heuristic = f.read().strip()
        else:
            heuristic = ""

        a = build_per_triplet_args(
            base, name, video_path, heuristic, start, delta, args_cli.output_root)

        elapsed = time.time() - t_start
        print(
            f"\n[batch] === [{i+1}/{len(plan)}] triplet {name} === "
            f"start_index={start} delta_index={delta} heuristic={heuristic!r} "
            f"(elapsed {elapsed/60:.1f} min)"
        )
        try:
            ES.run_search(a, pipe)
            manifest.append({
                "triplet": name,
                "video": video_path,
                "start_index": start,
                "delta_index": delta,
                "heuristic": heuristic,
                "output_dir": a.output_dir,
                "status": "ok",
                "wall_s": time.time() - t_start,
            })
        except Exception as e:
            print(f"[batch] [{name}] FAILED: {e}")
            traceback.print_exc()
            manifest.append({
                "triplet": name,
                "video": video_path,
                "status": "failed",
                "error": str(e),
            })
        finally:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            torch.cuda.empty_cache()

    total = time.time() - t_start
    print(f"\n[batch] done. total wall time: {total/60:.1f} min")
    print(f"[batch] manifest at {manifest_path}")


if __name__ == "__main__":
    main()
