"""Sample DROID clips for large z_init curation.

Each row is one video clip: a unique episode/view plus a uniformly sampled
start frame that supports a delta_index+1 frame Wan clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--droid_root",
                   default="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--views", default="0")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--delta_index", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--full_scan", action="store_true",
                   help="Scan every candidate before sampling. Default shuffles "
                   "episodes and stops once enough valid clips are found, which "
                   "is much faster on GPFS.")
    return p.parse_args()


def video_length_from_annotation(root: str, split: str, ep: int) -> int | None:
    path = os.path.join(root, "annotation", split, f"{ep}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            n = json.load(f).get("video_length")
        return int(n) if n is not None else None
    except Exception:
        return None


def main():
    args = parse_args()
    views = parse_csv_ints(args.views)
    rng = random.Random(args.seed)
    n_frames = args.delta_index + 1
    vid_root = Path(args.droid_root) / "videos" / args.split
    episodes = sorted(int(p.name) for p in vid_root.iterdir()
                      if p.is_dir() and p.name.isdigit())

    candidates = []
    if not args.full_scan:
        rng.shuffle(episodes)
    for ep in episodes:
        n = video_length_from_annotation(args.droid_root, args.split, ep)
        if n is None or n < n_frames:
            continue
        for view in views:
            vid = vid_root / str(ep) / f"{view}.mp4"
            if vid.exists():
                candidates.append((ep, view, n, str(vid)))
        if not args.full_scan and len(candidates) >= args.num_samples:
            break
    if len(candidates) < args.num_samples:
        raise SystemExit(
            f"Only {len(candidates)} valid candidates for {args.num_samples} samples")

    if args.full_scan:
        rng.shuffle(candidates)
    rows = []
    for idx, (ep, view, n, vid) in enumerate(candidates[: args.num_samples]):
        start_frame = rng.randint(0, n - n_frames)
        rows.append({
            "index": idx,
            "split": args.split,
            "episode": ep,
            "view": view,
            "start_frame": start_frame,
            "delta_index": args.delta_index,
            "frames": n_frames,
            "video_length": n,
            "video_path": vid,
            "sample_key": f"ep{ep}_v{view}_s{start_frame}",
        })

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    out_csv = Path(args.out_csv) if args.out_csv else out_jsonl.with_suffix(".csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[plan] wrote {len(rows)} samples -> {out_jsonl}")
    print(f"[plan] views={views} split={args.split} seed={args.seed} delta={args.delta_index}")


if __name__ == "__main__":
    main()
