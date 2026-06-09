"""Create a capped high-action DROID window precompute plan.

The existing DROID cache stores one 33-frame window per episode. This planner
selects many windows from one camera view while respecting a storage cap. It
prioritizes windows with larger action variation so stationary prefixes are
unlikely to be selected.
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--droid_root",
                   default="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--candidate_stride", type=int, default=4,
                   help="Candidate start-frame stride before high-action ranking.")
    p.add_argument("--min_windows_per_episode", type=int, default=50,
                   help="Requested lower target. Used for reporting; the global "
                   "cap may force fewer selected episodes instead of fewer "
                   "windows per selected episode.")
    p.add_argument("--max_windows_per_episode", type=int, default=50,
                   help="Take at most this many high-action windows per episode.")
    p.add_argument("--cache_cap_gib", type=float, default=500.0)
    p.add_argument("--bytes_per_sample", type=int, default=240000,
                   help="Estimated cached bytes per window. 240k assumes fp16 "
                   "latents; current fp32 cache is about 465k.")
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--summary_json", default=None)
    p.add_argument("--sort_output", choices=["episode_start", "score"],
                   default="episode_start")
    return p.parse_args()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def episode_ids(droid_root: Path, split: str, view: int) -> list[int]:
    videos = droid_root / "videos" / split
    actions = droid_root / "actions" / split
    annotation = droid_root / "annotation" / split
    ids = []
    for vdir in videos.iterdir():
        if not vdir.is_dir() or not vdir.name.isdigit():
            continue
        ep = int(vdir.name)
        if (
            (vdir / f"{view}.mp4").exists()
            and (actions / f"{ep}.json").exists()
            and (annotation / f"{ep}.json").exists()
        ):
            ids.append(ep)
    return sorted(ids)


def action_window_score(actions: np.ndarray, start: int, frames: int) -> float:
    clip = actions[start:start + frames - 1].astype(np.float64, copy=False)
    if clip.shape[0] == 0:
        return 0.0
    base_delta = clip - clip[:1]
    cart_span = np.linalg.norm(base_delta[:, :6], axis=1).mean()
    if clip.shape[0] > 1:
        step_delta = np.diff(clip[:, :6], axis=0)
        step_motion = np.linalg.norm(step_delta, axis=1).mean()
    else:
        step_motion = 0.0
    grip = 0.0
    if clip.shape[1] > 6:
        grip = float(np.abs(clip[:, 6:] - clip[:1, 6:]).mean())
    return float(cart_span + 0.5 * step_motion + 0.1 * grip)


def quantiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {}
    arr = np.array(vals, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def main():
    args = parse_args()
    if args.frames < 5 or (args.frames - 1) % 4 != 0:
        raise SystemExit(f"--frames must be 4n+1, got {args.frames}")
    if args.candidate_stride <= 0:
        raise SystemExit("--candidate_stride must be positive")
    if args.max_windows_per_episode <= 0:
        raise SystemExit("--max_windows_per_episode must be positive")

    droid_root = Path(args.droid_root)
    max_samples = int(args.cache_cap_gib * (1024 ** 3) // args.bytes_per_sample)
    ids = episode_ids(droid_root, args.split, args.view)
    if args.max_episodes > 0:
        ids = ids[:args.max_episodes]

    actions_root = droid_root / "actions" / args.split
    ann_root = droid_root / "annotation" / args.split
    video_root = droid_root / "videos" / args.split

    selected_heap = []
    tie = 0
    stats = Counter()
    lengths = []
    candidates_per_episode = []
    selected_source_scores = []

    for ep in ids:
        stats["episodes_seen"] += 1
        ann_path = ann_root / f"{ep}.json"
        act_path = actions_root / f"{ep}.json"
        vid_path = video_root / str(ep) / f"{args.view}.mp4"
        try:
            ann = load_json(ann_path)
            actions_blob = load_json(act_path)
            actions = np.asarray(actions_blob["action"], dtype=np.float32)
        except Exception:
            stats["episodes_bad_json"] += 1
            continue
        video_length = int(ann.get("video_length") or 0)
        usable_length = min(video_length, int(actions.shape[0]) + 1)
        lengths.append(video_length)
        if usable_length < args.frames or usable_length < args.candidate_stride:
            stats["episodes_too_short"] += 1
            continue

        starts = list(range(0, usable_length - args.frames + 1, args.candidate_stride))
        if not starts:
            stats["episodes_no_candidates"] += 1
            continue
        scored = [
            (action_window_score(actions, s, args.frames), s)
            for s in starts
        ]
        scored.sort(reverse=True)
        top = scored[:args.max_windows_per_episode]
        candidates_per_episode.append(len(scored))
        stats["episodes_with_candidates"] += 1
        stats["candidate_windows"] += len(scored)
        selected_source_scores.extend(score for score, _ in top)

        for rank, (score, start) in enumerate(top):
            name = f"ep{ep}_v{args.view}_s{start:05d}"
            rec = {
                "name": name,
                "split": args.split,
                "episode_id": int(ep),
                "view": int(args.view),
                "start_frame": int(start),
                "frames": int(args.frames),
                "candidate_stride": int(args.candidate_stride),
                "video_length": int(video_length),
                "usable_length": int(usable_length),
                "action_score": float(score),
                "rank_in_episode": int(rank),
                "video_path": str(vid_path),
                "actions_path": str(act_path),
                "annotation_path": str(ann_path),
            }
            item = (score, tie, rec)
            tie += 1
            if len(selected_heap) < max_samples:
                heapq.heappush(selected_heap, item)
            elif score > selected_heap[0][0]:
                heapq.heapreplace(selected_heap, item)

    selected = [item[2] for item in selected_heap]
    if args.sort_output == "score":
        selected.sort(key=lambda r: (-r["action_score"], r["episode_id"], r["start_frame"]))
    else:
        selected.sort(key=lambda r: (r["episode_id"], r["start_frame"]))

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in selected:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    per_ep = Counter(r["episode_id"] for r in selected)
    selected_scores = [r["action_score"] for r in selected]
    summary = {
        "droid_root": str(droid_root),
        "split": args.split,
        "view": args.view,
        "frames": args.frames,
        "candidate_stride": args.candidate_stride,
        "requested_min_windows_per_episode": args.min_windows_per_episode,
        "max_windows_per_episode": args.max_windows_per_episode,
        "cache_cap_gib": args.cache_cap_gib,
        "bytes_per_sample": args.bytes_per_sample,
        "max_samples_by_cap": max_samples,
        "selected_windows": len(selected),
        "selected_episodes": len(per_ep),
        "estimated_cache_gib": len(selected) * args.bytes_per_sample / (1024 ** 3),
        "stats": dict(stats),
        "video_length_quantiles": quantiles(lengths),
        "candidate_windows_per_episode_quantiles": quantiles(candidates_per_episode),
        "selected_windows_per_episode_quantiles": quantiles(list(per_ep.values())),
        "selected_action_score_quantiles": quantiles(selected_scores),
        "episode_top_window_score_quantiles": quantiles(selected_source_scores),
        "out_jsonl": str(out_path),
    }
    summary_path = Path(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
