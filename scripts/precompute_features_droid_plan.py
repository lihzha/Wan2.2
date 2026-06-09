"""Materialize a DROID window plan into Wan VAE latent cache files."""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from embedding_search import (  # noqa: E402
    clip_to_tensor_signed,
    fit_clip_to_pipeline,
    to_tensor_signed,
)
from wan.configs.wan_ti2v_5B import ti2v_5B  # noqa: E402
from wan.modules.vae2_2 import Wan2_2_VAE  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan_jsonl", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_root", default="data/droid_cache_windows")
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--save_dtype", choices=["float32", "float16", "bfloat16"],
                   default="float16")
    p.add_argument("--max_area", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_plan(path: str, shard_index: int, num_shards: int, limit: int):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard_index must be in [0, num_shards)")
    n = len(rows)
    lo = n * shard_index // num_shards
    hi = n * (shard_index + 1) // num_shards
    rows = rows[lo:hi]
    if limit > 0:
        rows = rows[:limit]
    rows.sort(key=lambda r: (r["split"], r["episode_id"], r["view"], r["start_frame"]))
    return rows, lo, hi, n


def cache_done(out_dir: Path) -> bool:
    return all((out_dir / f).exists() for f in ("z_I0.pt", "z_video.pt", "actions.npy", "meta.json"))


def cast_for_save(t: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "float16":
        return t.half()
    if dtype == "bfloat16":
        return t.bfloat16()
    return t.float()


def read_actions(path: str) -> np.ndarray:
    with open(path) as f:
        return np.asarray(json.load(f)["action"], dtype=np.float32)


def main():
    args = parse_args()
    rows, lo, hi, total = load_plan(
        args.plan_jsonl, args.shard_index, args.num_shards, args.limit)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[droid-plan-cache] plan={args.plan_jsonl} total={total} "
        f"shard={args.shard_index}/{args.num_shards} rows={len(rows)} "
        f"range=[{lo},{hi}) out={out_root} dtype={args.save_dtype}",
        flush=True,
    )
    if not rows:
        return

    import decord
    from PIL import Image
    decord.bridge.set_bridge("native")

    config = ti2v_5B
    device = torch.device(args.device)
    print(f"[droid-plan-cache] loading VAE {args.ckpt_dir}/{config.vae_checkpoint}", flush=True)
    vae = Wan2_2_VAE(
        vae_pth=os.path.join(args.ckpt_dir, config.vae_checkpoint),
        device=device,
    )

    n_ok, n_skip, n_fail = 0, 0, 0
    group_key = lambda r: (r["split"], r["episode_id"], r["view"], r["video_path"])
    for (_, ep, view, video_path), group_iter in groupby(rows, key=group_key):
        group = list(group_iter)
        try:
            vr = decord.VideoReader(video_path)
            actions = read_actions(group[0]["actions_path"])
        except Exception as exc:
            n_fail += len(group)
            print(f"[droid-plan-cache] FAIL load ep={ep} view={view}: {exc}", flush=True)
            continue

        for rec in group:
            out_dir = out_root / rec["split"] / rec["name"]
            if cache_done(out_dir) and not args.force:
                n_ok += 1
                continue
            start = int(rec["start_frame"])
            frames = int(rec["frames"])
            end = start + frames
            if len(vr) < end:
                n_skip += 1
                continue
            try:
                idxs = list(range(start, end))
                frames_np = vr.get_batch(idxs).asnumpy()
                clip_pil = [Image.fromarray(f).convert("RGB") for f in frames_np]
                h0, w0 = frames_np.shape[1], frames_np.shape[2]
                max_area = args.max_area if args.max_area > 0 else h0 * w0
                clip_pil, (oh, ow) = fit_clip_to_pipeline(
                    clip_pil, config.vae_stride, config.patch_size, max_area)
                clip_pixels = clip_to_tensor_signed(clip_pil, device)
                img0_pixels = to_tensor_signed(clip_pil[0], device)

                a_start = start
                a_end = start + frames - 1
                if a_end > actions.shape[0]:
                    pad = np.repeat(actions[-1:], a_end - actions.shape[0], axis=0)
                    actions_use = np.concatenate([actions, pad], axis=0)
                else:
                    actions_use = actions
                actions_clip = actions_use[a_start:a_end]

                with torch.no_grad():
                    z_i0 = vae.encode([img0_pixels.unsqueeze(1)])[0].float().cpu()
                    z_video = vae.encode([clip_pixels])[0].float().cpu()

                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(cast_for_save(z_i0, args.save_dtype), out_dir / "z_I0.pt")
                torch.save(cast_for_save(z_video, args.save_dtype), out_dir / "z_video.pt")
                np.save(out_dir / "actions.npy", actions_clip)
                meta = {
                    "source": "droid_ctrl_world",
                    "episode_id": int(ep),
                    "view": int(view),
                    "split": rec["split"],
                    "start_frame": start,
                    "frames": frames,
                    "video_length": int(rec.get("video_length", len(vr))),
                    "usable_length": int(rec.get("usable_length", len(vr))),
                    "action_score": float(rec.get("action_score", 0.0)),
                    "rank_in_episode": int(rec.get("rank_in_episode", -1)),
                    "candidate_stride": int(rec.get("candidate_stride", -1)),
                    "oh": int(oh),
                    "ow": int(ow),
                    "max_area": int(max_area),
                    "save_dtype": args.save_dtype,
                    "z_I0_shape": list(z_i0.shape),
                    "z_video_shape": list(z_video.shape),
                    "action_shape": list(actions_clip.shape),
                    "video_path": video_path,
                }
                with open(out_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2, sort_keys=True)
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                print(
                    f"[droid-plan-cache] FAIL {rec['name']} "
                    f"start={start}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            if (n_ok + n_skip + n_fail) % 100 == 0:
                print(
                    f"[droid-plan-cache] progress ok={n_ok} "
                    f"skip={n_skip} fail={n_fail}",
                    flush=True,
                )
                torch.cuda.empty_cache()

    print(f"[droid-plan-cache] done ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
    if n_skip or n_fail:
        raise SystemExit(
            f"planned cache shard incomplete: ok={n_ok} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
