"""Collect Wan positive-inversion endpoints z_init for DROID clips.

This is intentionally lighter than scripts/batch_inversion_droid.py: it runs
only the reverse-Euler inversion endpoint and skips per-step context
optimization. Repeating the same episode with different seeds tests whether
the inversion endpoint itself has a stochastic component.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import embedding_search as ES  # noqa: E402


def parse_csv_ints(text: str) -> list[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--droid_root",
                   default="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--episodes", default="0",
                   help="Comma-separated episode ids. If empty, use start_at/limit.")
    p.add_argument("--start_at", type=int, default=0)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--seeds", default="0,1,2,3",
                   help="Comma-separated seeds used only to probe determinism.")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_root", default="runs/zinit_probe_droid")
    p.add_argument("--heuristic", default="")
    p.add_argument("--L_pos", type=int, default=1)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--delta_index", type=int, default=32)
    p.add_argument("--max_area", type=int, default=61440)
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--inversion_guide_scale", type=float, default=1.0)
    p.add_argument("--guide_scale", type=float, default=5.0,
                   help="Stored for comparability; endpoint inversion uses inversion_guide_scale.")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def list_episode_ids(droid_root: str, split: str) -> list[int]:
    vid_dir = os.path.join(droid_root, "videos", split)
    ids = [int(n) for n in os.listdir(vid_dir)
           if n.isdigit() and os.path.isdir(os.path.join(vid_dir, n))]
    ids.sort()
    return ids


def load_droid_clip(args, episode: int):
    import decord
    from PIL import Image

    n_frames = args.delta_index + 1
    vid = os.path.join(args.droid_root, "videos", args.split, str(episode),
                       f"{args.view}.mp4")
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(vid)
    end = args.start_frame + n_frames
    if end > len(vr):
        raise ValueError(f"{vid} has {len(vr)} frames; need [0,{end})")
    frames_np = vr.get_batch(list(range(args.start_frame, end))).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in frames_np], vid


def force_context_len(context: torch.Tensor, length: int) -> torch.Tensor:
    if length <= 0:
        return context.contiguous()
    if context.shape[0] > length:
        return context[:length].contiguous()
    if context.shape[0] < length:
        pad = torch.zeros(length - context.shape[0], context.shape[1],
                          device=context.device, dtype=context.dtype)
        return torch.cat([context, pad], dim=0).contiguous()
    return context.contiguous()


def endpoint_stats(z_init: torch.Tensor) -> dict:
    # Frame 0 is the pinned encoded image, not free initial noise.
    z = z_init[:, 1:].float() if z_init.dim() == 4 and z_init.shape[1] > 1 else z_init.float()
    flat = z.reshape(-1)
    q = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=True)),
        "rms": float(torch.sqrt(torch.mean(flat.square()))),
        "min": float(flat.min()),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q50": float(q[2]),
        "q95": float(q[3]),
        "q99": float(q[4]),
        "max": float(flat.max()),
        "norm": float(flat.norm()),
    }


def main():
    args = parse_args()
    if args.delta_index < 4 or args.delta_index % 4 != 0:
        raise SystemExit("--delta_index must be a positive multiple of 4.")

    episodes = parse_csv_ints(args.episodes)
    if not episodes:
        episodes = list_episode_ids(args.droid_root, args.split)[args.start_at:]
        if args.limit:
            episodes = episodes[:args.limit]
    seeds = parse_csv_ints(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    out_split = Path(args.output_root) / args.split
    out_split.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.output_root) / "zinit_probe_summary.csv"

    print(f"[zinit-probe] episodes={episodes} seeds={seeds} -> {out_split}")
    pipe = ES.build_pipeline(args)
    device = pipe.device

    pipe.text_encoder.model.to(device)
    with torch.no_grad():
        context_pos = pipe.encode_prompt(args.heuristic).detach().float()
        context_pos = force_context_len(context_pos, args.L_pos)
        null_init = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    pipe.model.to(device)

    rows = []
    t0 = time.time()
    for ep in episodes:
        clip_pil, video_path = load_droid_clip(args, ep)
        clip_pil, (oh, ow) = ES.fit_clip_to_pipeline(
            clip_pil, pipe.vae_stride, pipe.patch_size, args.max_area)
        img0_pixels = ES.to_tensor_signed(clip_pil[0], device)
        target_clip_pixels = ES.clip_to_tensor_signed(clip_pil, device)

        sigmas, mask2_full, seq_len, shape = ES._build_inversion_geometry(
            pipe, oh, ow, args.delta_index + 1, args.sampling_steps,
            args.shift, device)
        with torch.no_grad():
            z_I0_single = pipe.vae.encode([img0_pixels.unsqueeze(1)])[0]
            z_I0_full = z_I0_single.expand(-1, shape[1], -1, -1).contiguous().float()
            z_target_video = pipe.vae.encode([target_clip_pixels])[0].float()

        for seed in seeds:
            run_dir = out_split / f"ep{ep}_v{args.view}" / f"seed{seed}"
            out_pt = run_dir / "z_init_probe.pt"
            if out_pt.exists() and not args.overwrite:
                print(f"[zinit-probe] skip existing {out_pt}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"[zinit-probe] ep{ep}_v{args.view} seed={seed}")
            traj = ES.compute_inversion_trajectory(
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
            z_init = traj[0].detach().cpu()
            stats = endpoint_stats(z_init)
            row = {
                "episode": ep,
                "view": args.view,
                "sample_key": f"ep{ep}_v{args.view}",
                "seed": seed,
                "path": str(out_pt),
                "shape": "x".join(str(x) for x in z_init.shape),
                "wall_s": round(time.time() - t0, 3),
                **stats,
            }
            torch.save({
                "z_init": z_init,
                "sigmas": sigmas.detach().cpu(),
                "episode": ep,
                "view": args.view,
                "seed": seed,
                "heuristic": args.heuristic,
                "L_pos": args.L_pos,
                "video_path": video_path,
                "start_frame": args.start_frame,
                "delta_index": args.delta_index,
                "sampling_steps": args.sampling_steps,
                "inversion_guide_scale": args.inversion_guide_scale,
                "guide_scale": args.guide_scale,
                "max_area": args.max_area,
                "stats_unpinned": stats,
            }, out_pt)
            with open(run_dir / "stats.json", "w") as f:
                json.dump(row, f, indent=2)
            rows.append(row)
            torch.cuda.empty_cache()

    if rows:
        write_header = not summary_path.exists() or args.overwrite
        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
    print(f"[zinit-probe] done; wrote {len(rows)} endpoints")


if __name__ == "__main__":
    main()
