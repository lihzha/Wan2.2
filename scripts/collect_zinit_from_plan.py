"""Collect deterministic DDIM-style Wan inversion endpoints from a clip plan."""
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan_jsonl", required=True)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_root", default="runs/zinit_10k_ddim_droid")
    p.add_argument("--start_at", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_area", type=int, default=61440)
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--inversion_guide_scale", type=float, default=1.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--heuristic", default="")
    p.add_argument("--L_pos", type=int, default=1)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_plan(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_clip(video_path: str, start_frame: int, frames: int):
    import decord
    from PIL import Image

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    end = start_frame + frames
    if end > len(vr):
        raise ValueError(f"{video_path} has {len(vr)} frames; need [0,{end})")
    batch = vr.get_batch(list(range(start_frame, end))).asnumpy()
    return [Image.fromarray(f).convert("RGB") for f in batch]


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


def free_flat(z: torch.Tensor) -> torch.Tensor:
    return (z[:, 1:] if z.shape[1] > 1 else z).reshape(-1).float()


def stats(z: torch.Tensor) -> dict:
    flat = free_flat(z)
    q = torch.quantile(flat, torch.tensor([0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]))
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=True)),
        "rms": float(torch.sqrt(torch.mean(flat.square()))),
        "q001": float(q[0]),
        "q01": float(q[1]),
        "q05": float(q[2]),
        "q50": float(q[3]),
        "q95": float(q[4]),
        "q99": float(q[5]),
        "q999": float(q[6]),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "norm": float(flat.norm()),
    }


def main():
    args = parse_args()
    plan = read_plan(args.plan_jsonl)
    rows = plan[args.start_at:]
    if args.limit:
        rows = rows[: args.limit]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    shard_csv = out_root / f"manifest_shard_{args.start_at:06d}_{len(rows):05d}.csv"
    print(f"[zinit-plan] start_at={args.start_at} limit={args.limit} rows={len(rows)}")

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

    out_rows = []
    t0 = time.time()
    for local_i, row in enumerate(rows):
        idx = int(row["index"])
        sample_dir = out_root / "samples" / f"{idx:06d}_{row['sample_key']}"
        z_path = sample_dir / "z_init.pt"
        stats_path = sample_dir / "stats.json"
        if z_path.exists() and stats_path.exists() and not args.overwrite:
            status = "skip"
            with open(stats_path) as f:
                st = json.load(f)
        else:
            sample_dir.mkdir(parents=True, exist_ok=True)
            status = "ok"
            try:
                clip_pil = load_clip(row["video_path"], int(row["start_frame"]), int(row["frames"]))
                clip_pil, (oh, ow) = ES.fit_clip_to_pipeline(
                    clip_pil, pipe.vae_stride, pipe.patch_size, args.max_area)
                img0_pixels = ES.to_tensor_signed(clip_pil[0], device)
                target_pixels = ES.clip_to_tensor_signed(clip_pil, device)
                sigmas, mask2_full, seq_len, shape = ES._build_inversion_geometry(
                    pipe, oh, ow, int(row["frames"]), args.sampling_steps,
                    args.shift, device)
                with torch.no_grad():
                    z_I0_single = pipe.vae.encode([img0_pixels.unsqueeze(1)])[0]
                    z_I0_full = z_I0_single.expand(-1, shape[1], -1, -1).contiguous().float()
                    z_target = pipe.vae.encode([target_pixels])[0].float()
                traj = ES.compute_inversion_trajectory(
                    pipe=pipe,
                    z_target_video=z_target,
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
                st = stats(z_init)
                save_z = z_init.half() if args.dtype == "float16" else z_init.float()
                torch.save({
                    "z_init": save_z,
                    "z_dtype": args.dtype,
                    "sigmas": sigmas.detach().cpu(),
                    "plan": row,
                    "heuristic": args.heuristic,
                    "L_pos": args.L_pos,
                    "sampling_steps": args.sampling_steps,
                    "inversion_guide_scale": args.inversion_guide_scale,
                    "guide_scale": args.guide_scale,
                    "max_area": args.max_area,
                    "stats_unpinned": st,
                }, z_path)
                with open(stats_path, "w") as f:
                    json.dump({**row, **st, "z_init_path": str(z_path)}, f, indent=2)
                del traj, z_init, z_target, target_pixels
                torch.cuda.empty_cache()
            except Exception as e:
                status = "failed"
                st = {"error": str(e)}
                with open(sample_dir / "error.json", "w") as f:
                    json.dump({**row, "error": str(e)}, f, indent=2)

        out_row = {
            **row,
            "status": status,
            "z_init_path": str(z_path) if status != "failed" else "",
            "stats_path": str(stats_path) if status != "failed" else "",
            "wall_s": round(time.time() - t0, 3),
            **st,
        }
        out_rows.append(out_row)
        if local_i < 3 or (local_i + 1) % 25 == 0 or local_i == len(rows) - 1:
            print(
                f"[zinit-plan] {local_i + 1}/{len(rows)} idx={idx} "
                f"{row['sample_key']} status={status} "
                f"std={out_row.get('std', float('nan'))}")

    keys = list(out_rows[0].keys()) if out_rows else []
    with open(shard_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[zinit-plan] wrote {shard_csv}")


if __name__ == "__main__":
    main()
