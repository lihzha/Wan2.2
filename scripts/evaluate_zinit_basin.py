"""Replay optimized positive contexts from own, fresh, and cross z_init.

This tests whether a per-step context trajectory is tied to its inversion
endpoint even when endpoints all look approximately Gaussian marginally.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import embedding_search as ES  # noqa: E402


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", default="runs/droid_inv/train")
    p.add_argument("--triplets_root", default="data/droid_cache/train")
    p.add_argument("--names", default="ep0_v0,ep1_v0,ep2_v0,ep3_v0")
    p.add_argument("--fresh_seeds", default="0,1")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--out_dir", default="runs/zinit_basin_eval")
    p.add_argument("--cross_all", action="store_true",
                   help="Replay every context with every named sample's z_init.")
    return p.parse_args()


def load_positive(path: Path) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("positive_embeddings", "null_init", "sigmas", "z_init"):
        if key not in blob:
            raise KeyError(f"{path} missing {key}")
    return blob


def make_geometry(pipe, z_video: torch.Tensor, z_I0: torch.Tensor, device):
    C, Fz, Hz, Wz = z_video.shape
    z_I0_full = z_I0.to(device).float().expand(-1, Fz, -1, -1).contiguous()
    mask2_full = torch.ones(C, Fz, Hz, Wz, device=device, dtype=torch.float32)
    mask2_full[:, 0] = 0.0
    seq_len = (Fz * Hz * Wz) // (pipe.patch_size[1] * pipe.patch_size[2])
    seq_len = int((seq_len + pipe.sp_size - 1) // pipe.sp_size) * pipe.sp_size
    return z_I0_full, mask2_full, seq_len


def free_flat(z: torch.Tensor) -> torch.Tensor:
    return (z[:, 1:] if z.shape[1] > 1 else z).reshape(-1).float()


def main():
    args = parse_args()
    names = parse_csv(args.names)
    fresh_seeds = parse_csv_ints(args.fresh_seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[basin] names={names} fresh_seeds={fresh_seeds}")
    pipe = ES.build_pipeline(args)
    device = pipe.device
    pipe.model.to(device)

    positives = {}
    z_sources = {}
    targets = {}
    for name in names:
        pos_path = Path(args.runs_root) / name / "positive_embeddings.pt"
        cache_dir = Path(args.triplets_root) / name
        positives[name] = load_positive(pos_path)
        z_sources[name] = positives[name]["z_init"].float()
        z_video = torch.load(cache_dir / "z_video.pt", map_location="cpu", weights_only=False).float()
        z_I0 = torch.load(cache_dir / "z_I0.pt", map_location="cpu", weights_only=False).float()
        targets[name] = (z_video, z_I0)

    rows = []
    for ctx_name in names:
        blob = positives[ctx_name]
        z_video_cpu, z_I0_cpu = targets[ctx_name]
        z_video = z_video_cpu.to(device).float()
        z_I0_full, mask2_full, seq_len = make_geometry(pipe, z_video_cpu, z_I0_cpu, device)
        pos_embeds = [e.to(device).float() for e in blob["positive_embeddings"]]
        null_init = blob["null_init"].to(device).float()
        sigmas = blob["sigmas"].to(device).float()
        own_z = blob["z_init"].float()

        replay_sources: list[tuple[str, torch.Tensor]] = [("own", own_z)]
        for seed in fresh_seeds:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            fresh = torch.randn(tuple(own_z.shape), generator=gen, device=device).cpu()
            replay_sources.append((f"fresh_seed{seed}", fresh))
        cross_names = names if args.cross_all else [n for n in names if n != ctx_name][:1]
        for other_name in cross_names:
            if other_name == ctx_name:
                continue
            replay_sources.append((f"cross_{other_name}", z_sources[other_name]))

        for z_label, z_init_cpu in replay_sources:
            z_init = z_init_cpu.to(device).float()
            z_out = ES.regenerate_with_positive_embeds(
                pipe=pipe,
                z_init=z_init,
                z_I0_full=z_I0_full,
                mask2_full=mask2_full,
                seq_len=seq_len,
                sigmas=sigmas,
                null_context_list=[null_init],
                positive_embeds=pos_embeds,
                guide_scale=float(blob.get("guide_scale", 5.0)),
                param_dtype=pipe.param_dtype,
            )
            mse = F.mse_loss(z_out, z_video).item()
            mse_last = F.mse_loss(z_out[:, -1], z_video[:, -1]).item()
            z_rmse = torch.sqrt(torch.mean((free_flat(z_init_cpu) - free_flat(own_z)).square())).item()
            row = {
                "context_sample": ctx_name,
                "z_source": z_label,
                "latent_mse_full": mse,
                "latent_rmse_full": mse ** 0.5,
                "latent_mse_last": mse_last,
                "zinit_rmse_to_own": z_rmse,
            }
            rows.append(row)
            print("[basin] " + " ".join(f"{k}={v}" for k, v in row.items()))
            del z_out
            torch.cuda.empty_cache()

    with open(out_dir / "basin_replay.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[basin] wrote {out_dir / 'basin_replay.csv'}")


if __name__ == "__main__":
    main()
