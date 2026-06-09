"""Stochastic reverse-flow inversion probes for DROID clips.

Wan's existing DDIM-style inversion is an ODE inverse: for fixed sample,
context, scheduler, and weights it returns a single endpoint. This script adds
controlled noise during the reverse pass to create a same-sample endpoint cloud.

For each stochastic pivot trajectory, the optional context optimization reuses
the existing positive-inversion inner loop. That validates whether the sampled
endpoint is merely noisy, or can still serve as a usable inversion endpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import embedding_search as ES  # noqa: E402


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--droid_root",
                   default="/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--episodes", default="0",
                   help="Comma-separated episode ids.")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--etas", default="0.0,0.3,0.6")
    p.add_argument("--optimize_etas", default="0.6",
                   help="Comma-separated eta values for positive-context optimization.")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_root", default="runs/stochastic_zinit_droid")
    p.add_argument("--heuristic", default="")
    p.add_argument("--L_pos", type=int, default=1)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--delta_index", type=int, default=32)
    p.add_argument("--max_area", type=int, default=61440)
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--inversion_guide_scale", type=float, default=1.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--null_inner_iters", type=int, default=10)
    p.add_argument("--null_lr", type=float, default=1e-2)
    p.add_argument("--save_trajectory", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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


def free_flat(z: torch.Tensor) -> torch.Tensor:
    return (z[:, 1:] if z.shape[1] > 1 else z).reshape(-1).float()


def endpoint_stats(z_init: torch.Tensor) -> dict:
    flat = free_flat(z_init)
    q = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.50, 0.95, 0.99]))
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=True)),
        "rms": float(torch.sqrt(torch.mean(flat.square()))),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q50": float(q[2]),
        "q95": float(q[3]),
        "q99": float(q[4]),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "norm": float(flat.norm()),
    }


def endpoint_pair(a: torch.Tensor, b: torch.Tensor) -> dict:
    aa = free_flat(a)
    bb = free_flat(b)
    diff = aa - bb
    cos = torch.dot(aa, bb) / (aa.norm() * bb.norm() + 1e-12)
    return {
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "mse": float(torch.mean(diff.square())),
        "cosine": float(cos),
    }


def compute_stochastic_inversion_trajectory(
    pipe,
    z_target_video,
    z_I0_full,
    mask2_full,
    seq_len,
    sigmas,
    context_list,
    inversion_guide_scale,
    context_null_list,
    param_dtype,
    eta: float,
    seed: int,
):
    """Reverse-Euler trajectory with optional Gaussian reverse increments.

    The increment scale sqrt(sigma_i^2 - sigma_{i+1}^2) is a diagnostic
    convention, not an exact Wan training SDE. eta=0 is the existing
    deterministic DDIM-style inversion.
    """
    N = len(sigmas) - 1
    traj = [None] * (N + 1)
    traj[N] = ES._apply_first_frame_pin(
        z_target_video.detach().float(), z_I0_full, mask2_full)
    use_cfg = abs(inversion_guide_scale - 1.0) > 1e-6
    mask2_zero = mask2_full[0]
    gen = torch.Generator(device=pipe.device)
    gen.manual_seed(int(seed))

    with torch.no_grad():
        for k in range(N):
            i = N - 1 - k
            sigma_i = sigmas[i].item()
            sigma_ip1 = sigmas[i + 1].item()
            v = ES._dit_velocity(
                pipe, traj[i + 1], sigma_i, context_list, seq_len,
                mask2_zero, param_dtype)
            if use_cfg:
                v_null = ES._dit_velocity(
                    pipe, traj[i + 1], sigma_i, context_null_list, seq_len,
                    mask2_zero, param_dtype)
                v = v_null + inversion_guide_scale * (v - v_null)
            step = (sigma_i - sigma_ip1) * v
            if eta > 0:
                noise_scale = eta * math.sqrt(max(sigma_i * sigma_i - sigma_ip1 * sigma_ip1, 0.0))
                eps = torch.randn(
                    traj[i + 1].shape, device=pipe.device, dtype=torch.float32,
                    generator=gen)
                step = step + noise_scale * eps
            traj[i] = ES._apply_first_frame_pin(
                traj[i + 1] + step, z_I0_full, mask2_full).detach()
    return traj


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.delta_index < 4 or args.delta_index % 4 != 0:
        raise SystemExit("--delta_index must be a positive multiple of 4.")
    episodes = parse_csv_ints(args.episodes)
    seeds = parse_csv_ints(args.seeds)
    etas = parse_csv_floats(args.etas)
    optimize_etas = set(parse_csv_floats(args.optimize_etas))
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[stoch-inv] episodes={episodes} seeds={seeds} etas={etas} optimize={sorted(optimize_etas)}")
    pipe = ES.build_pipeline(args)
    device = pipe.device

    pipe.text_encoder.model.to(device)
    with torch.no_grad():
        context_pos = force_context_len(
            pipe.encode_prompt(args.heuristic).detach().float(), args.L_pos)
        null_init = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    pipe.model.to(device)

    endpoint_rows = []
    recon_rows = []
    pair_records = []
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

        for eta in etas:
            eta_key = f"eta{eta:g}".replace(".", "p")
            for seed in seeds:
                run_dir = out_root / args.split / f"ep{ep}_v{args.view}" / eta_key / f"seed{seed}"
                out_pt = run_dir / "stochastic_inversion.pt"
                if out_pt.exists() and not args.overwrite:
                    print(f"[stoch-inv] skip existing {out_pt}")
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)

                print(f"[stoch-inv] ep{ep}_v{args.view} eta={eta:g} seed={seed}")
                traj = compute_stochastic_inversion_trajectory(
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
                    eta=eta,
                    seed=seed,
                )
                z_init = traj[0].detach().cpu()
                stats = endpoint_stats(z_init)
                endpoint_row = {
                    "episode": ep,
                    "view": args.view,
                    "sample_key": f"ep{ep}_v{args.view}",
                    "eta": eta,
                    "seed": seed,
                    "shape": "x".join(str(x) for x in z_init.shape),
                    "path": str(out_pt),
                    "wall_s": round(time.time() - t0, 3),
                    **stats,
                }
                endpoint_rows.append(endpoint_row)
                pair_records.append((endpoint_row, z_init))

                blob = {
                    "z_init": z_init,
                    "sigmas": sigmas.detach().cpu(),
                    "episode": ep,
                    "view": args.view,
                    "seed": seed,
                    "eta": eta,
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
                }
                if args.save_trajectory:
                    blob["trajectory"] = [t.detach().cpu() for t in traj]
                torch.save(blob, out_pt)
                with open(run_dir / "stats.json", "w") as f:
                    json.dump(endpoint_row, f, indent=2)

                if eta in optimize_etas:
                    pos_embeds, _ = ES.optimize_positive_embeddings(
                        pipe=pipe,
                        traj=traj,
                        z_I0_full=z_I0_full,
                        mask2_full=mask2_full,
                        seq_len=seq_len,
                        sigmas=sigmas,
                        null_context_list=[null_init],
                        positive_init=context_pos,
                        inner_iters=args.null_inner_iters,
                        pos_lr=args.null_lr,
                        guide_scale=args.guide_scale,
                        param_dtype=pipe.param_dtype,
                        log_dir=str(run_dir),
                    )
                    z_recon = ES.regenerate_with_positive_embeds(
                        pipe=pipe,
                        z_init=traj[0],
                        z_I0_full=z_I0_full,
                        mask2_full=mask2_full,
                        seq_len=seq_len,
                        sigmas=sigmas,
                        null_context_list=[null_init],
                        positive_embeds=pos_embeds,
                        guide_scale=args.guide_scale,
                        param_dtype=pipe.param_dtype,
                    )
                    full_mse = F.mse_loss(z_recon, z_target_video).item()
                    last_mse = F.mse_loss(z_recon[:, -1], z_target_video[:, -1]).item()
                    recon_row = {
                        "episode": ep,
                        "view": args.view,
                        "sample_key": f"ep{ep}_v{args.view}",
                        "eta": eta,
                        "seed": seed,
                        "latent_mse_full": full_mse,
                        "latent_rmse_full": full_mse ** 0.5,
                        "latent_mse_last": last_mse,
                        "path": str(run_dir / "positive_embeddings.pt"),
                    }
                    recon_rows.append(recon_row)
                    torch.save({
                        "positive_embeddings": [e.detach().cpu() for e in pos_embeds],
                        "null_init": null_init.detach().cpu(),
                        "sigmas": sigmas.detach().cpu(),
                        "guide_scale": args.guide_scale,
                        "heuristic": args.heuristic,
                        "z_init": z_init,
                        "eta": eta,
                        "seed": seed,
                        "reconstruction": recon_row,
                    }, run_dir / "positive_embeddings.pt")
                    with open(run_dir / "reconstruction.json", "w") as f:
                        json.dump(recon_row, f, indent=2)
                    print(
                        f"[stoch-inv] recon ep{ep} eta={eta:g} seed={seed}: "
                        f"mse={full_mse:.6f} last={last_mse:.6f}"
                    )
                    del z_recon

                del traj
                torch.cuda.empty_cache()

    pair_rows = []
    for i in range(len(pair_records)):
        row_i, z_i = pair_records[i]
        for j in range(i + 1, len(pair_records)):
            row_j, z_j = pair_records[j]
            if row_i["shape"] != row_j["shape"]:
                continue
            same_sample = row_i["sample_key"] == row_j["sample_key"]
            same_eta = float(row_i["eta"]) == float(row_j["eta"])
            p = endpoint_pair(z_i, z_j)
            pair_rows.append({
                "sample_a": row_i["sample_key"],
                "sample_b": row_j["sample_key"],
                "eta_a": row_i["eta"],
                "eta_b": row_j["eta"],
                "seed_a": row_i["seed"],
                "seed_b": row_j["seed"],
                "same_sample": int(same_sample),
                "same_eta": int(same_eta),
                **p,
            })

    write_csv(out_root / "endpoint_stats.csv", endpoint_rows)
    write_csv(out_root / "endpoint_pairs.csv", pair_rows)
    write_csv(out_root / "reconstruction.csv", recon_rows)
    summary = {
        "episodes": episodes,
        "seeds": seeds,
        "etas": etas,
        "optimize_etas": sorted(optimize_etas),
        "n_endpoints": len(endpoint_rows),
        "n_reconstructions": len(recon_rows),
    }
    with open(out_root / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
