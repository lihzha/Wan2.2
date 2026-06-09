"""Export decoded videos from an action-conditioned Wan overfit checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import embedding_search as ES  # noqa: E402
from models.action_conditioned_wan import (  # noqa: E402
    ActionConditionedWanConfig,
    ActionConditionedWanModel,
)
from train_adaptor import (  # noqa: E402
    TripletLatentDataset,
    build_latent_geometry,
    build_sigma_schedule,
    build_wan_pipeline,
)
from train_action_conditioned_wan import make_fixed_noise  # noqa: E402


def parse_seed_list(text: str | None) -> list[int]:
    if not text:
        return []
    seeds = []
    for part in text.replace(",", " ").split():
        part = part.strip()
        if part:
            seeds.append(int(part))
    return seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Training run dir containing ckpt_best.pt/config.json.")
    p.add_argument("--ckpt_path", default=None,
                   help="Explicit checkpoint. Default: <run_dir>/ckpt_best.pt.")
    p.add_argument("--triplets_root", default="data/droid_cache/train")
    p.add_argument("--overfit_one", default=None,
                   help="Sample name. Default comes from checkpoint args.")
    p.add_argument("--ckpt_dir", default=None,
                   help="Wan checkpoint dir. Default comes from checkpoint args.")
    p.add_argument("--output_dir", default=None,
                   help="Default: <run_dir>/videos_best.")
    p.add_argument("--sampling_steps", type=int, default=None,
                   help="Default comes from checkpoint args.")
    p.add_argument("--shift", type=float, default=None,
                   help="Default comes from checkpoint args.")
    p.add_argument("--guide_scale", type=float, default=None,
                   help="Default comes from checkpoint train_guide_scale.")
    p.add_argument("--seed", type=int, default=None,
                   help="Default comes from checkpoint args.")
    p.add_argument("--fixed_z_init_path", default=None,
                   help="Default: <run_dir>/fixed_z_init.pt when present.")
    p.add_argument("--eval_noise_mode", choices=["fixed", "random"],
                   default="fixed",
                   help="fixed uses fixed_z_init_path/run seed; random exports "
                   "one or more newly sampled z_init tensors from eval seeds.")
    p.add_argument("--eval_seeds", default=None,
                   help="Comma/space-separated random eval seeds. Used only "
                   "with --eval_noise_mode random.")
    p.add_argument("--eval_seed_start", type=int, default=None,
                   help="First random eval seed when --eval_seeds is omitted. "
                   "Default: checkpoint seed + 1000.")
    p.add_argument("--num_eval_noises", type=int, default=1,
                   help="Number of random eval noises when --eval_seeds is "
                   "omitted.")
    p.add_argument("--include_null", action="store_true",
                   help="Also decode a null-context CFG baseline.")
    p.add_argument("--fps", type=int, default=16)
    return p.parse_args()


def config_from_checkpoint(ckpt: dict) -> ActionConditionedWanConfig:
    allowed = {f.name for f in fields(ActionConditionedWanConfig)}
    cfg = dict(ckpt.get("config", {}))
    cfg = {k: v for k, v in cfg.items() if k in allowed}
    layers = cfg.get("side_adapter_layers")
    if layers is not None:
        cfg["side_adapter_layers"] = tuple(layers)
    return ActionConditionedWanConfig(**cfg)


def load_one_batch(triplets_root: str, name: str):
    sample = TripletLatentDataset(triplets_root, [name])[0]
    return {
        "name": [sample["name"]],
        "actions": sample["actions"].unsqueeze(0),
        "z_I0": sample["z_I0"].unsqueeze(0),
        "z_video": sample["z_video"].unsqueeze(0),
    }


@torch.no_grad()
def rollout_latent(
    pipe,
    model: ActionConditionedWanModel | None,
    batch: dict,
    sigmas: torch.Tensor,
    null_context: torch.Tensor,
    seq_len: int,
    mask2: torch.Tensor,
    guide_scale: float,
    fixed_noise: torch.Tensor,
):
    device = pipe.device
    actions = batch["actions"].to(device, non_blocking=True).float()
    z_i0 = batch["z_I0"].to(device, non_blocking=True).float()
    z_video = batch["z_video"].to(device, non_blocking=True).float()[0]
    _, fz, _, _ = z_video.shape
    z_i0_full = z_i0[0].expand(-1, fz, -1, -1).contiguous()
    mask2_zero = mask2[0]
    z = ES._apply_first_frame_pin(fixed_noise.float(), z_i0_full, mask2)

    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i].item()
        sigma_ip1 = sigmas[i + 1].item()
        timestep = ES._format_timestep(
            sigma_i, pipe.num_train_timesteps, mask2_zero, seq_len, device)
        v_uncond = ES._dit_velocity(
            pipe, z, sigma_i, [null_context], seq_len, mask2_zero,
            pipe.param_dtype)
        if model is None:
            v = v_uncond
        else:
            with torch.amp.autocast("cuda", dtype=pipe.param_dtype):
                v_cond = model(
                    x=[z],
                    t=timestep,
                    context=[null_context],
                    seq_len=seq_len,
                    actions=actions,
                )[0].float()
            v = v_uncond + guide_scale * (v_cond - v_uncond)
        z = ES._apply_first_frame_pin(
            z + (sigma_ip1 - sigma_i) * v, z_i0_full, mask2)
    return z, z_video


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else run_dir / "ckpt_best.pt"
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "videos_best"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("args", {})
    sample_name = args.overfit_one or train_args.get("overfit_one")
    ckpt_dir = args.ckpt_dir or train_args.get("ckpt_dir")
    sampling_steps = args.sampling_steps or int(train_args.get("sampling_steps", 1))
    shift = args.shift if args.shift is not None else float(train_args.get("shift", 5.0))
    guide_scale = (
        args.guide_scale if args.guide_scale is not None
        else float(train_args.get("train_guide_scale", 5.0))
    )
    seed = args.seed if args.seed is not None else int(train_args.get("seed", 0))
    if not sample_name:
        raise ValueError("sample name missing; pass --overfit_one")
    if not ckpt_dir:
        raise ValueError("Wan checkpoint dir missing; pass --ckpt_dir")

    batch = load_one_batch(args.triplets_root, sample_name)
    pipe = build_wan_pipeline(ckpt_dir)
    pipe.text_encoder.model.to(pipe.device)
    null_context = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    pipe.model.to(pipe.device)

    cfg = config_from_checkpoint(ckpt)
    model = ActionConditionedWanModel(pipe.model, cfg).to(pipe.device).eval()
    missing, unexpected = model.load_trainable_state_dict(ckpt["model"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")

    sigmas = build_sigma_schedule(pipe, sampling_steps, shift)
    seq_len, mask2 = build_latent_geometry(pipe, batch["z_video"].shape[1:])
    fixed_path = (
        Path(args.fixed_z_init_path) if args.fixed_z_init_path
        else run_dir / "fixed_z_init.pt"
    )
    if args.eval_noise_mode == "fixed":
        if fixed_path.exists():
            noise_specs = [(
                "fixed",
                seed,
                torch.load(
                    fixed_path, map_location=pipe.device,
                    weights_only=False).float(),
            )]
        else:
            noise_specs = [(
                f"seed{seed}",
                seed,
                make_fixed_noise(batch["z_video"].shape[1:], pipe.device, seed),
            )]
    else:
        eval_seeds = parse_seed_list(args.eval_seeds)
        if not eval_seeds:
            if args.num_eval_noises < 1:
                raise ValueError("--num_eval_noises must be >= 1")
            start = (
                int(args.eval_seed_start) if args.eval_seed_start is not None
                else seed + 1000
            )
            eval_seeds = list(range(start, start + args.num_eval_noises))
        noise_specs = [
            (
                f"seed{eval_seed}",
                eval_seed,
                make_fixed_noise(
                    batch["z_video"].shape[1:], pipe.device, eval_seed),
            )
            for eval_seed in eval_seeds
        ]

    z_video = batch["z_video"].to(pipe.device, non_blocking=True).float()[0]
    video_gt = pipe.vae.decode([z_video])[0]
    gt_path = out_dir / "ground_truth.mp4"
    ES.save_video(video_gt, str(gt_path), fps=args.fps)

    metrics = {
        "run_dir": str(run_dir),
        "ckpt_path": str(ckpt_path),
        "sample": sample_name,
        "mode": cfg.mode,
        "sampling_steps": sampling_steps,
        "guide_scale": guide_scale,
        "train_seed": seed,
        "eval_noise_mode": args.eval_noise_mode,
        "evals": [],
        "videos": {
            "ground_truth": str(gt_path),
        },
    }

    single_fixed_export = args.eval_noise_mode == "fixed" and len(noise_specs) == 1
    for noise_name, eval_seed, z_init in noise_specs:
        suffix = "" if single_fixed_export else f"_{noise_name}"
        z_pred, _ = rollout_latent(
            pipe=pipe,
            model=model,
            batch=batch,
            sigmas=sigmas,
            null_context=null_context,
            seq_len=seq_len,
            mask2=mask2,
            guide_scale=guide_scale,
            fixed_noise=z_init.to(pipe.device),
        )
        video_pred = pipe.vae.decode([z_pred])[0]
        sample_path = out_dir / f"sample{suffix}.mp4"
        latent_path = out_dir / f"sample_latent{suffix}.pt"
        ES.save_video(video_pred, str(sample_path), fps=args.fps)
        torch.save(z_pred.detach().cpu(), latent_path)

        eval_metrics = {
            "noise_name": noise_name,
            "eval_seed": eval_seed,
            "latent_mse": float(F.mse_loss(z_pred, z_video).item()),
            "z_pred_std": float(z_pred.std().item()),
            "z_target_std": float(z_video.std().item()),
            "z_init_std": float(z_init.std().item()),
            "videos": {
                "sample": str(sample_path),
                "sample_latent": str(latent_path),
            },
        }

        if args.include_null:
            z_null, _ = rollout_latent(
                pipe=pipe,
                model=None,
                batch=batch,
                sigmas=sigmas,
                null_context=null_context,
                seq_len=seq_len,
                mask2=mask2,
                guide_scale=guide_scale,
                fixed_noise=z_init.to(pipe.device),
            )
            video_null = pipe.vae.decode([z_null])[0]
            null_path = out_dir / f"null_only{suffix}.mp4"
            ES.save_video(video_null, str(null_path), fps=args.fps)
            eval_metrics["null_latent_mse"] = float(F.mse_loss(z_null, z_video).item())
            eval_metrics["videos"]["null_only"] = str(null_path)

        metrics["evals"].append(eval_metrics)

    if len(metrics["evals"]) == 1:
        first = metrics["evals"][0]
        metrics["seed"] = first["eval_seed"]
        metrics["latent_mse"] = first["latent_mse"]
        metrics["z_pred_std"] = first["z_pred_std"]
        metrics["z_target_std"] = first["z_target_std"]
        metrics["videos"].update(first["videos"])
        if "null_latent_mse" in first:
            metrics["null_latent_mse"] = first["null_latent_mse"]

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
