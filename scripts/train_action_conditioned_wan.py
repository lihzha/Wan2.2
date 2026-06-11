"""One-sample rollout overfit for action-conditioned Wan prototypes.

This is intentionally narrow: it trains the prototype wrapper in
models/action_conditioned_wan.py on one cached DROID clip with either a fixed
or freshly sampled initial noise and a configurable number of diffusion steps.
The fixed-noise path tests local authority in one basin; the fresh-noise path
tests whether the control channel can survive the train/eval behavior where
z_init changes from sample to sample under a fixed RNG stream.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
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
from models.action_conditioned_wan import (  # noqa: E402
    ActionConditionedWanConfig,
    ActionConditionedWanModel,
)
from train_adaptor import (  # noqa: E402
    TripletLatentDataset,
    build_latent_geometry,
    build_sigma_schedule,
    build_wan_pipeline,
    model_grad_checkpointing,
)


def parse_layers(text: str) -> tuple[int, ...] | None:
    if not text or text.lower() in ("all", "none"):
        return None if text.lower() != "none" else tuple()
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            if b < a:
                raise ValueError(f"bad layer range {part!r}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return tuple(dict.fromkeys(out))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets_root", default="data/droid_cache/train")
    p.add_argument("--overfit_one", default="ep0_v0")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["pre_context", "side_adapter"], required=True)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_len", type=int, default=32)
    p.add_argument("--action_repr", choices=["delta", "raw"], default="delta")
    p.add_argument("--action_tokens", type=int, default=8)
    p.add_argument("--action_hidden", type=int, default=512)
    p.add_argument("--action_heads", type=int, default=4)
    p.add_argument("--pre_context_tokens", type=int, default=8)
    p.add_argument("--pre_context_heads", type=int, default=0)
    p.add_argument("--side_adapter_layers", default="24-29")
    p.add_argument("--side_adapter_hidden", type=int, default=512)
    p.add_argument("--side_adapter_heads", type=int, default=8)
    p.add_argument("--sampling_steps", type=int, default=1)
    p.add_argument("--noise_mode", choices=["fixed", "fresh"], default="fixed",
                   help="fixed reuses one z_init for every optimizer step; "
                   "fresh draws a new Gaussian z_init each optimizer step. "
                   "The saved fixed_z_init.pt is always used for final eval.")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--train_guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--ckpt_interval", type=int, default=100)
    p.add_argument("--save_optimizer", action="store_true",
                   help="Include AdamW state in checkpoints. Disabled by "
                   "default for overfit diagnostics to keep files small.")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false",
                   dest="gradient_checkpointing")
    return p.parse_args()


def cosine_lr(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return max(1e-8, (step + 1) / warmup)
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
    return min_ratio + (1.0 - min_ratio) * cos


def make_fixed_noise(shape, device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)


def rollout_loss(
    pipe,
    model: ActionConditionedWanModel,
    batch: dict,
    sigmas: torch.Tensor,
    null_context: torch.Tensor,
    seq_len: int,
    mask2: torch.Tensor,
    guide_scale: float,
    z_init_noise: torch.Tensor,
):
    device = pipe.device
    actions = batch["actions"].to(device, non_blocking=True).float()
    z_i0 = batch["z_I0"].to(device, non_blocking=True).float()
    z_video_b = batch["z_video"].to(device, non_blocking=True).float()
    batch_size = z_video_b.shape[0]
    if z_init_noise.dim() == z_video_b.dim() - 1:
        if batch_size != 1:
            raise ValueError(
                "unbatched z_init_noise is only valid with batch_size=1")
        z_init_noise_b = z_init_noise.unsqueeze(0)
    else:
        z_init_noise_b = z_init_noise
    if z_init_noise_b.shape != z_video_b.shape:
        raise ValueError(
            f"z_init_noise shape {tuple(z_init_noise_b.shape)} must match "
            f"z_video shape {tuple(z_video_b.shape)}")

    _, _, fz, _, _ = z_video_b.shape
    if z_i0.shape[2] == 1:
        z_i0_full = z_i0.expand(-1, -1, fz, -1, -1).contiguous()
    elif z_i0.shape[2] == fz:
        z_i0_full = z_i0.contiguous()
    else:
        raise ValueError(
            f"z_I0 temporal shape {tuple(z_i0.shape)} is not 1 or {fz}")
    z = ES._apply_first_frame_pin(z_init_noise_b, z_i0_full, mask2.unsqueeze(0))
    mask2_zero = mask2[0]
    context = [null_context for _ in range(batch_size)]

    for i in range(len(sigmas) - 1):
        sigma_i = sigmas[i].item()
        sigma_ip1 = sigmas[i + 1].item()
        timestep = ES._format_timestep(
            sigma_i, pipe.num_train_timesteps, mask2_zero, seq_len, device)
        timestep = timestep.expand(batch_size, -1).contiguous()
        z_list = [z_b for z_b in z]
        with torch.amp.autocast("cuda", dtype=pipe.param_dtype):
            v_cond = model(
                x=z_list,
                t=timestep,
                context=context,
                seq_len=seq_len,
                actions=actions,
            )
            v_cond = torch.stack(v_cond, dim=0).float()
        if abs(guide_scale - 1.0) > 1e-6:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=pipe.param_dtype):
                    v_uncond = pipe.model(
                        [z_b.detach() for z_b in z],
                        t=timestep,
                        context=context,
                        seq_len=seq_len,
                    )
                v_uncond = torch.stack(v_uncond, dim=0).float()
            v = v_uncond + guide_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = ES._apply_first_frame_pin(
            z + (sigma_ip1 - sigma_i) * v, z_i0_full, mask2.unsqueeze(0))

    loss = F.mse_loss(z, z_video_b)
    logs = {
        "loss": float(loss.item()),
        "latent_mse": float(loss.item()),
        "z_pred_std": float(z.detach().std().item()),
        "z_target_std": float(z_video_b.detach().std().item()),
        "z_init_std": float(z_init_noise_b.detach().std().item()),
    }
    return loss, logs


@torch.no_grad()
def eval_rollout(*args, **kwargs):
    model = args[1]
    was_training = model.training
    model.eval()
    loss, logs = rollout_loss(*args, **kwargs)
    if was_training:
        model.train()
    return loss, logs


def save_ckpt(path: str, model: ActionConditionedWanModel, optim, args, step, logs):
    payload = {
        "step": step,
        "args": vars(args),
        "config": model.config.__dict__,
        "model": model.trainable_state_dict(),
        "logs": logs,
    }
    if args.save_optimizer:
        payload["optimizer"] = optim.state_dict()
    torch.save(payload, path)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ds = TripletLatentDataset(args.triplets_root, [args.overfit_one])
    batch = ds[0]
    batch = {
        "name": [batch["name"]],
        "actions": batch["actions"].unsqueeze(0),
        "z_I0": batch["z_I0"].unsqueeze(0),
        "z_video": batch["z_video"].unsqueeze(0),
    }

    pipe = build_wan_pipeline(args.ckpt_dir)
    pipe.text_encoder.model.to(pipe.device)
    with torch.no_grad():
        null_context = pipe.encode_prompt("").detach().float()
    pipe.text_encoder.model.cpu()
    torch.cuda.empty_cache()
    pipe.model.to(pipe.device)

    sigmas = build_sigma_schedule(pipe, args.sampling_steps, args.shift)
    seq_len, mask2 = build_latent_geometry(pipe, batch["z_video"].shape[1:])
    eval_noise = make_fixed_noise(
        batch["z_video"].shape[1:], pipe.device, args.seed)
    torch.save(eval_noise.detach().cpu(), out_dir / "fixed_z_init.pt")

    cfg = ActionConditionedWanConfig(
        mode=args.mode,
        action_dim=args.action_dim,
        action_len=args.action_len,
        action_repr=args.action_repr,
        action_tokens=args.action_tokens,
        action_hidden=args.action_hidden,
        action_heads=args.action_heads,
        pre_context_tokens=args.pre_context_tokens,
        pre_context_heads=args.pre_context_heads or None,
        side_adapter_layers=parse_layers(args.side_adapter_layers),
        side_adapter_hidden=args.side_adapter_hidden,
        side_adapter_heads=args.side_adapter_heads or None,
    )
    model = ActionConditionedWanModel(pipe.model, cfg).to(pipe.device)
    trainable = list(model.adapter_parameters())
    optim = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999))

    print(f"[action-overfit] sample={args.overfit_one} mode={args.mode} "
          f"steps={args.sampling_steps} train_steps={args.total_steps}")
    print(f"[action-overfit] seq_len={seq_len} z_shape={tuple(batch['z_video'].shape[1:])} "
          f"trainable_params={sum(p.numel() for p in trainable)/1e6:.2f}M")
    print(f"[action-overfit] noise_mode={args.noise_mode} "
          f"eval_noise_seed={args.seed} guide_scale={args.train_guide_scale}")

    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step", "loss", "latent_mse", "z_pred_std", "z_target_std",
                "z_init_std",
                "lr", "grad_norm", "wall_s",
            ],
        )
        writer.writeheader()

    best = float("inf")
    t0 = time.time()
    ckpt_ctx = (
        model_grad_checkpointing(pipe) if args.gradient_checkpointing
        else contextlib.nullcontext()
    )
    model.train()
    with ckpt_ctx:
        for step in range(args.total_steps):
            scale = cosine_lr(
                step, args.total_steps, args.warmup_steps, args.lr_min_ratio)
            for group in optim.param_groups:
                group["lr"] = args.lr * scale

            if args.noise_mode == "fresh":
                train_noise = torch.randn_like(eval_noise)
            else:
                train_noise = eval_noise

            loss, logs = rollout_loss(
                pipe=pipe,
                model=model,
                batch=batch,
                sigmas=sigmas,
                null_context=null_context,
                seq_len=seq_len,
                mask2=mask2,
                guide_scale=args.train_guide_scale,
                z_init_noise=train_noise,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()

            logs["lr"] = optim.param_groups[0]["lr"]
            logs["grad_norm"] = float(grad_norm.item())
            logs["wall_s"] = round(time.time() - t0, 3)
            if logs["loss"] < best:
                best = logs["loss"]
                save_ckpt(out_dir / "ckpt_best.pt", model, optim, args, step, logs)
            if (
                (step + 1) % args.ckpt_interval == 0
                or step == args.total_steps - 1
            ):
                save_ckpt(out_dir / "ckpt_latest.pt", model, optim, args, step, logs)
            if step == 0 or (step + 1) % args.log_interval == 0:
                print(
                    f"[action-overfit] step={step + 1:05d} "
                    f"loss={logs['loss']:.6f} grad={logs['grad_norm']:.3f} "
                    f"lr={logs['lr']:.2e}")
                with open(log_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "step", "loss", "latent_mse", "z_pred_std",
                            "z_target_std", "z_init_std", "lr",
                            "grad_norm", "wall_s",
                        ],
                    )
                    writer.writerow({"step": step + 1, **logs})

    with torch.no_grad():
        _, eval_logs = eval_rollout(
            pipe, model, batch, sigmas, null_context, seq_len, mask2,
            args.train_guide_scale, eval_noise)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "best_loss": best,
            "best_train_loss": best,
            "final_eval": eval_logs,
            "eval_noise_seed": args.seed,
        }, f, indent=2)
    print(f"[action-overfit] done best={best:.6f} final={eval_logs['loss']:.6f}")


if __name__ == "__main__":
    main()
