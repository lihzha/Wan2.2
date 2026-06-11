"""Run one DROID action-conditioned optimizer step for GPU memory profiling."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from models.action_conditioned_wan import (  # noqa: E402
    ActionConditionedWanConfig,
    ActionConditionedWanModel,
)
from train_action_conditioned_wan import (  # noqa: E402
    parse_layers,
    rollout_loss,
)
from train_action_conditioned_wan_droid import (  # noqa: E402
    LazyTripletLatentDataset,
    make_noise_for_batch,
    names_from_manifest,
)
from train_adaptor import (  # noqa: E402
    TripletLatentDataset,
    build_latent_geometry,
    build_sigma_schedule,
    build_wan_pipeline,
    collate,
    model_grad_checkpointing,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets_root", default="data/droid_cache/train")
    p.add_argument("--train_manifest_jsonl", default=None)
    p.add_argument("--ckpt_dir", default="Wan2.2-TI2V-5B")
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_len", type=int, default=32)
    p.add_argument("--action_repr", choices=["delta", "raw"], default="delta")
    p.add_argument("--action_tokens", type=int, default=8)
    p.add_argument("--action_hidden", type=int, default=512)
    p.add_argument("--action_heads", type=int, default=4)
    p.add_argument("--pre_context_tokens", type=int, default=8)
    p.add_argument("--pre_context_heads", type=int, default=0)
    p.add_argument("--side_adapter_layers", default="0-29")
    p.add_argument("--side_adapter_hidden", type=int, default=512)
    p.add_argument("--side_adapter_heads", type=int, default=8)
    p.add_argument("--sampling_steps", type=int, default=25)
    p.add_argument("--noise_mode", choices=["fresh", "fixed"], default="fresh")
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--train_guide_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false",
                   dest="gradient_checkpointing")
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def mib(value: int) -> float:
    return round(float(value) / (1024.0 * 1024.0), 1)


def write_result(path: str | None, payload: dict):
    line = json.dumps(payload, sort_keys=True)
    print(f"[batch-profile] {line}", flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(line + "\n")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.train_manifest_jsonl:
        names = names_from_manifest(args.train_manifest_jsonl)
        ds_cls = LazyTripletLatentDataset
    else:
        names = [p.name for p in Path(args.triplets_root).iterdir() if p.is_dir()]
        names.sort()
        ds_cls = TripletLatentDataset
    if args.max_train_samples > 0:
        names = names[:args.max_train_samples]
    if len(names) < args.batch_size:
        raise ValueError(
            f"need at least {args.batch_size} samples, found {len(names)}")

    train_ds = ds_cls(args.triplets_root, names[:max(args.batch_size, 1)])
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    batch = next(iter(loader))

    t0 = time.time()
    try:
        pipe = build_wan_pipeline(args.ckpt_dir)
        pipe.text_encoder.model.to(pipe.device)
        with torch.no_grad():
            null_context = pipe.encode_prompt("").detach().float()
        pipe.text_encoder.model.cpu()
        torch.cuda.empty_cache()
        pipe.model.to(pipe.device)

        first = train_ds[0]
        sigmas = build_sigma_schedule(pipe, args.sampling_steps, args.shift)
        seq_len, mask2 = build_latent_geometry(pipe, first["z_video"].shape)
        cfg = ActionConditionedWanConfig(
            mode="side_adapter",
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

        free_before, total = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        ckpt_ctx = (
            model_grad_checkpointing(pipe) if args.gradient_checkpointing
            else contextlib.nullcontext()
        )
        model.train()
        with ckpt_ctx:
            train_noise = make_noise_for_batch(args, batch, pipe.device)
            optim.zero_grad(set_to_none=True)
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
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable, args.grad_clip)
            optim.step()
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        payload = {
            "status": "ok",
            "batch_size": args.batch_size,
            "loss": logs["loss"],
            "grad_norm": float(grad_norm.item()),
            "elapsed_s": round(time.time() - t0, 3),
            "gpu_total_mib": mib(total),
            "gpu_free_before_mib": mib(free_before),
            "gpu_free_after_mib": mib(free_after),
            "max_allocated_mib": mib(torch.cuda.max_memory_allocated()),
            "max_reserved_mib": mib(torch.cuda.max_memory_reserved()),
        }
        write_result(args.output_json, payload)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        payload = {
            "status": "oom",
            "batch_size": args.batch_size,
            "elapsed_s": round(time.time() - t0, 3),
            "error": str(exc).splitlines()[0],
        }
        write_result(args.output_json, payload)
        raise SystemExit(42)


if __name__ == "__main__":
    main()
