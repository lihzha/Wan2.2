"""Dataset training for action-conditioned Wan prototypes on DROID.

This is the multi-sample counterpart to train_action_conditioned_wan.py. It
keeps the Wan backbone frozen, trains only the action-conditioning wrapper, and
uses the same closed-loop rollout loss so results are comparable to the
one-sample overfit diagnostics.

The default setup is intentionally conservative: batch size 1, fresh Gaussian
z_init per optimizer step, 25 diffusion steps, and a small deterministic
validation subset.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from models.action_conditioned_wan import (  # noqa: E402
    ActionConditionedWanConfig,
    ActionConditionedWanModel,
)
from train_adaptor import (  # noqa: E402
    TripletLatentDataset,
    build_latent_geometry,
    build_sigma_schedule,
    build_wan_pipeline,
    collate,
    model_grad_checkpointing,
)
from train_action_conditioned_wan import (  # noqa: E402
    cosine_lr,
    eval_rollout,
    make_fixed_noise,
    parse_layers,
    rollout_loss,
    save_ckpt,
)


def list_triplet_names(root: str) -> list[str]:
    names = [
        p.name for p in Path(root).iterdir()
        if p.is_dir() and (p / "actions.npy").exists()
    ]
    return sorted(names)


def stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "little")
    return (int(base_seed) + offset) % (2**31 - 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplets_root", default="data/droid_cache/train")
    p.add_argument("--val_triplets_root", default="data/droid_cache/val")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--mode", choices=["pre_context", "side_adapter"],
                   default="side_adapter")
    p.add_argument("--train_names", nargs="*", default=None)
    p.add_argument("--val_names", nargs="*", default=None)
    p.add_argument("--train_manifest_jsonl", default=None,
                   help="Optional DROID window-plan JSONL. If set, training "
                   "names are read from the manifest and file existence is "
                   "checked lazily in __getitem__, avoiding millions of stat "
                   "calls for large caches.")
    p.add_argument("--val_manifest_jsonl", default=None)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=8)
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
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--val_interval", type=int, default=1000)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_optimizer", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false",
                   dest="gradient_checkpointing")
    return p.parse_args()


def names_from_manifest(path: str) -> list[str]:
    names = []
    with open(path) as f:
        for line in f:
            if line.strip():
                names.append(json.loads(line)["name"])
    return names


class LazyTripletLatentDataset(Dataset):
    """TripletLatentDataset-compatible loader without eager per-file stat calls."""

    def __init__(self, triplets_root: str, names: list[str]):
        self.triplets_root = Path(triplets_root)
        self.names = list(names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        tdir = self.triplets_root / name
        return {
            "name": name,
            "actions": torch.from_numpy(np.load(tdir / "actions.npy")).float(),
            "z_I0": torch.load(tdir / "z_I0.pt", map_location="cpu",
                               weights_only=False).float(),
            "z_video": torch.load(tdir / "z_video.pt", map_location="cpu",
                                  weights_only=False).float(),
        }


def make_noise_for_batch(args, batch, device):
    shape = batch["z_video"].shape[1:]
    if args.noise_mode == "fresh":
        return torch.randn(*shape, device=device, dtype=torch.float32)
    name = batch["name"][0]
    return make_fixed_noise(shape, device, stable_seed(args.seed, name))


@torch.no_grad()
def run_validation(
    args,
    pipe,
    model,
    val_loader,
    sigmas,
    null_context,
    seq_len,
    mask2,
):
    if val_loader is None:
        return None
    was_training = model.training
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if args.max_val_samples > 0 and i >= args.max_val_samples:
            break
        name = batch["name"][0]
        eval_noise = make_fixed_noise(
            batch["z_video"].shape[1:], pipe.device,
            stable_seed(args.seed + 10000, name),
        )
        _, logs = eval_rollout(
            pipe, model, batch, sigmas, null_context, seq_len, mask2,
            args.train_guide_scale, eval_noise)
        losses.append(float(logs["loss"]))
    if was_training:
        model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def main():
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("closed-loop rollout training currently requires batch_size=1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.train_manifest_jsonl:
        train_names = names_from_manifest(args.train_manifest_jsonl)
    else:
        train_names = args.train_names or list_triplet_names(args.triplets_root)
    if args.max_train_samples > 0:
        train_names = train_names[:args.max_train_samples]
    if not train_names:
        raise ValueError(f"no training samples found under {args.triplets_root}")

    val_names = None
    if args.val_triplets_root and Path(args.val_triplets_root).is_dir():
        if args.val_manifest_jsonl:
            val_names = names_from_manifest(args.val_manifest_jsonl)
        else:
            val_names = args.val_names or list_triplet_names(args.val_triplets_root)
        if args.max_val_samples > 0:
            val_names = val_names[:args.max_val_samples]

    ds_cls = LazyTripletLatentDataset if args.train_manifest_jsonl else TripletLatentDataset
    train_ds = ds_cls(args.triplets_root, train_names)
    val_ds = (
        (LazyTripletLatentDataset if args.val_manifest_jsonl else TripletLatentDataset)(
            args.val_triplets_root, val_names)
        if val_names else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate,
        )
        if val_ds is not None else None
    )

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

    print(f"[action-droid] train_samples={len(train_ds)} "
          f"val_samples={0 if val_ds is None else len(val_ds)} mode={args.mode}")
    print(f"[action-droid] steps={args.sampling_steps} "
          f"train_steps={args.total_steps} noise_mode={args.noise_mode} "
          f"guide_scale={args.train_guide_scale}")
    print(f"[action-droid] seq_len={seq_len} z_shape={tuple(first['z_video'].shape)} "
          f"trainable_params={sum(p.numel() for p in trainable)/1e6:.2f}M")

    train_log = out_dir / "train_log.csv"
    val_log = out_dir / "val_log.csv"
    train_fields = [
        "step", "sample", "loss", "latent_mse", "z_pred_std",
        "z_target_std", "z_init_std", "lr", "grad_norm", "wall_s",
    ]
    with open(train_log, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=train_fields).writeheader()
    with open(val_log, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["step", "val_loss", "n_val", "wall_s"]).writeheader()

    best_val = float("inf")
    latest_val = None
    t0 = time.time()
    train_iter = iter(train_loader)
    ckpt_ctx = (
        model_grad_checkpointing(pipe) if args.gradient_checkpointing
        else contextlib.nullcontext()
    )

    model.train()
    with ckpt_ctx:
        for step in range(args.total_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            scale = cosine_lr(
                step, args.total_steps, args.warmup_steps, args.lr_min_ratio)
            for group in optim.param_groups:
                group["lr"] = args.lr * scale

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
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()

            logs["lr"] = optim.param_groups[0]["lr"]
            logs["grad_norm"] = float(grad_norm.item())
            logs["wall_s"] = round(time.time() - t0, 3)
            if step == 0 or (step + 1) % args.log_interval == 0:
                print(
                    f"[action-droid] step={step + 1:06d} "
                    f"sample={batch['name'][0]} loss={logs['loss']:.6f} "
                    f"grad={logs['grad_norm']:.3f} lr={logs['lr']:.2e}")
                with open(train_log, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=train_fields)
                    writer.writerow({"step": step + 1, "sample": batch["name"][0], **logs})

            should_val = (
                val_loader is not None and args.val_interval > 0
                and ((step + 1) % args.val_interval == 0 or step == args.total_steps - 1)
            )
            if should_val:
                latest_val = run_validation(
                    args, pipe, model, val_loader, sigmas, null_context, seq_len, mask2)
                wall = round(time.time() - t0, 3)
                print(
                    f"[action-droid] step={step + 1:06d} "
                    f"val_loss={latest_val:.6f} n_val={args.max_val_samples}")
                with open(val_log, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["step", "val_loss", "n_val", "wall_s"])
                    writer.writerow({
                        "step": step + 1,
                        "val_loss": latest_val,
                        "n_val": args.max_val_samples,
                        "wall_s": wall,
                    })
                if latest_val < best_val:
                    best_val = latest_val
                    save_ckpt(
                        out_dir / "ckpt_best_val.pt", model, optim, args,
                        step, {**logs, "val_loss": latest_val})

            if (
                (step + 1) % args.ckpt_interval == 0
                or step == args.total_steps - 1
            ):
                save_ckpt(out_dir / "ckpt_latest.pt", model, optim, args, step, logs)

    summary = {
        "train_samples": len(train_ds),
        "val_samples": 0 if val_ds is None else len(val_ds),
        "latest_train": logs,
        "latest_val_loss": latest_val,
        "best_val_loss": None if best_val == float("inf") else best_val,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[action-droid] done latest={logs['loss']:.6f} best_val={summary['best_val_loss']}")


if __name__ == "__main__":
    main()
