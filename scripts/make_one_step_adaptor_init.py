"""Create a reduced-step adaptor_init.pt from a multi-step adaptor init.

This is a diagnostic helper for short-step denoise/generation overfit tests.
The normal adaptor init stores a sigma-indexed context table `mu` with shape
[N, L, D] and `beta` with shape [N]. For `--sampling_steps K`, the adaptor
must also have N=K or train/eval timestep indexing no longer matches.
"""
from __future__ import annotations

import argparse
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Input multi-step adaptor_init.pt")
    p.add_argument("--dst", required=True, help="Output reduced-step adaptor_init.pt")
    p.add_argument(
        "--num_steps",
        type=int,
        default=1,
        help="Number of output sampling steps.",
    )
    p.add_argument(
        "--source_step",
        type=int,
        default=0,
        help="Source row for --num_steps 1 when --source_steps is omitted.",
    )
    p.add_argument(
        "--source_steps",
        default=None,
        help="Comma-separated source rows to copy, length must equal --num_steps.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    mu = ckpt["mu"].float()
    beta = ckpt["beta"].float()
    if mu.dim() != 3:
        raise ValueError(f"expected mu [N,L,D], got {tuple(mu.shape)}")
    if beta.dim() != 1 or beta.shape[0] != mu.shape[0]:
        raise ValueError(
            f"expected beta [N] matching mu, got beta={tuple(beta.shape)} "
            f"mu={tuple(mu.shape)}"
        )
    if args.num_steps < 1:
        raise ValueError(f"--num_steps must be positive, got {args.num_steps}")
    if args.source_steps is None:
        if args.num_steps != 1:
            raise ValueError("--source_steps is required when --num_steps != 1")
        source_steps = [args.source_step]
    else:
        source_steps = [int(x) for x in args.source_steps.split(",") if x != ""]
    if len(source_steps) != args.num_steps:
        raise ValueError(
            f"--source_steps length {len(source_steps)} must equal "
            f"--num_steps {args.num_steps}"
        )
    bad = [i for i in source_steps if not (0 <= i < mu.shape[0])]
    if bad:
        raise ValueError(
            f"source rows must be in [0, {mu.shape[0] - 1}], got {bad}"
        )

    out = dict(ckpt)
    idx = torch.tensor(source_steps, dtype=torch.long)
    out["mu"] = mu.index_select(0, idx).contiguous()
    out["beta"] = beta.index_select(0, idx).contiguous()

    # Keep metadata useful when present. The actual train/eval schedule is
    # rebuilt from --sampling_steps/--shift, but storing two endpoints avoids
    # downstream diagnostics assuming len(sigmas) == N + 1 and failing.
    sigmas = ckpt.get("sigmas", None)
    if torch.is_tensor(sigmas) and sigmas.numel() >= 2:
        start_sigmas = sigmas.index_select(0, idx).float()
        out["sigmas"] = torch.cat([start_sigmas, sigmas[-1:].float()])

    out["reduced_step_source"] = {
        "src": args.src,
        "source_steps": [int(i) for i in source_steps],
        "source_num_steps": int(mu.shape[0]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(out, args.dst)
    print(
        f"wrote {args.dst}: mu={tuple(out['mu'].shape)} "
        f"beta={tuple(out['beta'].shape)}"
    )


if __name__ == "__main__":
    main()
