"""Export optimized per-step contexts as a compatible adaptor checkpoint.

For one-clip overfit diagnostics, this creates an adaptor whose output is the
given context trajectory regardless of actions/I0:

    C_t(actions, I0) = mu_t = optimized_context_t

The residual head is zeroed, so `scripts/eval_adaptor.py` can load the result
as a normal checkpoint and replay it from seed-0.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch

from models.trajectory_adaptor import TrajectoryAdaptor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--context_path", required=True,
                   help=".pt with positive_embeddings list or tensor contexts")
    p.add_argument("--adaptor_init", required=True)
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_len", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--arch_fusion", choices=["concat", "cross_attn"], default="cross_attn")
    p.add_argument("--arch_head", choices=["rank1", "rankk", "perstep"], default="perstep")
    p.add_argument("--rank_k", type=int, default=4)
    p.add_argument("--n_xattn_layers", type=int, default=2)
    p.add_argument("--step_emb_dim", type=int, default=128)
    p.add_argument("--train_guide_scale", type=float, default=5.0)
    return p.parse_args()


def load_contexts(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(d, torch.Tensor):
        ctx = d.float()
    elif "positive_embeddings" in d:
        pe = d["positive_embeddings"]
        ctx = torch.stack([e.float() for e in pe], dim=0) if isinstance(pe, list) else pe.float()
    else:
        raise KeyError(f"{path} has neither tensor content nor positive_embeddings")
    if ctx.dim() != 3:
        raise ValueError(f"expected contexts [N,L,D], got {tuple(ctx.shape)}")
    return ctx


def zero_residual(adaptor: TrajectoryAdaptor):
    if adaptor.head == "perstep":
        last = adaptor.residual_mlp[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)
        return
    if hasattr(adaptor, "gamma_head"):
        torch.nn.init.zeros_(adaptor.gamma_head.weight)
        torch.nn.init.zeros_(adaptor.gamma_head.bias)
    if hasattr(adaptor, "b_head"):
        torch.nn.init.zeros_(adaptor.b_head.weight)
        torch.nn.init.zeros_(adaptor.b_head.bias)


def main():
    args = parse_args()
    contexts = load_contexts(args.context_path)
    adaptor = TrajectoryAdaptor(
        adaptor_init_path=args.adaptor_init,
        hidden=args.hidden,
        action_dim=args.action_dim,
        action_len=args.action_len,
        fusion=args.arch_fusion,
        head=args.arch_head,
        rank_k=args.rank_k,
        n_xattn_layers=args.n_xattn_layers,
        step_emb_dim=args.step_emb_dim,
    )
    if tuple(adaptor.mu.shape) != tuple(contexts.shape):
        raise ValueError(
            f"context shape {tuple(contexts.shape)} does not match adaptor mu "
            f"{tuple(adaptor.mu.shape)}")
    with torch.no_grad():
        adaptor.mu.copy_(contexts)
        zero_residual(adaptor)

    ckpt_args = {
        "hidden": args.hidden,
        "action_dim": args.action_dim,
        "action_len": args.action_len,
        "action_repr": "delta",
        "b_head_init_scale": 1e-4,
        "arch_fusion": args.arch_fusion,
        "arch_head": args.arch_head,
        "rank_k": args.rank_k,
        "n_xattn_layers": args.n_xattn_layers,
        "step_emb_dim": args.step_emb_dim,
        "train_guide_scale": args.train_guide_scale,
        "sampling_steps": contexts.shape[0],
    }
    os.makedirs(os.path.dirname(args.output_ckpt), exist_ok=True)
    torch.save(
        {
            "step": -1,
            "adaptor": adaptor.state_dict(),
            "optimizer": None,
            "args": ckpt_args,
            "source_context_path": args.context_path,
        },
        args.output_ckpt,
    )
    print(f"[export] wrote {args.output_ckpt} from {args.context_path}")


if __name__ == "__main__":
    main()
