"""Analyze the timestep-wise variability of optimized text embeddings.

Loads a positive_embeddings.pt or null_embeddings.pt saved by
scripts/embedding_search.py and produces a directory of:

  pairwise_cosine.{png,csv}  — N×N cosine similarity between C_i and C_j
  pairwise_l2.{png,csv}      — N×N L2 distance ||C_i − C_j||
  step_velocity.{png,csv}    — ||C_i − C_{i-1}|| and ||C_i − C_0|| vs σ
  pca.{png,csv}              — singular-value spectrum across the N-step
                               trajectory + cumulative variance
  per_token.{png,csv}        — (only if L > 1) ||C_i,l − C_0,l|| heatmap
                               showing which tokens move most across steps
  norms.csv                  — per-step ||C_i||
  summary.txt                — human-readable summary with PC counts and
                               key distance statistics

Why this matters: if you want a single adaptor f(action, I_0) → {C_t} to
subsume per-clip per-step null/positive-text inversion, the trajectory
{C_t} has to be smooth and low-rank enough for one network to fit it.
This script measures both.

Usage:
  python scripts/analyze_embeddings.py \\
      --embeddings_pt runs/pos_capacity_001/positive_embeddings.pt
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--embeddings_pt",
        required=True,
        help="Path to positive_embeddings.pt or null_embeddings.pt.",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Where to write outputs. Default: <embeddings_pt's dir>/analysis.",
    )
    return p.parse_args()


def load_embedding_trajectory(path):
    """Load and normalize either embedding-trajectory format.
    Returns (embeds [N, L, D] float, sigmas [N+1] or None, role str)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "positive_embeddings" in d:
        role, seq = "positive", d["positive_embeddings"]
    elif "null_embeddings" in d:
        role, seq = "null", d["null_embeddings"]
    else:
        raise SystemExit(
            f"{path}: contains neither 'positive_embeddings' nor "
            f"'null_embeddings'."
        )
    shapes = {tuple(t.shape) for t in seq}
    if len(shapes) > 1:
        raise SystemExit(f"Embeddings have varying shapes across steps: {shapes}")
    embeds = torch.stack([t.float() for t in seq])  # [N, L, D]
    sigmas = d.get("sigmas")
    if sigmas is not None:
        sigmas = sigmas.float()
    return embeds, sigmas, role


def save_heatmap(matrix, path, title, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xlabel("step j")
    ax.set_ylabel("step i")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_csv(matrix, path, header=None):
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    np.savetxt(
        path, matrix, delimiter=",",
        header="" if header is None else header, comments="",
    )


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.embeddings_pt)), "analysis")
    os.makedirs(out_dir, exist_ok=True)

    embeds, sigmas, role = load_embedding_trajectory(args.embeddings_pt)
    N, L, D = embeds.shape
    print(f"[analyze] loaded {role} embeddings: N={N} steps, L={L} tokens, D={D} dim")
    if sigmas is not None:
        print(f"[analyze] sigmas: {sigmas[0].item():.4f} → {sigmas[-1].item():.4f}  (length {len(sigmas)})")

    flat = embeds.reshape(N, -1)  # [N, L*D]
    norms = flat.norm(dim=1)      # [N]

    # ---- Pairwise distances ----
    flat_unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos_sim = (flat_unit @ flat_unit.T).numpy()
    l2_dist = torch.cdist(flat[None], flat[None]).squeeze(0).numpy()

    save_heatmap(
        cos_sim, os.path.join(out_dir, "pairwise_cosine.png"),
        f"Cosine similarity between C_i and C_j ({role}, N={N})",
        vmin=float(cos_sim.min()), vmax=1.0, cmap="viridis",
    )
    save_heatmap(
        l2_dist, os.path.join(out_dir, "pairwise_l2.png"),
        f"L2 distance ||C_i − C_j|| ({role}, N={N})",
        cmap="magma",
    )
    save_csv(cos_sim, os.path.join(out_dir, "pairwise_cosine.csv"))
    save_csv(l2_dist, os.path.join(out_dir, "pairwise_l2.csv"))

    # ---- Step velocity & drift from anchor C_0 ----
    step_diff = (flat[1:] - flat[:-1]).norm(dim=1).numpy()  # [N-1]
    from_zero = (flat - flat[0:1]).norm(dim=1).numpy()      # [N]

    fig, ax = plt.subplots(figsize=(8, 5))
    if sigmas is not None and len(sigmas) >= N:
        x_vel = sigmas[1:N].numpy()
        x_zero = sigmas[:N].numpy()
        ax.plot(x_vel, step_diff, "o-", label="step velocity ||C_i − C_{i-1}||")
        ax.plot(x_zero, from_zero, "s-", label="drift from C_0  ||C_i − C_0||")
        ax.set_xlabel("σ at step i  (denoising direction →)")
        ax.invert_xaxis()
    else:
        ax.plot(np.arange(1, N), step_diff, "o-", label="step velocity")
        ax.plot(np.arange(N), from_zero, "s-", label="drift from C_0")
        ax.set_xlabel("step i")
    ax.set_ylabel("L2 distance")
    ax.set_title(f"How fast {role} embeddings move across timesteps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "step_velocity.png"), dpi=120)
    plt.close(fig)

    sv_table = np.column_stack([
        np.arange(N),
        sigmas[:N].numpy() if (sigmas is not None and len(sigmas) >= N) else np.full(N, np.nan),
        norms.numpy(),
        from_zero,
        np.concatenate([[np.nan], step_diff]),
    ])
    save_csv(
        sv_table, os.path.join(out_dir, "step_velocity.csv"),
        header="step_i,sigma_i,norm,dist_from_C0,step_velocity",
    )
    save_csv(
        norms.numpy(), os.path.join(out_dir, "norms.csv"),
        header="norm",
    )

    # ---- PCA across the N-step trajectory ----
    centered = flat - flat.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centered)             # length min(N, L*D) = N
    var = (s ** 2) / max(N - 1, 1)
    var_ratio = (var / var.sum()).numpy()
    cum = np.cumsum(var_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(np.arange(1, len(var_ratio) + 1), var_ratio, "o-")
    axes[0].set_xlabel("PC index")
    axes[0].set_ylabel("variance ratio")
    axes[0].set_title("PCA spectrum across timesteps")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(np.arange(1, len(cum) + 1), cum, "o-")
    axes[1].axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90%")
    axes[1].axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="99%")
    axes[1].set_xlabel("# PCs")
    axes[1].set_ylabel("cumulative variance")
    axes[1].set_title("Cumulative variance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pca.png"), dpi=120)
    plt.close(fig)
    save_csv(
        np.column_stack([np.arange(1, len(var_ratio) + 1), var_ratio, cum]),
        os.path.join(out_dir, "pca.csv"),
        header="pc,variance_ratio,cumulative",
    )
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)

    # ---- Per-token analysis (only if L > 1) ----
    per_token_n90 = None
    if L > 1:
        # Movement of each token across steps, relative to step 0.
        per_token = (embeds - embeds[0:1]).norm(dim=-1).numpy()  # [N, L]
        save_heatmap(
            per_token, os.path.join(out_dir, "per_token.png"),
            f"||C_i,l − C_0,l||  ({role}, N={N}, L={L})",
            cmap="magma",
        )
        save_csv(per_token, os.path.join(out_dir, "per_token.csv"))
        # How many tokens carry 90% of the total movement (averaged over steps)?
        tok_total = per_token.mean(axis=0)              # [L]
        tok_sorted = np.sort(tok_total)[::-1]
        tok_cum = np.cumsum(tok_sorted) / tok_sorted.sum()
        per_token_n90 = int(np.searchsorted(tok_cum, 0.9) + 1)

    # ---- Summary ----
    off_diag = ~np.eye(N, dtype=bool)
    summary = [
        f"# Embedding-trajectory analysis ({role})",
        f"path:  {args.embeddings_pt}",
        f"shape: N={N} steps, L={L} tokens, D={D} dim",
        "",
        "## Pairwise cosine similarity (across timesteps)",
        f"  min(off-diag):   {cos_sim[off_diag].min():.4f}",
        f"  mean(off-diag):  {cos_sim[off_diag].mean():.4f}",
        f"  C_0 vs C_{{N-1}}: {cos_sim[0, -1]:.4f}",
        "",
        "## L2 magnitudes",
        f"  per-step norms: range [{norms.min().item():.2f}, {norms.max().item():.2f}], "
        f"mean {norms.mean().item():.2f}",
        f"  step velocity (consecutive ||C_i − C_{{i-1}}||): "
        f"median {np.median(step_diff):.4f}, max {step_diff.max():.4f}",
        f"  total drift ||C_{{N-1}} − C_0||: {from_zero[-1]:.4f}",
        "",
        "## PCA across the N-step trajectory",
        f"  PCs needed for 90% variance: {n90} / {len(var_ratio)}",
        f"  PCs needed for 99% variance: {n99} / {len(var_ratio)}",
        f"  top-3 variance fractions:    "
        f"{var_ratio[0]:.3f}, {var_ratio[1]:.3f}, {var_ratio[2]:.3f}",
    ]
    if per_token_n90 is not None:
        summary += [
            "",
            "## Per-token movement",
            f"  tokens carrying 90% of mean drift: {per_token_n90} / {L}",
        ]
    summary += [
        "",
        "## Reading the numbers (capacity-of-an-adaptor lens)",
        "- Few PCs for 99% variance => trajectory lives on a low-dim subspace",
        "  inside the [N, L*D] embedding space. An adaptor only needs to",
        "  output that many dimensions (plus a fixed σ-conditioned basis).",
        "- Step velocity should grow toward the low-σ (right) end of the plot.",
        "  Sharp jumps anywhere = optimizer found a qualitatively different",
        "  basin at that step; bad for a smooth adaptor.",
        "- Cosine similarity:",
        "    ~1.0 across all (i, j)  => 'one prompt fits all'; per-step",
        "                                conditioning is unnecessary.",
        "    decreases with |i − j|  => smooth σ-dependent drift; adaptor",
        "                                needs a σ embedding but is otherwise",
        "                                easy.",
        "    looks like noise        => bad — each step is independent.",
        "- High mean(off-diag) cosine + low PC count is the *easy* regime.",
    ]
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary) + "\n")

    print("\n".join(summary))
    print(f"\n[analyze] all outputs written to {out_dir}")


if __name__ == "__main__":
    main()
