"""Cross-video factorization for adaptor initialization.

Loads every <runs_root>/*/positive_embeddings.pt (or null_embeddings.pt
with --mode null) and computes the factorization

    C_t^v  =  μ_t  +  β_t · γ_v · b^v

where:
    μ_t   = mean across videos, shape [N, L, D]   (shared σ-trajectory)
    β_t   = shared α-curve from rank-1 SVD of residuals, shape [N]
    γ_v   = per-video scalar (how much residual this video uses)
    b^v   = per-video unit direction, shape [L, D]

Saves:
    adaptor_init.pt            μ, β, per-video (γ, b), sigmas, names
    cross_video_report.json    diagnostic numbers
    cross_video_cosine.png     V×V cosine heatmaps for α-curves and b-directions
    factorization_components.png  μ-norm vs σ, β vs σ, γ histogram
    rank1_distribution.png     histogram of per-video rank-1 fractions

Verdict printed at the end: HOLDS / marginal / WEAK based on the
α-cosine and rank-1 fraction thresholds.

Usage:
  python scripts/cross_video_factorize.py \\
      --runs_root runs/batch_inv_positive \\
      --mode positive \\
      --out_dir runs/batch_inv_positive/factorization
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs_root",
        required=True,
        help="Directory containing per-video subdirs, each with "
        "{positive,null}_embeddings.pt.",
    )
    p.add_argument("--mode", choices=["positive", "null"], required=True)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_all(runs_root, mode):
    fname = f"{mode}_embeddings.pt"
    paths = sorted(glob.glob(os.path.join(runs_root, "*", fname)))
    if not paths:
        raise SystemExit(f"No {fname} under {runs_root}/*/")
    print(f"[cross] {len(paths)} runs found")

    seqs, names, sigmas_list = [], [], []
    for p in paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        key = f"{mode}_embeddings"
        if key not in d:
            print(f"[cross] WARN {p}: no '{key}' key, skipping")
            continue
        E = torch.stack([t.float() for t in d[key]])  # [N, L, D]
        seqs.append(E)
        names.append(os.path.basename(os.path.dirname(p)))
        if "sigmas" in d:
            sigmas_list.append(d["sigmas"].float())

    # Truncate to common (N, L, D) if shapes vary.
    shapes = {tuple(e.shape) for e in seqs}
    if len(shapes) > 1:
        print(f"[cross] WARN heterogeneous shapes {shapes}; truncating.")
        min_N = min(e.shape[0] for e in seqs)
        min_L = min(e.shape[1] for e in seqs)
        seqs = [e[:min_N, :min_L] for e in seqs]

    stacked = torch.stack(seqs)  # [V, N, L, D]
    sigmas = sigmas_list[0] if sigmas_list else None
    return stacked, names, sigmas, paths


def factorize(stacked):
    """stacked: [V, N, L, D]. Returns dict of artifacts."""
    V, N, L, D = stacked.shape
    M = L * D
    flat = stacked.reshape(V, N, M)

    mu = flat.mean(dim=0)                # [N, M]
    resid = flat - mu.unsqueeze(0)       # [V, N, M]

    alphas, bs, sigvs = [], [], []
    for v in range(V):
        U, S, Vh = torch.linalg.svd(resid[v], full_matrices=False)
        alphas.append(U[:, 0] * S[0])    # [N]   absorb scale here
        bs.append(Vh[0])                  # [M]   unit
        sigvs.append(S)
    A = torch.stack(alphas)               # [V, N]
    B = torch.stack(bs)                   # [V, M]
    sigvs = torch.stack(sigvs)            # [V, min(N, M)]

    # Sign-align so each α curve has positive max-abs entry.
    for v in range(V):
        idx = A[v].abs().argmax()
        if A[v, idx] < 0:
            A[v] = -A[v]
            B[v] = -B[v]

    A_unit = A / A.norm(dim=1, keepdim=True).clamp_min(1e-9)
    alpha_cos = (A_unit @ A_unit.T).numpy()
    B_unit = B / B.norm(dim=1, keepdim=True).clamp_min(1e-9)
    b_cos = (B_unit @ B_unit.T).numpy()

    # Shared β-curve: mean across videos of the sign-aligned α.
    beta = A.mean(dim=0)                  # [N]
    # γ_v = projection of A^v onto β / ||β||²
    gammas = (A * beta).sum(dim=1) / (beta @ beta).clamp_min(1e-9)  # [V]

    return dict(
        mu=mu.reshape(N, L, D),
        beta=beta,
        alphas=A,
        gammas=gammas,
        b_directions=B.reshape(V, L, D),
        alpha_cos=alpha_cos,
        b_cos=b_cos,
        sigvs=sigvs,
    )


def report(art, names, sigmas, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    V, N = art["alphas"].shape
    off = ~np.eye(V, dtype=bool)
    alpha_off = art["alpha_cos"][off]
    b_off = art["b_cos"][off]

    sv = art["sigvs"]
    rank1_frac = (sv[:, 0] ** 2 / (sv ** 2).sum(dim=1)).numpy()
    # Cumulative variance to estimate residual rank:
    cum = torch.cumsum(sv ** 2, dim=1) / (sv ** 2).sum(dim=1, keepdim=True)
    rank_for_90 = (cum < 0.9).sum(dim=1).numpy() + 1   # PCs needed for 90%
    rank_for_99 = (cum < 0.99).sum(dim=1).numpy() + 1

    diag = {
        "V": int(V),
        "N": int(N),
        "alpha_cosine_off_diag": {
            "min": float(alpha_off.min()),
            "mean": float(alpha_off.mean()),
            "median": float(np.median(alpha_off)),
            "max": float(alpha_off.max()),
        },
        "b_cosine_off_diag": {
            "min": float(b_off.min()),
            "mean": float(b_off.mean()),
            "median": float(np.median(b_off)),
            "max": float(b_off.max()),
        },
        "per_video_residual_rank1_fraction": {
            "min": float(rank1_frac.min()),
            "mean": float(rank1_frac.mean()),
            "median": float(np.median(rank1_frac)),
            "max": float(rank1_frac.max()),
        },
        "per_video_pcs_for_90": {
            "min": int(rank_for_90.min()),
            "mean": float(rank_for_90.mean()),
            "median": int(np.median(rank_for_90)),
            "max": int(rank_for_90.max()),
        },
        "per_video_pcs_for_99": {
            "min": int(rank_for_99.min()),
            "mean": float(rank_for_99.mean()),
            "median": int(np.median(rank_for_99)),
            "max": int(rank_for_99.max()),
        },
        "gamma": {
            "min": float(art["gammas"].min()),
            "mean": float(art["gammas"].mean()),
            "max": float(art["gammas"].max()),
            "std": float(art["gammas"].std()),
        },
        "mu_norm_per_step": (
            art["mu"].reshape(N, -1).norm(dim=1).numpy().tolist()
        ),
        "beta_per_step": art["beta"].numpy().tolist(),
        "names": names,
    }
    with open(os.path.join(out_dir, "cross_video_report.json"), "w") as f:
        json.dump(diag, f, indent=2)

    # ---- adaptor_init.pt ----
    init = {
        "mu": art["mu"].cpu(),
        "beta": art["beta"].cpu(),
        "per_video_gamma": art["gammas"].cpu(),
        "per_video_b": art["b_directions"].cpu(),
        "names": names,
        "sigmas": sigmas if sigmas is not None else None,
    }
    torch.save(init, os.path.join(out_dir, "adaptor_init.pt"))

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im = axes[0].imshow(art["alpha_cos"], vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_title(
        f"α-curves cross-video cosine\noff-diag mean={alpha_off.mean():.3f}")
    plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(art["b_cos"], vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1].set_title(
        f"b-directions cross-video cosine\noff-diag mean={b_off.mean():.3f}")
    plt.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "cross_video_cosine.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    if sigmas is not None:
        x = sigmas[:N].numpy()
        axes[0].plot(x, art["mu"].reshape(N, -1).norm(dim=1).numpy(), "o-")
        axes[1].plot(x, art["beta"].numpy(), "o-")
        for ax in axes[:2]:
            ax.set_xlabel("σ at step i")
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)
    else:
        axes[0].plot(art["mu"].reshape(N, -1).norm(dim=1).numpy(), "o-")
        axes[1].plot(art["beta"].numpy(), "o-")
    axes[0].set_title("Shared μ_t  (‖·‖ vs σ)")
    axes[1].set_title("Shared β_t curve")
    axes[2].hist(art["gammas"].numpy(), bins=min(30, max(V // 2, 5)))
    axes[2].axvline(art["gammas"].mean().item(), color="r",
                    linestyle="--", label=f"mean={art['gammas'].mean():.2f}")
    axes[2].set_xlabel("γ_v")
    axes[2].set_title(f"γ_v distribution (V={V})")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "factorization_components.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(rank1_frac, bins=min(30, max(V // 2, 5)))
    axes[0].axvline(rank1_frac.mean(), color="r", linestyle="--",
                    label=f"mean={rank1_frac.mean():.3f}")
    axes[0].set_xlabel("rank-1 fraction (top-1 PC / sum)")
    axes[0].set_title("How rank-1 is each video's residual?")
    axes[0].legend()
    axes[1].hist(rank_for_90, bins=range(1, int(rank_for_90.max()) + 2))
    axes[1].set_xlabel("# PCs for 90% of residual variance")
    axes[1].set_title("Residual rank per video")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rank1_distribution.png"), dpi=120)
    plt.close(fig)

    # ---- printed summary + verdict ----
    print("\n=== Cross-video factorization summary ===")
    print(f"V = {V} videos, N = {N} sampling steps, "
          f"L*D = {art['mu'].reshape(N, -1).shape[1]}")
    print(f"α-curves cosine across videos: "
          f"mean={alpha_off.mean():.4f}  "
          f"min={alpha_off.min():.4f}  max={alpha_off.max():.4f}")
    print(f"b-directions cosine across videos: "
          f"mean={b_off.mean():.4f}  "
          f"min={b_off.min():.4f}  max={b_off.max():.4f}")
    print(f"Per-video rank-1 fraction: "
          f"mean={rank1_frac.mean():.4f}  "
          f"min={rank1_frac.min():.4f}  max={rank1_frac.max():.4f}")
    print(f"Per-video PCs for 90% / 99%: "
          f"mean {rank_for_90.mean():.1f} / {rank_for_99.mean():.1f}")
    print(f"γ_v range: [{art['gammas'].min():.3f}, "
          f"{art['gammas'].max():.3f}], "
          f"mean={art['gammas'].mean():.3f}, std={art['gammas'].std():.3f}")

    print("\n=== Verdict ===")
    if alpha_off.mean() > 0.95 and rank1_frac.mean() > 0.65:
        print("✓ The rank-1 + shared-α factorization HOLDS at this V.")
        print("  Adaptor with the proposed architecture is well-justified.")
        print("  Use adaptor_init.pt to initialize μ and β.")
    elif alpha_off.mean() > 0.8 and rank1_frac.mean() > 0.5:
        print("~ Marginal. The shared-α curve is approximate; consider")
        print("  rank-2 residual (add a second (γ_2, b_2) head) or")
        print("  learning β as a per-video output too.")
    else:
        print("✗ Factorization weak. The 4-video result didn't scale.")
        print("  Drop the rank-1 assumption and predict {C_t}^v directly")
        print("  with a σ-conditioned MLP head, not a rank-1 outer product.")

    print(f"\nAll outputs in {out_dir}")


def main():
    args = parse_args()
    stacked, names, sigmas, paths = load_all(args.runs_root, args.mode)
    art = factorize(stacked)
    report(art, names, sigmas, args.out_dir)


if __name__ == "__main__":
    main()
