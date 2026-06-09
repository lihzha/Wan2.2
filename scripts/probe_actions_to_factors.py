"""Linear-probe sanity check: are the per-video factors (γ_v, b_v) from
cross_video_factorize.py predictable from action sequences alone?

If a ridge probe `action -> γ` lands above R² ~ 0.3 and `action -> b`
gives held-out cosine clearly above the random baseline, the
action-conditioned adaptor is justified. If both flop, fix the action
representation (or add I_0 features) before sinking compute into a full
adaptor training run.

Loads:
  adaptor_init.pt              — per-video γ, b, names from factorization
  data/triplets/<i>/actions.npy — (32, 8) action sequence per triplet

Reports:
  - Action diversity across videos (sanity check that actions vary)
  - Ridge R² for action -> γ at several α (with 5-fold CV)
  - Ridge held-out cosine for action -> b at several α
  - Same comparison after projecting b onto its top-K PCs across videos
  - Predicted-vs-true scatter for γ; histogram of held-out cosines for b

Usage:
  python scripts/probe_actions_to_factors.py \\
      --adaptor_init runs/batch_inv_positive/factorization/adaptor_init.pt
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adaptor_init",
        default="runs/batch_inv_positive/factorization/adaptor_init.pt",
    )
    p.add_argument("--triplets_root", default="data/triplets")
    p.add_argument(
        "--out_dir",
        default="runs/batch_inv_positive/factorization/probe",
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--action_repr",
        choices=["flat", "delta", "velocity", "delta_plus_first", "all"],
        default="all",
        help="How to encode the (32, 8) action sequence: 'flat' = raw "
        "flatten (dominated by ~constant home-config offset). 'delta' = "
        "row - row[0]. 'velocity' = row[i+1] - row[i]. "
        "'delta_plus_first' = concat([first row, deltas]). 'all' = run "
        "every representation and report which works best.",
    )
    p.add_argument(
        "--b_pca_k",
        type=int,
        default=10,
        help="Project b onto top-K PCs across videos before regressing. "
        "Helps with V << D regimes (here V=50, D=4096).",
    )
    return p.parse_args()


def encode_action(a, repr_kind):
    """a: [32, 8] action sequence. Returns 1-D feature."""
    if repr_kind == "flat":
        return a.flatten()                                        # [256]
    if repr_kind == "delta":
        return (a - a[0:1]).flatten()                             # [256]
    if repr_kind == "velocity":
        return (a[1:] - a[:-1]).flatten()                         # [248]
    if repr_kind == "delta_plus_first":
        return np.concatenate([a[0], (a - a[0:1])[1:].flatten()]) # [8 + 31*8 = 256]
    raise ValueError(repr_kind)


def load_factors_and_actions(adaptor_init_pt, triplets_root, repr_kind):
    d = torch.load(adaptor_init_pt, map_location="cpu", weights_only=False)
    names = d["names"]
    gammas = d["per_video_gamma"].float().numpy()             # [V]
    bs = d["per_video_b"].float().numpy()                     # [V, L, D]
    bs = bs.reshape(len(names), -1)                            # [V, L*D]

    rows = []
    for name in names:
        idx = name.replace("triplet_", "")
        p = os.path.join(triplets_root, idx, "actions.npy")
        if not os.path.exists(p):
            print(f"[probe] WARN missing {p}; dropping {name}")
            rows.append(None)
            continue
        a = np.load(p)
        assert a.shape == (32, 8), f"{p}: expected (32, 8), got {a.shape}"
        rows.append(encode_action(a.astype(np.float64), repr_kind))

    keep = [i for i, r in enumerate(rows) if r is not None]
    X = np.stack([rows[i] for i in keep])
    gammas = gammas[keep]
    bs = bs[keep]
    names = [names[i] for i in keep]
    return X, gammas, bs, names


def ridge_cv_scalar(X, y, alpha, folds, seed):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    preds, trues = np.zeros_like(y), np.zeros_like(y)
    fold_r2 = []
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], y[tr])
        p = m.predict(X[te])
        preds[te] = p
        trues[te] = y[te]
        ss_res = ((p - y[te]) ** 2).sum()
        ss_tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
        fold_r2.append(1.0 - ss_res / ss_tot)
    # Held-out R² aggregated across folds:
    ss_res = ((preds - trues) ** 2).sum()
    ss_tot = ((trues - trues.mean()) ** 2).sum() + 1e-12
    pooled_r2 = 1.0 - ss_res / ss_tot
    return pooled_r2, float(np.mean(fold_r2)), float(np.std(fold_r2)), preds


def ridge_cv_vector(X, Y, alpha, folds, seed):
    """Returns per-video held-out cosine."""
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    V, D = Y.shape
    preds = np.zeros_like(Y)
    for tr, te in kf.split(X):
        m = Ridge(alpha=alpha).fit(X[tr], Y[tr])
        preds[te] = m.predict(X[te])
    Y_unit = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    P_unit = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-9)
    cos = (Y_unit * P_unit).sum(axis=1)
    return cos


def run_probe_for_repr(args, repr_kind):
    print(f"\n##### action_repr = {repr_kind!r} #####")
    X, gammas, bs, names = load_factors_and_actions(
        args.adaptor_init, args.triplets_root, repr_kind)
    V, D_x = X.shape
    D_b = bs.shape[1]
    print(f"[probe] V={V} videos, action dim={D_x}, b dim={D_b}")

    # Action diversity
    X_unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    cos_act = X_unit @ X_unit.T
    off = ~np.eye(V, dtype=bool)
    print(
        f"[probe] action cosine across videos (off-diag): "
        f"mean={cos_act[off].mean():.3f}  "
        f"range=[{cos_act[off].min():.3f}, {cos_act[off].max():.3f}]"
    )

    # Per-feature standardization
    X_n = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # ---- γ probe ----
    print("\n=== γ_v probe  (action → γ, 5-fold CV) ===")
    print(f"{'alpha':>8}  {'pooled R²':>10}  {'fold mean ± std':>20}")
    best_gamma = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
        pr2, fm, fs, preds = ridge_cv_scalar(
            X_n, gammas, alpha, args.folds, args.seed)
        print(f"  α={alpha:>6.2f}  {pr2:>10.3f}  {fm:>10.3f} ± {fs:.3f}")
        if best_gamma is None or pr2 > best_gamma[0]:
            best_gamma = (pr2, alpha, preds)

    best_pr2, best_alpha_g, best_preds_g = best_gamma
    print(f"  → best α={best_alpha_g}, pooled R²={best_pr2:.3f} "
          f"(baseline mean-predictor: R²=0)")

    # ---- b probe: full 4096-D ----
    print(f"\n=== b_v probe  (action → b ∈ R^{D_b}, 5-fold CV, full dim) ===")
    print(f"{'alpha':>8}  {'mean cos':>10}  {'min':>7}  {'max':>7}")
    best_b_full = None
    for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        cos = ridge_cv_vector(X_n, bs, alpha, args.folds, args.seed)
        print(f"  α={alpha:>8.1f}  {cos.mean():>10.3f}  "
              f"{cos.min():>7.3f}  {cos.max():>7.3f}")
        if best_b_full is None or cos.mean() > best_b_full[0]:
            best_b_full = (cos.mean(), alpha, cos)

    # Random-direction baseline for cosine
    rng = np.random.default_rng(args.seed)
    rand = rng.normal(size=(V, D_b))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9
    b_unit = bs / (np.linalg.norm(bs, axis=1, keepdims=True) + 1e-9)
    rand_cos = (rand * b_unit).sum(axis=1)
    print(f"  baseline (random unit): mean={rand_cos.mean():.3f}  "
          f"std={rand_cos.std():.3f}")

    # ---- b probe: project to top-K PCs across videos first ----
    K = min(args.b_pca_k, V - 1)
    Y_centered = bs - bs.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Y_centered, full_matrices=False)
    # Y in K-PC coords: Z = U[:, :K] * S[:K]
    Z = U[:, :K] * S[:K]
    # Variance fraction kept:
    var_kept = (S[:K] ** 2).sum() / (S ** 2).sum()
    print(f"\n=== b_v probe  (top-{K} PC subspace, retains {var_kept:.1%} of cross-video variance) ===")
    print(f"{'alpha':>8}  {'mean cos':>10}  {'min':>7}  {'max':>7}")
    best_b_pca = None
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        # Predict Z from X_n; reconstruct b̂ = mean + V_K Z_pred
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        b_pred = np.zeros_like(bs)
        for tr, te in kf.split(X_n):
            m = Ridge(alpha=alpha).fit(X_n[tr], Z[tr])
            Z_te = m.predict(X_n[te])
            b_pred[te] = bs.mean(0, keepdims=True) + Z_te @ Vt[:K]
        b_pred_u = b_pred / (np.linalg.norm(b_pred, axis=1, keepdims=True) + 1e-9)
        cos = (b_pred_u * b_unit).sum(axis=1)
        print(f"  α={alpha:>8.3f}  {cos.mean():>10.3f}  "
              f"{cos.min():>7.3f}  {cos.max():>7.3f}")
        if best_b_pca is None or cos.mean() > best_b_pca[0]:
            best_b_pca = (cos.mean(), alpha, cos)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(gammas, best_preds_g, alpha=0.6)
    lims = [min(gammas.min(), best_preds_g.min()),
            max(gammas.max(), best_preds_g.max())]
    axes[0].plot(lims, lims, "k--", alpha=0.5, label="y=x")
    axes[0].set_xlabel("true γ_v")
    axes[0].set_ylabel("predicted γ_v (held-out)")
    axes[0].set_title(f"γ probe: pooled R²={best_pr2:.3f} at α={best_alpha_g}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(best_b_full[2], bins=15, alpha=0.7,
                 label=f"full-D ridge (α={best_b_full[1]:.0f})\nmean={best_b_full[0]:.3f}")
    axes[1].hist(best_b_pca[2], bins=15, alpha=0.5,
                 label=f"top-{K}-PC ridge (α={best_b_pca[1]:.3g})\nmean={best_b_pca[0]:.3f}")
    axes[1].axvline(rand_cos.mean(), color="gray", linestyle="--",
                    label=f"random baseline ({rand_cos.mean():.3f})")
    axes[1].set_xlabel("held-out cosine(true b, pred b)")
    axes[1].set_title("b probe")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, f"probe_results_{repr_kind}.png")
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)

    # ---- Verdict ----
    print("\n=== VERDICT ===")
    if best_pr2 >= 0.3:
        print(f"✓ γ_v IS predictable from action  (best R²={best_pr2:.3f}).")
    elif best_pr2 >= 0.1:
        print(f"~ γ_v marginally predictable  (best R²={best_pr2:.3f}).")
    else:
        print(f"✗ γ_v NOT predictable from action alone (best R²={best_pr2:.3f}).")

    best_b_cos = max(best_b_full[0], best_b_pca[0])
    rand_thresh = rand_cos.mean() + 2 * rand_cos.std()
    if best_b_cos >= max(0.3, rand_thresh):
        print(f"✓ b_v IS predictable from action  (best mean cos={best_b_cos:.3f} >> random {rand_cos.mean():.3f}).")
    elif best_b_cos >= rand_thresh:
        print(f"~ b_v marginally predictable  (cos={best_b_cos:.3f}, random baseline {rand_cos.mean():.3f}).")
    else:
        print(f"✗ b_v NOT predictable from action  (cos={best_b_cos:.3f} ≈ random {rand_cos.mean():.3f}).")

    print(f"\nIf γ works but b fails → b carries non-linear or non-action signal.")
    print(f"If both work → adaptor is well-justified, proceed to end-to-end training.")
    print(f"If both fail → action representation is the issue (try deltas, end-effector, normalization).")

    # Save report
    report = {
        "V": V,
        "action_dim": D_x,
        "b_dim": D_b,
        "action_cosine_offdiag_mean": float(cos_act[off].mean()),
        "gamma_probe": {
            "best_alpha": float(best_alpha_g),
            "pooled_r2": float(best_pr2),
        },
        "b_probe_full": {
            "best_alpha": float(best_b_full[1]),
            "mean_cosine": float(best_b_full[0]),
        },
        "b_probe_topK_pc": {
            "K": int(K),
            "variance_kept": float(var_kept),
            "best_alpha": float(best_b_pca[1]),
            "mean_cosine": float(best_b_pca[0]),
        },
        "random_baseline_cosine": {
            "mean": float(rand_cos.mean()),
            "std": float(rand_cos.std()),
        },
    }
    report["action_repr"] = repr_kind
    return report


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    reprs = (["flat", "delta", "velocity", "delta_plus_first"]
             if args.action_repr == "all" else [args.action_repr])
    all_reports = []
    for r in reprs:
        all_reports.append(run_probe_for_repr(args, r))

    print("\n\n##### COMPARISON ACROSS REPRESENTATIONS #####")
    print(f"{'repr':>20} {'act-cos':>9} {'γ R²':>9} {'b cos (full)':>14} {'b cos (PCA)':>13}")
    for r in all_reports:
        print(
            f"{r['action_repr']:>20} "
            f"{r['action_cosine_offdiag_mean']:>9.3f} "
            f"{r['gamma_probe']['pooled_r2']:>9.3f} "
            f"{r['b_probe_full']['mean_cosine']:>14.3f} "
            f"{r['b_probe_topK_pc']['mean_cosine']:>13.3f}"
        )

    with open(os.path.join(args.out_dir, "probe_report.json"), "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n[probe] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
