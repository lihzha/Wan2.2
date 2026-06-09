"""Analyze a large z_init collection produced by collect_zinit_from_plan.py."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pair_samples", type=int, default=100000)
    p.add_argument("--pca_records", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_z(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    z = d["z_init"] if isinstance(d, dict) else d
    return z.float().contiguous()


def free_flat(z: torch.Tensor) -> torch.Tensor:
    return (z[:, 1:] if z.shape[1] > 1 else z).reshape(-1)


def pair(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = a - b
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    aa = a - a.mean()
    bb = b - b.mean()
    corr = torch.dot(aa, bb) / (aa.norm() * bb.norm() + 1e-12)
    return {
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "mse": float(torch.mean(diff.square())),
        "cosine": float(cos),
        "corr": float(corr),
    }


def summarize(vals: list[float]) -> dict:
    t = torch.tensor(vals, dtype=torch.float32)
    return {
        "count": int(t.numel()),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=True)) if t.numel() > 1 else 0.0,
        "min": float(t.min()),
        "q05": float(torch.quantile(t, 0.05)),
        "q50": float(torch.quantile(t, 0.50)),
        "q95": float(torch.quantile(t, 0.95)),
        "max": float(t.max()),
    }


def read_manifests(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("manifest_shard_*.csv")):
        with open(path) as f:
            rows.extend(csv.DictReader(f))
    rows = [r for r in rows if r.get("status") in ("ok", "skip") and r.get("z_init_path")]
    rows.sort(key=lambda r: int(r["index"]))
    return rows


def pca_summary(records: list[dict]) -> dict:
    X = torch.stack([r["flat"] for r in records])
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    gram = (Xc @ Xc.T) / max(1, Xc.shape[1] - 1)
    evals = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = float(evals.sum())
    out = {
        "count": len(records),
        "dim": int(X.shape[1]),
        "mean_vector_rms": float(torch.sqrt(torch.mean(mean.squeeze(0).square()))),
        "mean_vector_norm_per_sqrt_dim": float(mean.squeeze(0).norm() / math.sqrt(X.shape[1])),
    }
    for k in (1, 2, 3, 5, 10, 20, 50, 100):
        kk = min(k, evals.numel())
        out[f"pc{k}_cum_explained"] = float(evals[:kk].sum() / total) if total > 0 else 0.0
    return out


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = read_manifests(root)
    if not rows:
        raise SystemExit(f"No manifest_shard_*.csv rows found under {root}")
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats_keys = ["mean", "std", "rms", "q001", "q01", "q05", "q50", "q95", "q99", "q999", "min", "max"]
    endpoint_stats = {k: summarize([float(r[k]) for r in rows if r.get(k, "") != ""])
                      for k in stats_keys}

    subset_n = min(args.pca_records, len(rows))
    subset_rows = rng.sample(rows, subset_n)
    subset = []
    for r in subset_rows:
        flat = free_flat(load_z(r["z_init_path"]))
        subset.append({"row": r, "flat": flat})

    pair_rows = []
    for _ in range(args.pair_samples):
        a, b = rng.sample(subset, 2)
        m = pair(a["flat"], b["flat"])
        pair_rows.append({
            "index_a": a["row"]["index"],
            "index_b": b["row"]["index"],
            "episode_a": a["row"]["episode"],
            "episode_b": b["row"]["episode"],
            **m,
        })
    with open(out_dir / "pair_samples.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    dim = subset[0]["flat"].numel()
    gen = torch.Generator().manual_seed(args.seed)
    random_pair_rows = []
    for _ in range(min(args.pair_samples, 10000)):
        a = torch.randn(dim, generator=gen)
        b = torch.randn(dim, generator=gen)
        random_pair_rows.append(pair(a, b))

    summary = {
        "root": str(root),
        "n_records": len(rows),
        "subset_records": subset_n,
        "endpoint_stats": endpoint_stats,
        "pair_stats": {k: summarize([float(r[k]) for r in pair_rows])
                       for k in ("rmse", "mse", "cosine", "corr")},
        "random_iid_pair_stats": {k: summarize([float(r[k]) for r in random_pair_rows])
                                  for k in ("rmse", "mse", "cosine", "corr")},
        "pca": pca_summary(subset),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
