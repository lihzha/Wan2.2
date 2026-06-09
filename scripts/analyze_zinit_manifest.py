"""Analyze a curated z_init manifest.

This intentionally samples pairs instead of writing all pairwise distances so
it remains cheap when the dataset grows to thousands of endpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--only_keep", action="store_true")
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument("--pair_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def load_rows(path: str, only_keep: bool):
    rows = list(csv.DictReader(open(path)))
    if only_keep:
        rows = [r for r in rows if int(r.get("keep", "0")) == 1]
    return rows


def load_z(path: str) -> torch.Tensor:
    d = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(d):
        z = d
    elif isinstance(d, dict) and torch.is_tensor(d.get("z_init")):
        z = d["z_init"]
    else:
        raise KeyError(f"{path} has no z_init")
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


def aggregate(rows: list[dict], key: str, metrics: list[str]):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r[key]].append(r)
    out = []
    for name, rs in sorted(buckets.items()):
        item = {key: name, "count": len(rs)}
        for m in metrics:
            vals = torch.tensor([float(r[m]) for r in rs])
            item[f"{m}_mean"] = float(vals.mean())
            item[f"{m}_std"] = float(vals.std(unbiased=True)) if len(vals) > 1 else 0.0
            item[f"{m}_min"] = float(vals.min())
            item[f"{m}_max"] = float(vals.max())
        out.append(item)
    return out


def pca(records):
    by_shape = defaultdict(list)
    for r in records:
        by_shape[r["shape"]].append(r)
    out = []
    for shape, rs in sorted(by_shape.items()):
        if len(rs) < 3:
            continue
        X = torch.stack([r["flat"] for r in rs])
        Xc = X - X.mean(dim=0, keepdim=True)
        gram = (Xc @ Xc.T) / max(1, Xc.shape[1] - 1)
        evals = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
        total = float(evals.sum())
        if total <= 0:
            continue
        row = {"shape": shape, "count": len(rs)}
        for k in (1, 2, 3, 5, 10, 20):
            kk = min(k, len(evals))
            row[f"pc{k}_cum_explained"] = float(evals[:kk].sum() / total)
        out.append(row)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.manifest, args.only_keep)
    if args.max_records:
        rows = rows[:args.max_records]

    records = []
    stats_rows = []
    for r in rows:
        z = load_z(r["z_init_path"])
        flat = free_flat(z)
        rec = {
            "sample_key": r["sample_key"],
            "eta": r["eta"],
            "seed": r["seed"],
            "shape": "x".join(str(x) for x in z.shape),
            "flat": flat,
        }
        records.append(rec)
        stats_rows.append({
            "sample_key": r["sample_key"],
            "eta": r["eta"],
            "seed": r["seed"],
            "shape": rec["shape"],
            "mean": float(flat.mean()),
            "std": float(flat.std(unbiased=True)),
            "rms": float(torch.sqrt(torch.mean(flat.square()))),
        })

    gen = torch.Generator().manual_seed(args.seed)
    pair_rows = []
    if len(records) >= 2:
        for _ in range(args.pair_samples):
            i = int(torch.randint(len(records), (1,), generator=gen))
            j = int(torch.randint(len(records), (1,), generator=gen))
            if i == j:
                continue
            a, b = records[i], records[j]
            if a["shape"] != b["shape"]:
                continue
            relation = "same_sample" if a["sample_key"] == b["sample_key"] else "different_sample"
            if a["sample_key"] == b["sample_key"] and a["eta"] == b["eta"]:
                relation = "same_sample_same_eta"
            pair_rows.append({
                "sample_a": a["sample_key"],
                "sample_b": b["sample_key"],
                "eta_a": a["eta"],
                "eta_b": b["eta"],
                "relation": relation,
                **pair(a["flat"], b["flat"]),
            })

    summary = {
        "manifest": args.manifest,
        "only_keep": args.only_keep,
        "n_records": len(records),
        "endpoint_stats_by_eta": aggregate(
            stats_rows, "eta", ["mean", "std", "rms"]),
        "pair_stats_by_relation": aggregate(
            pair_rows, "relation", ["rmse", "mse", "cosine", "corr"]),
        "pca": pca(records),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if pair_rows:
        with open(out_dir / "pairs_sampled.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
