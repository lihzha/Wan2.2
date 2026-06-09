"""Analyze structure of Wan inversion endpoints z_init.

The free-noise region excludes latent frame 0 because i2v pins it to the
encoded initial image at every denoising step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch


ARTIFACT_NAMES = ("z_init_probe.pt", "positive_embeddings.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--random_pairs", type=int, default=256)
    p.add_argument("--max_files", type=int, default=0)
    p.add_argument("--include_inversion_trajectory", action="store_true",
                   help="Fallback to inversion_trajectory.pt when no z_init artifact exists.")
    return p.parse_args()


def sample_key_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        m = re.fullmatch(r"(ep\d+_v\d+)(?:_seed.*)?", part)
        if m:
            return m.group(1)
        m = re.fullmatch(r"(triplet_\d+)(?:_seed.*)?", part)
        if m:
            return m.group(1)
    parent = path.parent.name
    return re.sub(r"_seed.*$", "", parent)


def run_key_from_path(path: Path) -> str:
    return str(path.parent)


def discover_artifacts(roots: list[str], include_traj: bool) -> list[Path]:
    found: list[Path] = []
    for root_s in roots:
        root = Path(root_s)
        if not root.exists():
            continue
        preferred_dirs = set()
        for name in ARTIFACT_NAMES:
            for path in root.rglob(name):
                found.append(path)
                preferred_dirs.add(path.parent)
        if include_traj:
            for path in root.rglob("inversion_trajectory.pt"):
                if path.parent not in preferred_dirs:
                    found.append(path)
    found = sorted(set(found))
    return found


def load_z_init(path: Path) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(blob):
        z = blob
    elif isinstance(blob, dict) and torch.is_tensor(blob.get("z_init")):
        z = blob["z_init"]
    elif isinstance(blob, dict) and "trajectory" in blob:
        z = blob["trajectory"][0]
    else:
        raise KeyError(f"{path} has no z_init or trajectory[0]")
    if z.dim() != 4:
        raise ValueError(f"{path}: expected [C,F,H,W], got {tuple(z.shape)}")
    return z.float().contiguous()


def free_noise_flat(z: torch.Tensor) -> torch.Tensor:
    # Frame 0 is pinned to encoded I0; analyze frames 1..F-1.
    zz = z[:, 1:] if z.shape[1] > 1 else z
    return zz.reshape(-1).float()


def tensor_stats(flat: torch.Tensor) -> dict:
    q = torch.quantile(flat, torch.tensor([0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]))
    centered = flat - flat.mean()
    std = flat.std(unbiased=True)
    skew = torch.mean((centered / (std + 1e-12)) ** 3)
    kurt = torch.mean((centered / (std + 1e-12)) ** 4) - 3.0
    return {
        "n": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(std),
        "rms": float(torch.sqrt(torch.mean(flat.square()))),
        "min": float(flat.min()),
        "q001": float(q[0]),
        "q01": float(q[1]),
        "q05": float(q[2]),
        "q50": float(q[3]),
        "q95": float(q[4]),
        "q99": float(q[5]),
        "q999": float(q[6]),
        "max": float(flat.max()),
        "skew": float(skew),
        "excess_kurtosis": float(kurt),
        "norm": float(flat.norm()),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a - a.mean()
    bb = b - b.mean()
    return cosine(aa, bb)


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = a - b
    return {
        "cosine": cosine(a, b),
        "corr": corrcoef(a, b),
        "mse": float(torch.mean(diff.square())),
        "rmse": float(torch.sqrt(torch.mean(diff.square()))),
        "l2": float(diff.norm()),
        "l2_per_sqrt_dim": float(diff.norm() / math.sqrt(diff.numel())),
    }


def maybe_load_config(path: Path) -> dict:
    cfg = path.parent / "config.json"
    if not cfg.exists() and path.parent.parent.name.startswith("ep"):
        cfg = path.parent.parent / "config.json"
    if not cfg.exists():
        return {}
    try:
        with open(cfg) as f:
            return json.load(f)
    except Exception:
        return {}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], key: str, metrics: list[str]) -> list[dict]:
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


def pca_by_shape(records: list[dict]) -> list[dict]:
    out = []
    by_shape = defaultdict(list)
    for r in records:
        by_shape[r["shape"]].append(r)
    for shape, rs in sorted(by_shape.items()):
        if len(rs) < 3:
            continue
        X = torch.stack([r["flat"] for r in rs], dim=0)
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean
        gram = (Xc @ Xc.T) / max(1, Xc.shape[1] - 1)
        evals = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
        total = float(evals.sum())
        if total <= 0:
            continue
        row = {
            "shape": shape,
            "count": len(rs),
            "mean_vector_rms": float(torch.sqrt(torch.mean(mean.squeeze(0).square()))),
            "mean_vector_norm_per_sqrt_dim": float(mean.squeeze(0).norm() / math.sqrt(X.shape[1])),
        }
        for k in (1, 2, 3, 5, 10):
            kk = min(k, len(evals))
            row[f"pc{k}_cum_explained"] = float(evals[:kk].sum() / total)
        out.append(row)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = discover_artifacts(args.roots, args.include_inversion_trajectory)
    if args.max_files:
        artifacts = artifacts[:args.max_files]
    print(f"[zinit-analysis] loading {len(artifacts)} artifacts")

    records = []
    artifact_rows = []
    for path in artifacts:
        try:
            z = load_z_init(path)
        except Exception as e:
            print(f"[zinit-analysis] skip {path}: {e}")
            continue
        flat = free_noise_flat(z)
        cfg = maybe_load_config(path)
        shape = "x".join(str(x) for x in z.shape)
        rec = {
            "path": str(path),
            "run_key": run_key_from_path(path),
            "sample_key": sample_key_from_path(path),
            "shape": shape,
            "flat": flat,
        }
        records.append(rec)
        row = {
            "path": str(path),
            "sample_key": rec["sample_key"],
            "run_key": rec["run_key"],
            "shape": shape,
            "seed": cfg.get("seed", ""),
            "sampling_steps": cfg.get("sampling_steps", ""),
            "max_area": cfg.get("max_area", ""),
            "heuristic": repr(cfg.get("heuristic", "")),
            **tensor_stats(flat),
        }
        artifact_rows.append(row)

    write_csv(out_dir / "artifacts.csv", artifact_rows)

    pair_rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if a["shape"] != b["shape"]:
                continue
            relation = "same_sample" if a["sample_key"] == b["sample_key"] else "different_sample"
            pair_rows.append({
                "path_a": a["path"],
                "path_b": b["path"],
                "sample_a": a["sample_key"],
                "sample_b": b["sample_key"],
                "shape": a["shape"],
                "relation": relation,
                **pair_metrics(a["flat"], b["flat"]),
            })
    write_csv(out_dir / "pairs.csv", pair_rows)

    random_rows = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    shapes = defaultdict(list)
    for r in records:
        shapes[r["shape"]].append(r)
    for shape, rs in sorted(shapes.items()):
        n = rs[0]["flat"].numel()
        for idx in range(args.random_pairs):
            a = torch.randn(n, generator=gen)
            b = torch.randn(n, generator=gen)
            random_rows.append({
                "shape": shape,
                "pair_index": idx,
                **pair_metrics(a, b),
            })
    write_csv(out_dir / "random_pairs.csv", random_rows)

    pca_rows = pca_by_shape(records)
    write_csv(out_dir / "pca_by_shape.csv", pca_rows)

    summary = {
        "n_artifacts": len(records),
        "roots": args.roots,
        "artifact_stats_by_shape": aggregate(
            artifact_rows, "shape", ["mean", "std", "rms", "skew", "excess_kurtosis"]),
        "pair_stats_by_relation": aggregate(
            pair_rows, "relation", ["cosine", "corr", "rmse", "mse"]),
        "random_pair_stats_by_shape": aggregate(
            random_rows, "shape", ["cosine", "corr", "rmse", "mse"]),
        "pca_by_shape": pca_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
