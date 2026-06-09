"""Build a manifest for curated z_init targets.

The stochastic inversion jobs save one directory per
sample/eta/seed containing:

  stochastic_inversion.pt   # z_init endpoint and metadata
  positive_embeddings.pt    # optional optimized contexts proving usability
  reconstruction.json       # validation metrics for that endpoint/context

This script scans those outputs and writes a filterable manifest for later
z_init/context-adaptor training.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True,
                   help="One or more stochastic_zinit output roots.")
    p.add_argument("--droid_cache_root", default="data/droid_cache/train",
                   help="Root containing ep*_v* feature cache dirs.")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_json", default=None)
    p.add_argument("--max_latent_mse", type=float, default=0.10)
    p.add_argument("--max_last_mse", type=float, default=0.20)
    p.add_argument("--require_reconstruction", action="store_true",
                   help="Only include endpoints with optimized contexts and metrics.")
    p.add_argument("--split", default="train")
    return p.parse_args()


def parse_run_dir(path: Path) -> dict:
    parts = path.parts
    sample_key = ""
    eta = ""
    seed = ""
    for part in parts:
        if re.fullmatch(r"ep\d+_v\d+", part):
            sample_key = part
        elif re.fullmatch(r"eta.*", part):
            eta = part.replace("eta", "").replace("p", ".")
        elif re.fullmatch(r"seed\d+", part):
            seed = part.replace("seed", "")
    return {"sample_key": sample_key, "eta": eta, "seed": seed}


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def maybe_float(d: dict, key: str):
    v = d.get(key, "")
    if v == "":
        return ""
    return float(v)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = []
    for root_s in args.roots:
        root = Path(root_s)
        if not root.exists():
            print(f"[manifest] WARN root not found: {root}")
            continue
        for z_path in sorted(root.rglob("stochastic_inversion.pt")):
            run_dir = z_path.parent
            parsed = parse_run_dir(run_dir)
            if not parsed["sample_key"]:
                print(f"[manifest] skip unrecognized path: {z_path}")
                continue
            stats = load_json(run_dir / "stats.json")
            recon = load_json(run_dir / "reconstruction.json")
            pos_path = run_dir / "positive_embeddings.pt"
            has_recon = bool(recon) and pos_path.exists()
            if args.require_reconstruction and not has_recon:
                continue

            full_mse = maybe_float(recon, "latent_mse_full")
            last_mse = maybe_float(recon, "latent_mse_last")
            keep = True
            if full_mse != "" and full_mse > args.max_latent_mse:
                keep = False
            if last_mse != "" and last_mse > args.max_last_mse:
                keep = False
            if args.require_reconstruction and (full_mse == "" or last_mse == ""):
                keep = False

            cache_dir = Path(args.droid_cache_root) / parsed["sample_key"]
            row = {
                "keep": int(keep),
                "sample_key": parsed["sample_key"],
                "split": args.split,
                "eta": parsed["eta"],
                "seed": parsed["seed"],
                "z_init_path": str(z_path),
                "context_path": str(pos_path) if pos_path.exists() else "",
                "stats_path": str(run_dir / "stats.json"),
                "reconstruction_path": str(run_dir / "reconstruction.json") if recon else "",
                "actions_path": str(cache_dir / "actions.npy"),
                "z_I0_path": str(cache_dir / "z_I0.pt"),
                "z_video_path": str(cache_dir / "z_video.pt"),
                "latent_mse_full": full_mse,
                "latent_mse_last": last_mse,
                "endpoint_mean": stats.get("mean", ""),
                "endpoint_std": stats.get("std", ""),
                "endpoint_rms": stats.get("rms", ""),
                "endpoint_q01": stats.get("q01", ""),
                "endpoint_q99": stats.get("q99", ""),
            }
            rows.append(row)

    rows.sort(key=lambda r: (r["sample_key"], float(r["eta"] or 0), int(r["seed"] or 0)))
    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    out_json = Path(args.out_json) if args.out_json else out_csv.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "n_total": len(rows),
            "n_keep": sum(int(r["keep"]) for r in rows),
            "max_latent_mse": args.max_latent_mse,
            "max_last_mse": args.max_last_mse,
            "require_reconstruction": args.require_reconstruction,
            "rows": rows,
        }, f, indent=2)
    print(f"[manifest] wrote {out_csv} rows={len(rows)} keep={sum(int(r['keep']) for r in rows)}")


if __name__ == "__main__":
    main()
