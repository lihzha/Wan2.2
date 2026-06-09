#!/usr/bin/env python3
"""Build motion-focused contact sheets and simple video diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_video_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected label=path, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("video label cannot be empty")
    return label, Path(path)


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = frames.shape[1:3]
    out_w, out_h = size
    if (w, h) == (out_w, out_h):
        return frames
    return np.stack(
        [cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) for frame in frames],
        axis=0,
    )


def detect_motion_roi(frames: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames], axis=0).astype(np.float32)
    motion = np.abs(gray - gray[:1]).max(axis=0)
    cutoff = max(float(np.percentile(motion, 94)), 8.0)
    mask = motion >= cutoff
    ys, xs = np.where(mask)
    h, w = motion.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    return x0, y0, x1 - x0, y1 - y0


def parse_roi(text: str | None, frames: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    if not text:
        return detect_motion_roi(frames, pad)
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def parse_frame_ids(text: str | None, n_frames: int) -> list[int]:
    if text:
        ids = [int(p) for p in text.split(",")]
    else:
        ids = np.linspace(0, n_frames - 1, min(6, n_frames)).round().astype(int).tolist()
    return [max(0, min(i, n_frames - 1)) for i in ids]


def crop(frames: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return frames[:, y : y + h, x : x + w]


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_sheet(
    rows: list[tuple[str, np.ndarray]],
    frame_ids: list[int],
    thumb_width: int,
    path: Path,
) -> None:
    cells = []
    for label, frames in rows:
        row = []
        for idx in frame_ids:
            frame = frames[idx]
            scale = thumb_width / frame.shape[1]
            thumb_h = max(1, int(round(frame.shape[0] * scale)))
            thumb = cv2.resize(frame, (thumb_width, thumb_h), interpolation=cv2.INTER_AREA)
            row.append(put_label(thumb, f"{label} f{idx:02d}"))
        cells.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(cells, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def make_motion_heatmap(frames: np.ndarray, roi: tuple[int, int, int, int], path: Path) -> np.ndarray:
    area = crop(frames, roi)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in area], axis=0).astype(np.float32)
    motion = np.abs(gray - gray[:1]).max(axis=0)
    motion_u8 = np.clip(motion / max(float(motion.max()), 1e-6) * 255, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(motion_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), heat)
    return motion


def masked_errors(gt: np.ndarray, pred: np.ndarray, motion: np.ndarray) -> dict[str, float]:
    n = min(len(gt), len(pred))
    gt = gt[:n].astype(np.float32)
    pred = pred[:n].astype(np.float32)
    motion = motion.astype(np.float32)
    mask = motion >= max(float(np.percentile(motion, 90)), 8.0)
    if not mask.any():
        mask = np.ones_like(motion, dtype=bool)
    err = np.abs(pred - gt)
    diff_gt = np.abs(gt[-1] - gt[0]).mean(axis=2)
    diff_pred = np.abs(pred[-1] - pred[0]).mean(axis=2)
    gt_vec = diff_gt[mask].reshape(-1)
    pred_vec = diff_pred[mask].reshape(-1)
    if gt_vec.std() > 1e-6 and pred_vec.std() > 1e-6:
        motion_corr = float(np.corrcoef(gt_vec, pred_vec)[0, 1])
    else:
        motion_corr = 0.0
    return {
        "mae": float(err.mean()),
        "masked_mae": float(err[:, mask, :].mean()),
        "last_mae": float(np.abs(pred[-1] - gt[-1]).mean()),
        "masked_last_mae": float(np.abs(pred[-1] - gt[-1])[mask].mean()),
        "first_last_l1": float(diff_pred.mean()),
        "masked_first_last_l1": float(diff_pred[mask].mean()),
        "gt_masked_first_last_l1": float(diff_gt[mask].mean()),
        "motion_corr": motion_corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="append", required=True, type=parse_video_spec)
    parser.add_argument("--gt_label", default="ground_truth")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--roi", help="Optional x,y,w,h crop. Defaults to GT motion ROI.")
    parser.add_argument("--frames", help="Comma-separated frame ids for sheets.")
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--thumb_width", type=int, default=180)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos = [(label, read_video(path)) for label, path in args.video]
    gt = dict(videos).get(args.gt_label)
    if gt is None:
        raise ValueError(f"gt label {args.gt_label!r} not in videos")

    base_size = (gt.shape[2], gt.shape[1])
    videos = [(label, resize_frames(frames, base_size)) for label, frames in videos]
    gt = dict(videos)[args.gt_label]
    min_frames = min(len(frames) for _, frames in videos)
    videos = [(label, frames[:min_frames]) for label, frames in videos]
    gt = gt[:min_frames]

    roi = parse_roi(args.roi, gt, args.pad)
    frame_ids = parse_frame_ids(args.frames, min_frames)
    roi_rows = [(label, crop(frames, roi)) for label, frames in videos]
    make_sheet(videos, frame_ids, args.thumb_width, args.out_dir / "full_sheet.jpg")
    make_sheet(roi_rows, frame_ids, args.thumb_width, args.out_dir / "roi_sheet.jpg")
    gt_motion = make_motion_heatmap(gt, roi, args.out_dir / "gt_motion_roi.jpg")

    metrics = {
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "frames": frame_ids,
        "videos": {},
    }
    gt_roi = crop(gt, roi)
    for label, frames in videos:
        label_metrics = {
            "num_frames": int(len(frames)),
            "height": int(frames.shape[1]),
            "width": int(frames.shape[2]),
        }
        label_metrics.update(masked_errors(gt_roi, crop(frames, roi), gt_motion))
        metrics["videos"][label] = label_metrics

    with (args.out_dir / "motion_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
