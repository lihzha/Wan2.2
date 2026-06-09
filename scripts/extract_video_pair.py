"""Extract a (start_frame, goal_frame) pair from a video for embedding search.

The current `embedding_search.py` supervises the model's *last generated frame*
against a single target image. This helper picks two frames from a video
(t_start and t_start + segment_len - 1), saves them as PNGs, and also dumps the
GT segment as a reference video for visual comparison.

Pick `segment_len` to be `4n+1` to match the TI2V VAE temporal stride; that
becomes the `--frames` arg for `embedding_search.py`.

Example:
  python scripts/extract_video_pair.py \
      --video data/clips/cup_move.mp4 \
      --t_start 0 --segment_len 17 \
      --output_dir data/triplets/001
  # then:
  python scripts/embedding_search.py \
      --start_frame data/triplets/001/I_0.png \
      --goal_frame  data/triplets/001/I_T.png \
      --frames 17 ...
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Source video file.")
    p.add_argument("--t_start", type=int, default=0,
                   help="Frame index used as I_0.")
    p.add_argument("--segment_len", type=int, default=17,
                   help="Number of frames in the goal segment. Should be 4n+1 "
                        "so the TI2V latent grid lines up.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_segment", action="store_true",
                   help="Also write gt_segment.mp4 (frames t_start .. t_start+segment_len-1).")
    p.add_argument("--fps", type=int, default=16)
    args = p.parse_args()
    if (args.segment_len - 1) % 4 != 0:
        raise SystemExit(
            f"--segment_len must be 4n+1; got {args.segment_len}. "
            f"Try 5, 9, 13, 17, 21, ..."
        )
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    reader = imageio.get_reader(args.video)
    meta = reader.get_meta_data()
    n_frames = reader.count_frames() if hasattr(reader, "count_frames") else None
    t0 = args.t_start
    tT = t0 + args.segment_len - 1
    if n_frames is not None and tT >= n_frames:
        raise SystemExit(
            f"goal frame index {tT} out of range (video has {n_frames} frames)."
        )

    frames = []
    for i, f in enumerate(reader):
        if i < t0:
            continue
        if i > tT:
            break
        frames.append(f)
    reader.close()

    if len(frames) != args.segment_len:
        raise SystemExit(
            f"Read {len(frames)} frames; expected {args.segment_len}. "
            f"Video may be shorter than t_start+segment_len."
        )

    Image.fromarray(frames[0]).save(os.path.join(args.output_dir, "I_0.png"))
    Image.fromarray(frames[-1]).save(os.path.join(args.output_dir, "I_T.png"))
    print(f"[extract] I_0  = frame {t0}  -> {args.output_dir}/I_0.png")
    print(f"[extract] I_T  = frame {tT}  -> {args.output_dir}/I_T.png")

    if args.save_segment:
        out_seg = os.path.join(args.output_dir, "gt_segment.mp4")
        with imageio.get_writer(out_seg, fps=args.fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        print(f"[extract] segment ({args.segment_len} frames) -> {out_seg}")

    # Echo the matching --frames arg to use downstream.
    print(f"[extract] use --frames {args.segment_len} when running embedding_search.py.")


if __name__ == "__main__":
    main()
