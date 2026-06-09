"""Single-process sweep over Wan 2.2 TI2V-5B generation hyperparameters.

Loads the pipeline once and iterates over (prompt × sample_guide_scale)
combinations, saving each output as
  <out_root>/<run_name>/video.mp4
  <out_root>/<run_name>/config.json

Designed to fit a single SLURM job: ~12 short videos at 1280x704 / 49 frames /
40 steps takes ~30-50 min on an H100/H200 with offload disabled.

Usage:
  python scripts/sweep_generation.py \\
      --start_frame data/triplets/0/frame_10.png \\
      --ckpt_dir Wan2.2-TI2V-5B \\
      --out_root data/triplets/0 \\
      --size 1280*704 --frame_num 49 --sample_steps 40 --sample_shift 5 \\
      --base_seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from wan.configs import MAX_AREA_CONFIGS, SUPPORTED_SIZES  # noqa: E402
from wan.configs.wan_ti2v_5B import ti2v_5B  # noqa: E402
from wan.textimage2video import WanTI2V  # noqa: E402
from wan.utils.utils import save_video  # noqa: E402


# Prompts. The model expects natural-language captions; the goal is the same
# semantic action ("robot gripper moves back and down") at four levels of
# specificity. Wan was trained on detailed bilingual captions, so descriptive
# camera + subject phrasing tends to elicit cleaner motion than terse prompts.
PROMPTS = {
    "p1_minimal": "The robot gripper moves back and down.",
    "p2_detailed": (
        "A white Franka Panda robotic arm with a black Robotiq gripper at its "
        "end. The gripper smoothly moves backward and downward."
    ),
    "p3_cinematic": (
        "Static camera, fixed angle. A white Franka Panda robotic arm with a "
        "black Robotiq gripper retracts, lowering the gripper backward and "
        "downward in a smooth motion."
    ),
    "p4_action": (
        "The white robotic arm pulls its black gripper backward, then lowers "
        "it down toward the table."
    ),
}

GUIDE_SCALES = [3.0, 5.0, 7.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start_frame", required=True, help="Path to I_0 PNG/JPG.")
    p.add_argument("--ckpt_dir", required=True, help="Wan2.2-TI2V-5B checkpoint dir.")
    p.add_argument(
        "--out_root",
        required=True,
        help="Per-run subfolders are created here, e.g. data/triplets/0.",
    )
    p.add_argument(
        "--size",
        default="1280*704",
        choices=SUPPORTED_SIZES["ti2v-5B"],
        help="Pixel area for the generated video (TI2V-5B keeps input AR).",
    )
    p.add_argument("--frame_num", type=int, default=49, help="Must be 4n+1.")
    p.add_argument("--sample_steps", type=int, default=40)
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument(
        "--negative_prompt",
        default="",
        help="Override negative prompt; default uses the config's neg prompt.",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a run if its video.mp4 already exists.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    print(f"[sweep] Loading WanTI2V-5B from {args.ckpt_dir}…")
    t0 = time.time()
    # offload disabled: T5 + DiT stay on GPU between runs (~20-30s saved/run).
    pipe = WanTI2V(
        config=ti2v_5B,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
        init_on_cpu=False,
        convert_model_dtype=True,
    )
    print(f"[sweep] Model ready in {time.time() - t0:.1f}s on {pipe.device}.")

    img = Image.open(args.start_frame).convert("RGB")
    max_area = MAX_AREA_CONFIGS[args.size]

    runs = [
        (pid, ptext, gs)
        for pid, ptext in PROMPTS.items()
        for gs in GUIDE_SCALES
    ]
    print(f"[sweep] {len(runs)} runs queued: {len(PROMPTS)} prompts x {len(GUIDE_SCALES)} guide scales")

    sweep_log_path = os.path.join(args.out_root, "_sweep_log.jsonl")
    sweep_t0 = time.time()
    for i, (pid, ptext, gs) in enumerate(runs):
        run_name = f"{pid}_gs{gs:g}"
        run_dir = os.path.join(args.out_root, run_name)
        video_path = os.path.join(run_dir, "video.mp4")
        cfg_path = os.path.join(run_dir, "config.json")

        if args.skip_existing and os.path.exists(video_path):
            print(f"[sweep] [{i+1}/{len(runs)}] SKIP {run_name} (video.mp4 exists)")
            continue

        os.makedirs(run_dir, exist_ok=True)
        print(
            f"[sweep] [{i+1}/{len(runs)}] {run_name}\n"
            f"          prompt: {ptext}\n"
            f"          guide_scale={gs} shift={args.sample_shift} "
            f"steps={args.sample_steps} frames={args.frame_num} seed={args.base_seed}"
        )

        cfg = {
            "prompt_id": pid,
            "prompt": ptext,
            "negative_prompt": args.negative_prompt or ti2v_5B.sample_neg_prompt,
            "sample_guide_scale": gs,
            "sample_shift": args.sample_shift,
            "sample_steps": args.sample_steps,
            "frame_num": args.frame_num,
            "size": args.size,
            "max_area": max_area,
            "base_seed": args.base_seed,
            "start_frame": os.path.abspath(args.start_frame),
            "ckpt_dir": os.path.abspath(args.ckpt_dir),
            "task": "ti2v-5B",
        }
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        run_t0 = time.time()
        try:
            video = pipe.generate(
                input_prompt=ptext,
                img=img,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=gs,
                n_prompt=args.negative_prompt,
                seed=args.base_seed,
                offload_model=False,
            )
        except Exception as e:
            print(f"[sweep] [{i+1}/{len(runs)}] FAILED: {e}")
            traceback.print_exc()
            with open(sweep_log_path, "a") as f:
                f.write(json.dumps({"run": run_name, "error": str(e)}) + "\n")
            torch.cuda.empty_cache()
            continue
        gen_s = time.time() - run_t0

        save_video(
            tensor=video[None],
            save_file=video_path,
            fps=ti2v_5B.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        cumulative_s = time.time() - sweep_t0
        print(f"[sweep] [{i+1}/{len(runs)}] OK  {run_name}  gen={gen_s:.1f}s  cum={cumulative_s:.1f}s")

        with open(sweep_log_path, "a") as f:
            f.write(
                json.dumps(
                    {"run": run_name, "gen_s": gen_s, "cumulative_s": cumulative_s, **cfg}
                )
                + "\n"
            )

        del video
        torch.cuda.empty_cache()

    print(f"[sweep] Done. Total wall time: {time.time() - sweep_t0:.1f}s. Outputs in {args.out_root}/")


if __name__ == "__main__":
    main()
