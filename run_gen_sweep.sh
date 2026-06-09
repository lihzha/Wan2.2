#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=gen-sweep
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Sweep over scripts/sweep_generation.py: 4 prompts x 3 guide_scales = 12 runs.
# Single Python process, model loaded once, ~30-50 min wall on H100/H200.
#
# Override the start frame / output root for other triplets via env vars:
#   START_FRAME=data/triplets/1/frame_10.png OUT_ROOT=data/triplets/1 sbatch run_gen_sweep.sh

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/gen-sweep

START_FRAME=${START_FRAME:-data/triplets/0/frame_10.png}
OUT_ROOT=${OUT_ROOT:-data/triplets/0}
CKPT_DIR=${CKPT_DIR:-Wan2.2-TI2V-5B}
SIZE=${SIZE:-1280*704}
FRAME_NUM=${FRAME_NUM:-49}
SAMPLE_STEPS=${SAMPLE_STEPS:-40}
SAMPLE_SHIFT=${SAMPLE_SHIFT:-5.0}
BASE_SEED=${BASE_SEED:-0}

echo "[gen-sweep] start_frame=${START_FRAME}"
echo "[gen-sweep] out_root=${OUT_ROOT}"
echo "[gen-sweep] size=${SIZE} frames=${FRAME_NUM} steps=${SAMPLE_STEPS} shift=${SAMPLE_SHIFT} seed=${BASE_SEED}"
nvidia-smi || true

python scripts/sweep_generation.py \
    --start_frame "$START_FRAME" \
    --ckpt_dir    "$CKPT_DIR" \
    --out_root    "$OUT_ROOT" \
    --size        "$SIZE" \
    --frame_num   "$FRAME_NUM" \
    --sample_steps "$SAMPLE_STEPS" \
    --sample_shift "$SAMPLE_SHIFT" \
    --base_seed   "$BASE_SEED" \
    --skip_existing
