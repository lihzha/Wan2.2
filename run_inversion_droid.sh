#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --job-name=droid-inv
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Re-derive the μ/β warm-start at 320x192 from DROID itself.
# 1) positive_inversion over N episodes (reuses one loaded pipeline)
# 2) cross_video_factorize -> a new, in-domain adaptor_init.pt
#
# Env overrides: LIMIT=32  VIEW=0  SPLIT=train

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

DROID_ROOT=/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world
LIMIT=${LIMIT:-32}
VIEW=${VIEW:-0}
SPLIT=${SPLIT:-train}
OUT_SPLIT=runs/droid_inv/${SPLIT}

echo "[$(date)] inverting ${LIMIT} DROID ${SPLIT} episodes (view ${VIEW}) at 320x192"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/batch_inversion_droid.py \
    --droid_root ${DROID_ROOT} \
    --split ${SPLIT} --view ${VIEW} --limit ${LIMIT} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --output_root runs/droid_inv \
    --max_area 61440 --L_pos 1 --sampling_steps 25 --guide_scale 5.0

echo "[$(date)] factorizing -> ${OUT_SPLIT}/factorization"
.venv/bin/python scripts/cross_video_factorize.py \
    --runs_root ${OUT_SPLIT} \
    --mode positive \
    --out_dir ${OUT_SPLIT}/factorization

echo "[$(date)] done. new warm-start: ${OUT_SPLIT}/factorization/adaptor_init.pt"
echo "[$(date)] check ${OUT_SPLIT}/factorization/cross_video_report.json for"
echo "          alpha_cosine (want ~0.99) and rank-1 fraction (want ~0.7)."
