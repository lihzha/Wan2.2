#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-adaptor
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Sample videos from a trained adaptor and compute SSIM vs ground truth.
# Wrapper around scripts/eval_adaptor.py.
#
# Environment overrides:
#   CKPT_PATH=runs/adaptor_train_v0/ckpt_latest.pt   (or 'none' for M0 init mode)
#   OUTPUT_DIR=runs/eval_v0
#   EVAL_NAMES="45 46 47 48 49"
#   GUIDE_SCALE=5.0

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

CKPT_PATH=${CKPT_PATH:-none}
OUTPUT_DIR=${OUTPUT_DIR:-runs/eval_$(date +%Y%m%d_%H%M%S)}
EVAL_NAMES=${EVAL_NAMES:-"45 46 47 48 49"}
GUIDE_SCALE=${GUIDE_SCALE:-5.0}

echo "[$(date)] eval -> ${OUTPUT_DIR}"
echo "          ckpt=${CKPT_PATH}"
echo "          eval_names=${EVAL_NAMES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/eval_adaptor.py \
    --triplets_root data/triplets \
    --adaptor_init runs/batch_inv_positive/factorization/adaptor_init.pt \
    --ckpt_path ${CKPT_PATH} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --eval_names ${EVAL_NAMES} \
    --output_dir ${OUTPUT_DIR} \
    --guide_scale ${GUIDE_SCALE} \
    --include_oracle \
    --include_null

echo "[$(date)] done"
