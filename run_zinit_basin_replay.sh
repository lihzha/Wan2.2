#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=zinit-basin
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

NAMES=${NAMES:-ep0_v0,ep1_v0,ep2_v0,ep3_v0}
FRESH_SEEDS=${FRESH_SEEDS:-0,1}
OUT_DIR=${OUT_DIR:-runs/zinit_basin_eval}

echo "[$(date)] z-init basin replay names=${NAMES} fresh=${FRESH_SEEDS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/evaluate_zinit_basin.py \
    --runs_root runs/droid_inv/train \
    --triplets_root data/droid_cache/train \
    --names "${NAMES}" \
    --fresh_seeds "${FRESH_SEEDS}" \
    --ckpt_dir Wan2.2-TI2V-5B \
    --out_dir "${OUT_DIR}" \
    --cross_all

echo "[$(date)] done -> ${OUT_DIR}/basin_replay.csv"
