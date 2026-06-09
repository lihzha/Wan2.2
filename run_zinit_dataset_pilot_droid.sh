#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --job-name=zinit-data
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

DROID_ROOT=${DROID_ROOT:-/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world}
SPLIT=${SPLIT:-train}
VIEW=${VIEW:-0}
EPISODES=${EPISODES:-0,1,2,3}
SEEDS=${SEEDS:-0,1,2,3}
ETAS=${ETAS:-0.3}
OPTIMIZE_ETAS=${OPTIMIZE_ETAS:-0.3}
OUT_ROOT=${OUT_ROOT:-runs/zinit_dataset_pilot_droid}

echo "[$(date)] z-init dataset pilot episodes=${EPISODES} seeds=${SEEDS} etas=${ETAS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/stochastic_inversion_droid.py \
    --droid_root "${DROID_ROOT}" \
    --split "${SPLIT}" \
    --view "${VIEW}" \
    --episodes "${EPISODES}" \
    --seeds "${SEEDS}" \
    --etas "${ETAS}" \
    --optimize_etas "${OPTIMIZE_ETAS}" \
    --ckpt_dir Wan2.2-TI2V-5B \
    --output_root "${OUT_ROOT}" \
    --max_area 61440 \
    --L_pos 1 \
    --sampling_steps 25 \
    --guide_scale 5.0 \
    --null_inner_iters 10 \
    --overwrite

.venv/bin/python scripts/build_zinit_dataset_manifest.py \
    --roots "${OUT_ROOT}" \
    --droid_cache_root "data/droid_cache/${SPLIT}" \
    --out_csv "${OUT_ROOT}/manifest.csv" \
    --out_json "${OUT_ROOT}/manifest.json" \
    --max_latent_mse 0.10 \
    --max_last_mse 0.20 \
    --require_reconstruction \
    --split "${SPLIT}"

.venv/bin/python scripts/analyze_zinit_manifest.py \
    --manifest "${OUT_ROOT}/manifest.csv" \
    --out_dir "${OUT_ROOT}/analysis" \
    --only_keep \
    --pair_samples 2000

echo "[$(date)] done -> ${OUT_ROOT}/manifest.csv"
