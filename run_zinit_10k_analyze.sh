#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=zinit10k-an
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

OUT_ROOT=${OUT_ROOT:-runs/zinit_10k_ddim_droid}

echo "[$(date)] analyzing ${OUT_ROOT}"
.venv/bin/python scripts/analyze_zinit_large.py \
    --root "${OUT_ROOT}" \
    --out_dir "${OUT_ROOT}/analysis" \
    --pair_samples 100000 \
    --pca_records 2000

echo "[$(date)] done -> ${OUT_ROOT}/analysis/summary.json"
