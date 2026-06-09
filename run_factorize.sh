#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=factorize
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Cross-video factorization. CPU-light — could drop the GPU request and
# request only --cpus-per-task=4 to queue faster, but kept GPU here to
# stay on the same partition the array job used. Runs in ~1 minute on
# this data scale.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

echo "[$(date)] factorizing across $(ls runs/batch_inv_positive/triplet_*/positive_embeddings.pt 2>/dev/null | wc -l) videos"

.venv/bin/python scripts/cross_video_factorize.py \
    --runs_root runs/batch_inv_positive \
    --mode positive \
    --out_dir runs/batch_inv_positive/factorization

echo "[$(date)] done"
