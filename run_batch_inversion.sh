#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=batch-inv-pos
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A_%a.out
#SBATCH --array=0-4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Shard 50 triplets across 5 array tasks (10 per shard, ~15 min/shard).
# If the triplet count changes, adjust --array=0-<N-1> and SHARD_SIZE so
# they cover the full range.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

SHARD_SIZE=10
START_AT=$((SLURM_ARRAY_TASK_ID * SHARD_SIZE))

echo "[$(date)] shard ${SLURM_ARRAY_TASK_ID}: start_at=${START_AT} limit=${SHARD_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/batch_inversion.py \
    --ckpt_dir Wan2.2-TI2V-5B \
    --output_root runs/batch_inv_positive \
    --mode positive_inversion \
    --heuristic_source empty \
    --L_pos 1 \
    --sampling_steps 25 \
    --null_inner_iters 10 \
    --guide_scale 5.0 \
    --start_at ${START_AT} \
    --limit ${SHARD_SIZE} \
    --manifest_suffix _shard${SLURM_ARRAY_TASK_ID}

echo "[$(date)] shard ${SLURM_ARRAY_TASK_ID} done"
