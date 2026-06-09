#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --job-name=zinit10k
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

OUT_ROOT=${OUT_ROOT:-runs/zinit_10k_ddim_droid}
PLAN=${PLAN:-${OUT_ROOT}/plan.jsonl}
SHARD_SIZE=${SHARD_SIZE:-500}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_AT=$((TASK_ID * SHARD_SIZE))

echo "[$(date)] zinit10k shard=${TASK_ID} start=${START_AT} size=${SHARD_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/collect_zinit_from_plan.py \
    --plan_jsonl "${PLAN}" \
    --ckpt_dir Wan2.2-TI2V-5B \
    --output_root "${OUT_ROOT}" \
    --start_at "${START_AT}" \
    --limit "${SHARD_SIZE}" \
    --max_area 61440 \
    --sampling_steps 25 \
    --guide_scale 5.0 \
    --L_pos 1 \
    --dtype float16

echo "[$(date)] shard ${TASK_ID} done"
