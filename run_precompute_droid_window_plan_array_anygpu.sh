#!/usr/bin/env bash
# Partition-free variant for backfill/QOS-routed cache jobs.
# Use sbatch command-line flags such as --qos and --time to choose the route.
#SBATCH --job-name=droid-win-cache
#SBATCH --output=slurm_outputs/droid-window-cache/out_%x_%A_%a.log
#SBATCH --error=slurm_outputs/droid-window-cache/err_%x_%A_%a.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=am43

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/droid-window-cache
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PLAN_JSONL=${PLAN_JSONL:?PLAN_JSONL is required}
OUT_ROOT=${OUT_ROOT:-data/droid_cache_windows_v0}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
SAVE_DTYPE=${SAVE_DTYPE:-float16}
LIMIT=${LIMIT:-0}

if [[ ! -f "${PLAN_JSONL}" ]]; then
  echo "missing plan: ${PLAN_JSONL}" >&2
  exit 2
fi
if [[ ! -d "Wan2.2-TI2V-5B" ]]; then
  echo "missing checkpoint dir: Wan2.2-TI2V-5B" >&2
  exit 2
fi

echo "[cache-launch] host=$(hostname) shard=${SHARD_INDEX}/${NUM_SHARDS}"
echo "[cache-launch] plan=${PLAN_JSONL}"
echo "[cache-launch] out_root=${OUT_ROOT} dtype=${SAVE_DTYPE} limit=${LIMIT}"
echo "[cache-launch] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/precompute_features_droid_plan.py \
  --plan_jsonl "${PLAN_JSONL}" \
  --ckpt_dir Wan2.2-TI2V-5B \
  --out_root "${OUT_ROOT}" \
  --shard_index "${SHARD_INDEX}" \
  --num_shards "${NUM_SHARDS}" \
  --save_dtype "${SAVE_DTYPE}" \
  --limit "${LIMIT}"
