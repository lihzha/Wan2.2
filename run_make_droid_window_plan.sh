#!/usr/bin/env bash
#SBATCH --job-name=droid-plan
#SBATCH --output=slurm_outputs/droid-window-cache/out_%x_%j.log
#SBATCH --error=slurm_outputs/droid-window-cache/err_%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=am43
#SBATCH --partition=ailab
#SBATCH --qos=ailab

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/droid-window-cache

DROID_ROOT=${DROID_ROOT:-/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world}
SPLIT=${SPLIT:-train}
VIEW=${VIEW:-0}
FRAMES=${FRAMES:-33}
CANDIDATE_STRIDE=${CANDIDATE_STRIDE:-4}
MIN_WINDOWS_PER_EPISODE=${MIN_WINDOWS_PER_EPISODE:-50}
MAX_WINDOWS_PER_EPISODE=${MAX_WINDOWS_PER_EPISODE:-50}
CACHE_CAP_GIB=${CACHE_CAP_GIB:-500}
BYTES_PER_SAMPLE=${BYTES_PER_SAMPLE:-240000}
PLAN_DIR=${PLAN_DIR:-runs/droid_window_plans}
PLAN_NAME=${PLAN_NAME:-${SPLIT}_v${VIEW}_${FRAMES}f_stride${CANDIDATE_STRIDE}_cap${CACHE_CAP_GIB}g_fp16_top${MAX_WINDOWS_PER_EPISODE}}

mkdir -p "${PLAN_DIR}"
OUT_JSONL=${OUT_JSONL:-${PLAN_DIR}/${PLAN_NAME}.jsonl}
SUMMARY_JSON=${SUMMARY_JSON:-${PLAN_DIR}/${PLAN_NAME}.summary.json}

echo "[plan-launch] split=${SPLIT} view=${VIEW} frames=${FRAMES} stride=${CANDIDATE_STRIDE}"
echo "[plan-launch] cap_gib=${CACHE_CAP_GIB} bytes_per_sample=${BYTES_PER_SAMPLE}"
echo "[plan-launch] requested min=${MIN_WINDOWS_PER_EPISODE} max=${MAX_WINDOWS_PER_EPISODE}"
echo "[plan-launch] out=${OUT_JSONL}"

.venv/bin/python scripts/make_droid_window_plan.py \
  --droid_root "${DROID_ROOT}" \
  --split "${SPLIT}" \
  --view "${VIEW}" \
  --frames "${FRAMES}" \
  --candidate_stride "${CANDIDATE_STRIDE}" \
  --min_windows_per_episode "${MIN_WINDOWS_PER_EPISODE}" \
  --max_windows_per_episode "${MAX_WINDOWS_PER_EPISODE}" \
  --cache_cap_gib "${CACHE_CAP_GIB}" \
  --bytes_per_sample "${BYTES_PER_SAMPLE}" \
  --out_jsonl "${OUT_JSONL}" \
  --summary_json "${SUMMARY_JSON}"
