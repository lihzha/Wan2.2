#!/usr/bin/env bash
#SBATCH --job-name=act-export
#SBATCH --output=slurm_outputs/action-overfit/out_%x_%j.log
#SBATCH --error=slurm_outputs/action-overfit/err_%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=am43
#SBATCH --partition=ailab
#SBATCH --qos=ailab

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/action-overfit

RUN_DIR=${RUN_DIR:?RUN_DIR is required}
CKPT_NAME=${CKPT_NAME:-ckpt_best.pt}
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_DIR}/videos_${CKPT_NAME%.pt}}
TRIPLETS_ROOT=${TRIPLETS_ROOT:-data/droid_cache/train}
INCLUDE_NULL=${INCLUDE_NULL:-1}
EVAL_NOISE_MODE=${EVAL_NOISE_MODE:-fixed}
NUM_EVAL_NOISES=${NUM_EVAL_NOISES:-1}
EVAL_SEED_START=${EVAL_SEED_START:-}
EVAL_SEEDS=${EVAL_SEEDS:-}

EXTRA_ARGS=(
  --eval_noise_mode "${EVAL_NOISE_MODE}"
  --num_eval_noises "${NUM_EVAL_NOISES}"
)

if [[ -n "${EVAL_SEED_START}" ]]; then
  EXTRA_ARGS+=(--eval_seed_start "${EVAL_SEED_START}")
fi

if [[ -n "${EVAL_SEEDS}" ]]; then
  EXTRA_ARGS+=(--eval_seeds "${EVAL_SEEDS}")
fi

if [[ "${INCLUDE_NULL}" = "1" ]]; then
  EXTRA_ARGS+=(--include_null)
fi

echo "[export-launch] host=$(hostname) run=${RUN_DIR} ckpt=${CKPT_NAME}"
echo "[export-launch] output=${OUTPUT_DIR} eval_noise_mode=${EVAL_NOISE_MODE}"

.venv/bin/python scripts/export_action_conditioned_wan_video.py \
  --run_dir "${RUN_DIR}" \
  --ckpt_path "${RUN_DIR}/${CKPT_NAME}" \
  --triplets_root "${TRIPLETS_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
