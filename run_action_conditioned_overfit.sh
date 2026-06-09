#!/usr/bin/env bash
#SBATCH --job-name=act-overfit
#SBATCH --output=slurm_outputs/action-overfit/out_%x_%j.log
#SBATCH --error=slurm_outputs/action-overfit/err_%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=am43
#SBATCH --partition=ailab
#SBATCH --qos=ailab

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/action-overfit
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

MODE=${MODE:-pre_context}
OVERFIT_ONE=${OVERFIT_ONE:-ep0_v0}
SAMPLING_STEPS=${SAMPLING_STEPS:-1}
NOISE_MODE=${NOISE_MODE:-fixed}
TOTAL_STEPS=${TOTAL_STEPS:-300}
SEED=${SEED:-0}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-20}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}
LOG_INTERVAL=${LOG_INTERVAL:-10}
CKPT_INTERVAL=${CKPT_INTERVAL:-100}
TRAIN_GUIDE_SCALE=${TRAIN_GUIDE_SCALE:-5.0}
ACTION_TOKENS=${ACTION_TOKENS:-8}
PRE_CONTEXT_TOKENS=${PRE_CONTEXT_TOKENS:-8}
SIDE_ADAPTER_LAYERS=${SIDE_ADAPTER_LAYERS:-24-29}
SIDE_ADAPTER_HIDDEN=${SIDE_ADAPTER_HIDDEN:-512}
SIDE_ADAPTER_HEADS=${SIDE_ADAPTER_HEADS:-8}
RUN_TAG=${RUN_TAG:-${MODE}_${SAMPLING_STEPS}step_${NOISE_MODE}_seed${SEED}}
OUTPUT_DIR=${OUTPUT_DIR:-runs/action_overfit_${OVERFIT_ONE}_${RUN_TAG}}

if [[ "$MODE" != "pre_context" && "$MODE" != "side_adapter" ]]; then
  echo "MODE must be pre_context or side_adapter, got: $MODE" >&2
  exit 2
fi

if [[ "$NOISE_MODE" != "fixed" && "$NOISE_MODE" != "fresh" ]]; then
  echo "NOISE_MODE must be fixed or fresh, got: $NOISE_MODE" >&2
  exit 2
fi

if [[ ! -d "data/droid_cache/train/${OVERFIT_ONE}" ]]; then
  echo "missing overfit sample: data/droid_cache/train/${OVERFIT_ONE}" >&2
  exit 2
fi

if [[ ! -d "Wan2.2-TI2V-5B" ]]; then
  echo "missing checkpoint dir: Wan2.2-TI2V-5B" >&2
  exit 2
fi

echo "[launch] host=$(hostname) cwd=$(pwd)"
echo "[launch] mode=${MODE} sample=${OVERFIT_ONE} sampling_steps=${SAMPLING_STEPS} noise_mode=${NOISE_MODE}"
echo "[launch] total_steps=${TOTAL_STEPS} seed=${SEED} lr=${LR} lr_min_ratio=${LR_MIN_RATIO}"
echo "[launch] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[launch] output=${OUTPUT_DIR}"

.venv/bin/python scripts/train_action_conditioned_wan.py \
  --triplets_root data/droid_cache/train \
  --overfit_one "${OVERFIT_ONE}" \
  --ckpt_dir Wan2.2-TI2V-5B \
  --output_dir "${OUTPUT_DIR}" \
  --mode "${MODE}" \
  --action_dim 7 \
  --action_len 32 \
  --action_repr delta \
  --action_tokens "${ACTION_TOKENS}" \
  --pre_context_tokens "${PRE_CONTEXT_TOKENS}" \
  --side_adapter_layers "${SIDE_ADAPTER_LAYERS}" \
  --side_adapter_hidden "${SIDE_ADAPTER_HIDDEN}" \
  --side_adapter_heads "${SIDE_ADAPTER_HEADS}" \
  --sampling_steps "${SAMPLING_STEPS}" \
  --noise_mode "${NOISE_MODE}" \
  --shift 5.0 \
  --train_guide_scale "${TRAIN_GUIDE_SCALE}" \
  --seed "${SEED}" \
  --total_steps "${TOTAL_STEPS}" \
  --lr "${LR}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr_min_ratio "${LR_MIN_RATIO}" \
  --log_interval "${LOG_INTERVAL}" \
  --ckpt_interval "${CKPT_INTERVAL}"
