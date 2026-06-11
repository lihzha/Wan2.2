#!/usr/bin/env bash
#SBATCH --job-name=act-droid-dist
#SBATCH --output=slurm_outputs/action-droid-dist/out_%x_%j.log
#SBATCH --error=slurm_outputs/action-droid-dist/err_%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=am43
#SBATCH --partition=ailab
#SBATCH --qos=ailab

set -euo pipefail

CODE_DIR=${CODE_DIR:-/scratch/gpfs/AM43/lz3952/Wan2.2}
cd "${CODE_DIR}"
mkdir -p slurm_outputs/action-droid-dist
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MODE=${MODE:-side_adapter}
SAMPLING_STEPS=${SAMPLING_STEPS:-25}
NOISE_MODE=${NOISE_MODE:-fresh}
TOTAL_STEPS=${TOTAL_STEPS:-10000}
SEED=${SEED:-0}
LR=${LR:-5e-5}
WARMUP_STEPS=${WARMUP_STEPS:-500}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.1}
TRAIN_GUIDE_SCALE=${TRAIN_GUIDE_SCALE:-5.0}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-0}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-8}
NUM_WORKERS=${NUM_WORKERS:-2}
VAL_INTERVAL=${VAL_INTERVAL:-1000}
LOG_INTERVAL=${LOG_INTERVAL:-20}
CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
ACTION_TOKENS=${ACTION_TOKENS:-8}
PRE_CONTEXT_TOKENS=${PRE_CONTEXT_TOKENS:-8}
SIDE_ADAPTER_LAYERS=${SIDE_ADAPTER_LAYERS:-0-29}
SIDE_ADAPTER_HIDDEN=${SIDE_ADAPTER_HIDDEN:-512}
SIDE_ADAPTER_HEADS=${SIDE_ADAPTER_HEADS:-8}
RUN_TAG=${RUN_TAG:-${MODE}_${SAMPLING_STEPS}step_${NOISE_MODE}_${TOTAL_STEPS}steps_lr${LR}_seed${SEED}_${GPUS_PER_NODE}gpu}
OUTPUT_DIR=${OUTPUT_DIR:-runs/action_droid_dist_${RUN_TAG}}
TRIPLETS_ROOT=${TRIPLETS_ROOT:-data/droid_cache/train}
VAL_TRIPLETS_ROOT=${VAL_TRIPLETS_ROOT:-data/droid_cache/val}
TRAIN_MANIFEST_JSONL=${TRAIN_MANIFEST_JSONL:-}
VAL_MANIFEST_JSONL=${VAL_MANIFEST_JSONL:-}
TORCHRUN=${TORCHRUN:-.venv/bin/torchrun}

MANIFEST_ARGS=()
if [[ -n "${TRAIN_MANIFEST_JSONL}" ]]; then
  MANIFEST_ARGS+=(--train_manifest_jsonl "${TRAIN_MANIFEST_JSONL}")
fi
if [[ -n "${VAL_MANIFEST_JSONL}" ]]; then
  MANIFEST_ARGS+=(--val_manifest_jsonl "${VAL_MANIFEST_JSONL}")
fi

if [[ "${MODE}" != "pre_context" && "${MODE}" != "side_adapter" ]]; then
  echo "MODE must be pre_context or side_adapter, got: ${MODE}" >&2
  exit 2
fi

if [[ "${NOISE_MODE}" != "fixed" && "${NOISE_MODE}" != "fresh" ]]; then
  echo "NOISE_MODE must be fixed or fresh, got: ${NOISE_MODE}" >&2
  exit 2
fi

if [[ ! -d "${TRIPLETS_ROOT}" || ! -d "${VAL_TRIPLETS_ROOT}" ]]; then
  echo "missing DROID cache: ${TRIPLETS_ROOT} or ${VAL_TRIPLETS_ROOT}" >&2
  exit 2
fi

if [[ ! -d "Wan2.2-TI2V-5B" ]]; then
  echo "missing checkpoint dir: Wan2.2-TI2V-5B" >&2
  exit 2
fi

if [[ ! -x "${TORCHRUN}" ]]; then
  echo "missing torchrun executable: ${TORCHRUN}" >&2
  exit 2
fi

echo "[launch] host=$(hostname) cwd=$(pwd)"
echo "[launch] code_dir=${CODE_DIR}"
echo "[launch] job=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NUM_NODES:-1} gpus_per_node=${GPUS_PER_NODE}"
echo "[launch] git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[launch] git_status=$(git status --short 2>/dev/null | wc -l | tr -d ' ') dirty files"
echo "[launch] mode=${MODE} sampling_steps=${SAMPLING_STEPS} noise_mode=${NOISE_MODE}"
echo "[launch] total_steps=${TOTAL_STEPS} local_batch_size=${BATCH_SIZE} global_batch_size=$((BATCH_SIZE * GPUS_PER_NODE)) seed=${SEED}"
echo "[launch] lr=${LR} warmup=${WARMUP_STEPS} lr_min_ratio=${LR_MIN_RATIO}"
echo "[launch] max_train_samples=${MAX_TRAIN_SAMPLES} max_val_samples=${MAX_VAL_SAMPLES} num_workers=${NUM_WORKERS}"
echo "[launch] triplets_root=${TRIPLETS_ROOT} val_triplets_root=${VAL_TRIPLETS_ROOT}"
echo "[launch] train_manifest=${TRAIN_MANIFEST_JSONL:-none} val_manifest=${VAL_MANIFEST_JSONL:-none}"
echo "[launch] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[launch] output=${OUTPUT_DIR}"

"${TORCHRUN}" --standalone --nproc_per_node="${GPUS_PER_NODE}" \
  scripts/train_action_conditioned_wan_droid.py \
  --triplets_root "${TRIPLETS_ROOT}" \
  --val_triplets_root "${VAL_TRIPLETS_ROOT}" \
  "${MANIFEST_ARGS[@]}" \
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
  --batch_size "${BATCH_SIZE}" \
  --total_steps "${TOTAL_STEPS}" \
  --lr "${LR}" \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr_min_ratio "${LR_MIN_RATIO}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --max_val_samples "${MAX_VAL_SAMPLES}" \
  --num_workers "${NUM_WORKERS}" \
  --val_interval "${VAL_INTERVAL}" \
  --log_interval "${LOG_INTERVAL}" \
  --ckpt_interval "${CKPT_INTERVAL}"
