#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-droid
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Evaluate one trained DROID adaptor on the held-out val cache.
# No --include_oracle (DROID has no per-episode inversion run); native
# 320x192 via --max_area 61440. The eval script reads action_dim + the
# arch variant from the checkpoint's saved args, so nothing arch-specific
# needs to be passed here.
#
# Required env: CKPT_PATH, OUTPUT_DIR
# Optional env: ADAPTOR_INIT, EVAL_NAMES, MAX_AREA, GUIDE_SCALE

set -euo pipefail
cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

CKPT_PATH=${CKPT_PATH:?set CKPT_PATH=runs/<variant>/ckpt_latest.pt}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR=runs/eval_<variant>}
ADAPTOR_INIT=${ADAPTOR_INIT:-runs/droid_inv/train/factorization/adaptor_init.pt}
EVAL_NAMES=${EVAL_NAMES:-$(ls data/droid_cache/val)}
MAX_AREA=${MAX_AREA:-61440}
# "auto" => eval_adaptor reads the checkpoint's train_guide_scale so eval
# matches how the model was trained (see §3.13). Set a number to force.
GUIDE_SCALE=${GUIDE_SCALE:-auto}
# Also score a few TRAINING episodes (same job) for the fit-vs-gen gap.
# Set N_TRAIN_EVAL=0 to skip.
TRAIN_ROOT=${TRAIN_ROOT:-data/droid_cache/train}
N_TRAIN_EVAL=${N_TRAIN_EVAL:-5}

echo "[$(date)] eval ${CKPT_PATH} -> ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -f "${CKPT_PATH}" ]; then
    echo "ERROR: checkpoint not found: ${CKPT_PATH}" >&2
    exit 1
fi
if [ ! -f "${ADAPTOR_INIT}" ]; then
    echo "ERROR: warm-start not found: ${ADAPTOR_INIT}" >&2
    echo "       This must be the adaptor_init.pt used to instantiate the checkpoint architecture." >&2
    exit 1
fi
if [ ! -d "data/droid_cache/val" ]; then
    echo "ERROR: data/droid_cache/val not found." >&2
    exit 1
fi

TRAIN_ARGS=""
if [ "${N_TRAIN_EVAL}" -gt 0 ] && [ -d "${TRAIN_ROOT}" ]; then
    TRAIN_ARGS="--train_triplets_root ${TRAIN_ROOT} --n_train_eval ${N_TRAIN_EVAL}"
fi

# "auto" => omit --guide_scale so eval_adaptor uses its -1 default and reads
# the checkpoint's train_guide_scale; otherwise force the given number.
GS_ARG=""
if [ "${GUIDE_SCALE}" != "auto" ]; then
    GS_ARG="--guide_scale ${GUIDE_SCALE}"
fi

.venv/bin/python scripts/eval_adaptor.py \
    --triplets_root data/droid_cache/val \
    --adaptor_init ${ADAPTOR_INIT} \
    --ckpt_path ${CKPT_PATH} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --eval_names ${EVAL_NAMES} \
    --include_null \
    --max_area ${MAX_AREA} \
    ${GS_ARG} \
    ${TRAIN_ARGS} \
    --output_dir ${OUTPUT_DIR}

echo "[$(date)] done -> ${OUTPUT_DIR}/summary.csv"
