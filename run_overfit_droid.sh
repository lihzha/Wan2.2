#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=overfit-droid
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# DROID single-clip overfit (Test 2 / §3.6 M1 for DROID): can the approach
# reconstruct ONE DROID clip from FRESH noise in the CFG-consistent w=5
# regime? Trains on a single episode, then evals that same clip.
#
#   sample.mp4 ≈ ground_truth.mp4  -> approach works on DROID; the sweep
#       failure is the 289-clip mapping (data/capacity), not feasibility.
#   sample.mp4 ≈ null_only.mp4     -> feasibility wall: a single L=1 per-step
#       context can't carry one DROID clip from arbitrary noise. Revisit L>1,
#       horizon, or the z_init regime — NOT more data.
#
# Env overrides:
#   OVERFIT_EP=ep0_v0  TOTAL_STEPS=4000  ARCH_FUSION=concat  ARCH_HEAD=rank1
#   LR=1e-4  WARMUP_STEPS=100  CKPT_INTERVAL=2000  LOG_INTERVAL=20
#   TRAIN_GUIDE_SCALE=5.0  ADAPTOR_INIT=...
#   LOSS_TYPE=rollout  INIT_CONTEXT_PATH=runs/droid_inv/train/ep0_v0/positive_embeddings.pt
#   FIXED_Z_INIT_PATH=runs/droid_inv/train/ep0_v0/positive_embeddings.pt

set -euo pipefail
cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

OVERFIT_EP=${OVERFIT_EP:-ep0_v0}
TOTAL_STEPS=${TOTAL_STEPS:-4000}
LR=${LR:-1e-4}
LR_MU_BETA=${LR_MU_BETA:-}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-2}
WARMUP_STEPS=${WARMUP_STEPS:-100}
CKPT_INTERVAL=${CKPT_INTERVAL:-2000}
VAL_INTERVAL=${VAL_INTERVAL:-500}
LOG_INTERVAL=${LOG_INTERVAL:-20}
GRAD_CLIP=${GRAD_CLIP:-1.0}
ARCH_FUSION=${ARCH_FUSION:-concat}
ARCH_HEAD=${ARCH_HEAD:-rank1}
RANK_K=${RANK_K:-4}
TRAIN_GUIDE_SCALE=${TRAIN_GUIDE_SCALE:-5.0}
ADAPTOR_INIT=${ADAPTOR_INIT:-runs/droid_inv/train/factorization/adaptor_init.pt}
LOSS_TYPE=${LOSS_TYPE:-denoise}
INIT_CONTEXT_PATH=${INIT_CONTEXT_PATH:-}
ZERO_RESIDUAL_ON_INIT_CONTEXT=${ZERO_RESIDUAL_ON_INIT_CONTEXT:-0}
ROLLOUT_UNCOND_GRAD=${ROLLOUT_UNCOND_GRAD:-0}
TRAIN_MU_ONLY=${TRAIN_MU_ONLY:-0}
# FIXED_NOISE=1 => one ε₀ (seed 0) for all steps; deterministic target,
# matches the fixed-seed eval (see §3.14). Strongly recommended for this test.
FIXED_NOISE=${FIXED_NOISE:-1}
FIXED_Z_INIT_PATH=${FIXED_Z_INIT_PATH:-}

FIXED_NOISE_ARG=""
NOISE_TAG=stochnoise
if [ "${FIXED_NOISE}" = "1" ]; then
    FIXED_NOISE_ARG="--fixed_noise"
    NOISE_TAG=fixednoise
fi
# RUN_TAG distinguishes runs so they don't overwrite each other (default
# encodes arch + noise mode; override for anything else).
RUN_TAG=${RUN_TAG:-${ARCH_FUSION}_${ARCH_HEAD}_${NOISE_TAG}}

TRAIN_DIR=runs/droid_overfit_${OVERFIT_EP}_${RUN_TAG}
EVAL_DIR=runs/eval_droid_overfit_${OVERFIT_EP}_${RUN_TAG}

echo "[$(date)] overfit ${OVERFIT_EP} [${RUN_TAG}]: fusion=${ARCH_FUSION} head=${ARCH_HEAD} w=${TRAIN_GUIDE_SCALE}"
echo "          init=${ADAPTOR_INIT}  -> ${TRAIN_DIR}"
echo "          steps=${TOTAL_STEPS} lr=${LR} loss_type=${LOSS_TYPE} fixed_noise=${FIXED_NOISE} ckpt_interval=${CKPT_INTERVAL}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [ ! -d "data/droid_cache/train" ]; then
    echo "ERROR: data/droid_cache/train not found. Run run_precompute_droid.sh first." >&2
    exit 1
fi
if [ ! -d "data/droid_cache/train/${OVERFIT_EP}" ]; then
    echo "ERROR: overfit episode not found: data/droid_cache/train/${OVERFIT_EP}" >&2
    exit 1
fi
if [ ! -d "Wan2.2-TI2V-5B" ]; then
    echo "ERROR: Wan2.2-TI2V-5B checkpoint dir not found." >&2
    exit 1
fi
if [ ! -f "${ADAPTOR_INIT}" ]; then
    echo "ERROR: warm-start not found: ${ADAPTOR_INIT}" >&2
    echo "       Run: sbatch run_inversion_droid.sh" >&2
    echo "       or set ADAPTOR_INIT to an existing adaptor_init.pt." >&2
    exit 1
fi
if [ -n "${INIT_CONTEXT_PATH}" ] && [ ! -f "${INIT_CONTEXT_PATH}" ]; then
    echo "ERROR: init context not found: ${INIT_CONTEXT_PATH}" >&2
    exit 1
fi
if [ -n "${FIXED_Z_INIT_PATH}" ] && [ ! -f "${FIXED_Z_INIT_PATH}" ]; then
    echo "ERROR: fixed z_init source not found: ${FIXED_Z_INIT_PATH}" >&2
    exit 1
fi

LR_MU_BETA_ARG=""
if [ -n "${LR_MU_BETA}" ]; then
    LR_MU_BETA_ARG="--lr_mu_beta ${LR_MU_BETA}"
fi
INIT_CONTEXT_ARG=""
if [ -n "${INIT_CONTEXT_PATH}" ]; then
    INIT_CONTEXT_ARG="--init_context_path ${INIT_CONTEXT_PATH}"
fi
ZERO_RESIDUAL_ARG=""
if [ "${ZERO_RESIDUAL_ON_INIT_CONTEXT}" = "1" ]; then
    ZERO_RESIDUAL_ARG="--zero_residual_on_init_context"
fi
ROLLOUT_UNCOND_ARG=""
if [ "${ROLLOUT_UNCOND_GRAD}" = "1" ]; then
    ROLLOUT_UNCOND_ARG="--rollout_uncond_grad"
fi
TRAIN_MU_ONLY_ARG=""
if [ "${TRAIN_MU_ONLY}" = "1" ]; then
    TRAIN_MU_ONLY_ARG="--train_mu_only"
fi
FIXED_Z_INIT_ARG=""
if [ -n "${FIXED_Z_INIT_PATH}" ]; then
    FIXED_Z_INIT_ARG="--fixed_z_init_path ${FIXED_Z_INIT_PATH}"
    FIXED_NOISE_ARG=""
fi

# ---- 1. overfit a single clip ----
.venv/bin/python scripts/train_adaptor.py \
    --triplets_root data/droid_cache/train \
    --overfit_one ${OVERFIT_EP} \
    --adaptor_init ${ADAPTOR_INIT} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --action_dim 7 --action_len 32 \
    --arch_fusion ${ARCH_FUSION} --arch_head ${ARCH_HEAD} --rank_k ${RANK_K} \
    --train_guide_scale ${TRAIN_GUIDE_SCALE} \
    --cfg_dropout 0.0 \
    ${FIXED_NOISE_ARG} ${FIXED_Z_INIT_ARG} \
    --loss_type ${LOSS_TYPE} ${ROLLOUT_UNCOND_ARG} \
    ${INIT_CONTEXT_ARG} ${ZERO_RESIDUAL_ARG} \
    ${TRAIN_MU_ONLY_ARG} \
    --sampling_steps 25 --shift 5.0 \
    --total_steps ${TOTAL_STEPS} --warmup_steps ${WARMUP_STEPS} \
    --lr ${LR} ${LR_MU_BETA_ARG} --weight_decay ${WEIGHT_DECAY} \
    --grad_clip ${GRAD_CLIP} \
    --val_interval ${VAL_INTERVAL} \
    --ckpt_interval ${CKPT_INTERVAL} --log_interval ${LOG_INTERVAL} \
    --output_dir ${TRAIN_DIR}

EVAL_CKPT=${TRAIN_DIR}/ckpt_best.pt
if [ ! -f "${EVAL_CKPT}" ]; then
    EVAL_CKPT=${TRAIN_DIR}/ckpt_latest.pt
fi
SAMPLE_Z_INIT_ARG=""
if [ -f "${TRAIN_DIR}/fixed_z_init.pt" ]; then
    SAMPLE_Z_INIT_ARG="--sample_z_init_path ${TRAIN_DIR}/fixed_z_init.pt"
fi

# ---- 2. eval that same clip (guide_scale auto = w trained) ----
.venv/bin/python scripts/eval_adaptor.py \
    --triplets_root data/droid_cache/train \
    --eval_names ${OVERFIT_EP} \
    --adaptor_init ${ADAPTOR_INIT} \
    --ckpt_path ${EVAL_CKPT} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --oracle_root runs/droid_inv/train \
    --include_oracle --include_oracle_seed --include_null \
    ${SAMPLE_Z_INIT_ARG} \
    --max_area 61440 \
    --output_dir ${EVAL_DIR}

echo "[$(date)] done."
echo "  compare: ${EVAL_DIR}/val/triplet_${OVERFIT_EP}/{sample,sample_fixed_zinit,ground_truth,oracle,null_only}.mp4"
