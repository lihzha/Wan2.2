#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=train-droid
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Train the rank-1 adaptor on the DROID cache at 320x192.
# Differences from run_train_adaptor.sh:
#   --action_dim 7        DROID actions are 6 cartesian + 1 gripper
#   separate train/ val/  via --val_triplets_root
# μ/β warm-start reused from the 480²-grid factorization (text-space,
# resolution-independent; see docs/adaptor_design.md §3.10).

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

OUTPUT_DIR=${OUTPUT_DIR:-runs/adaptor_droid_v0}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-1}
# In-domain warm-start re-derived at 320x192 (see §3.11). Override if needed.
ADAPTOR_INIT=${ADAPTOR_INIT:-runs/droid_inv/train/factorization/adaptor_init.pt}
# Architecture variant (see §3.12). Defaults reproduce the original net.
ARCH_FUSION=${ARCH_FUSION:-concat}
ARCH_HEAD=${ARCH_HEAD:-rank1}
RANK_K=${RANK_K:-4}
# CFG scale the loss is consistent with. Must match the warm-start's
# inversion scale (adaptor_init.pt -> w=5) and the eval. See §3.13.
TRAIN_GUIDE_SCALE=${TRAIN_GUIDE_SCALE:-5.0}
# FIXED_NOISE=1 => one fixed epsilon_0 (seed 0) every step, matching the
# fixed-seed eval and making the target a deterministic function of
# (action,I_0). This is the default for DROID because eval_adaptor.py is a
# deterministic replay from seed 0; stochastic-noise training learns a
# washed-out noise-marginal context that can look good in one-step loss while
# failing decoded train-set replay. See docs/adaptor_design.md §3.14.
FIXED_NOISE=${FIXED_NOISE:-1}
FIXED_NOISE_ARG=""
[ "${FIXED_NOISE}" = "1" ] && FIXED_NOISE_ARG="--fixed_noise"

if [ ! -d "data/droid_cache/train" ] || [ ! -d "data/droid_cache/val" ]; then
    echo "ERROR: data/droid_cache/{train,val} not found. Run run_precompute_droid.sh first." >&2
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

echo "[$(date)] training DROID adaptor -> ${OUTPUT_DIR}"
echo "          init=${ADAPTOR_INIT} fusion=${ARCH_FUSION} head=${ARCH_HEAD} fixed_noise=${FIXED_NOISE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/train_adaptor.py \
    --triplets_root data/droid_cache/train \
    --val_triplets_root data/droid_cache/val \
    --adaptor_init ${ADAPTOR_INIT} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --action_dim 7 \
    --action_len 32 \
    --sampling_steps 25 \
    --shift 5.0 \
    --arch_fusion ${ARCH_FUSION} \
    --arch_head ${ARCH_HEAD} \
    --rank_k ${RANK_K} \
    --train_guide_scale ${TRAIN_GUIDE_SCALE} \
    --cfg_dropout 0.0 \
    ${FIXED_NOISE_ARG} \
    --batch_size ${BATCH_SIZE} \
    --total_steps ${TOTAL_STEPS} \
    --lr 1e-4 \
    --val_interval 1000 \
    --ckpt_interval 5000 \
    --output_dir ${OUTPUT_DIR}

echo "[$(date)] done"
