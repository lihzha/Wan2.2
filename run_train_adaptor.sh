#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --job-name=train-adaptor
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# End-to-end training for the rank-1 trajectory adaptor.
# Wan TI2V-5B is frozen; gradients flow only into the adaptor. Loss is
# Wan's own flow-matching denoising loss (see docs/adaptor_design.md §2.5).
#
# Defaults match the inversion configuration that produced
# runs/batch_inv_positive/factorization/adaptor_init.pt:
#   sampling_steps=25, shift=5.0, max_area=230400.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

OUTPUT_DIR=${OUTPUT_DIR:-runs/adaptor_train_v0}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
BATCH_SIZE=${BATCH_SIZE:-1}

echo "[$(date)] training adaptor -> ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/train_adaptor.py \
    --triplets_root data/triplets \
    --adaptor_init runs/batch_inv_positive/factorization/adaptor_init.pt \
    --ckpt_dir Wan2.2-TI2V-5B \
    --val_names 45 46 47 48 49 \
    --sampling_steps 25 \
    --shift 5.0 \
    --batch_size ${BATCH_SIZE} \
    --total_steps ${TOTAL_STEPS} \
    --lr 1e-4 \
    --weight_decay 1e-2 \
    --warmup_steps 500 \
    --cfg_dropout 0.1 \
    --val_interval 1000 \
    --ckpt_interval 5000 \
    --output_dir ${OUTPUT_DIR}

echo "[$(date)] done"
