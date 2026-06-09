#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=precompute-droid
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Encode a verification subset of the DROID droid_ctrl_world dataset with
# Wan's VAE at native 320x192, view 0. Writes the existing per-dir cache
# format under data/droid_cache/{train,val}/ so train_adaptor.py needs no
# new dataloader.
#
# Env overrides:
#   TRAIN_LIMIT=300   VAL_LIMIT=30   VIEW=0

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

DROID_ROOT=/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world
TRAIN_LIMIT=${TRAIN_LIMIT:-300}
VAL_LIMIT=${VAL_LIMIT:-30}
VIEW=${VIEW:-0}

echo "[$(date)] precompute DROID train (limit ${TRAIN_LIMIT}) + val (limit ${VAL_LIMIT}), view ${VIEW}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/precompute_features_droid.py \
    --droid_root ${DROID_ROOT} \
    --split train --limit ${TRAIN_LIMIT} --view ${VIEW} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --out_root data/droid_cache

.venv/bin/python scripts/precompute_features_droid.py \
    --droid_root ${DROID_ROOT} \
    --split val --limit ${VAL_LIMIT} --view ${VIEW} \
    --ckpt_dir Wan2.2-TI2V-5B \
    --out_root data/droid_cache

echo "[$(date)] done"
