#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --job-name=zinit-probe
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

DROID_ROOT=${DROID_ROOT:-/scratch/gpfs/AM43/yy4041/data/droid_ctrl_world}
SPLIT=${SPLIT:-train}
VIEW=${VIEW:-0}
EPISODES=${EPISODES:-0,1,2,3,4,5,6,7}
SEEDS=${SEEDS:-0,1,2,3}
OUT_ROOT=${OUT_ROOT:-runs/zinit_probe_droid}
ANALYSIS_ROOTS=${ANALYSIS_ROOTS:-"${OUT_ROOT}/${SPLIT} runs/droid_inv/${SPLIT} runs/batch_inv_positive runs/identifiability"}

echo "[$(date)] z-init probe episodes=${EPISODES} seeds=${SEEDS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/collect_zinit_inversion_droid.py \
    --droid_root "${DROID_ROOT}" \
    --split "${SPLIT}" \
    --view "${VIEW}" \
    --episodes "${EPISODES}" \
    --seeds "${SEEDS}" \
    --ckpt_dir Wan2.2-TI2V-5B \
    --output_root "${OUT_ROOT}" \
    --max_area 61440 \
    --L_pos 1 \
    --sampling_steps 25 \
    --guide_scale 5.0 \
    --overwrite

echo "[$(date)] analyzing z-init artifacts"
.venv/bin/python scripts/analyze_zinit_distribution.py \
    --roots ${ANALYSIS_ROOTS} \
    --out_dir "${OUT_ROOT}/analysis" \
    --random_pairs 256

echo "[$(date)] done -> ${OUT_ROOT}/analysis/summary.json"
