#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=precompute-feats
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# One-time preprocessing: VAE-encode I_0 and the 33-frame reference clip
# for every triplet. After this, training never touches Wan's VAE again.
# Outputs z_I0.pt + z_video.pt + meta.json under each data/triplets/<i>/.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/${SLURM_JOB_NAME}

echo "[$(date)] precomputing VAE features for $(ls -d data/triplets/*/ 2>/dev/null | wc -l) triplets"

.venv/bin/python scripts/precompute_features.py \
    --triplets_root data/triplets \
    --ckpt_dir Wan2.2-TI2V-5B \
    --max_area 230400

echo "[$(date)] done"
