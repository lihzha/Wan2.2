#!/bin/bash
# Architecture sweep LAUNCHER (run on the login node — NOT a SLURM job
# itself). Submits one training job per architecture variant by sbatch-ing
# run_train_droid.sh with the arch env vars set.
#
# Each variant trains on the DROID 320x192 cache and writes to its own
# runs/<TAG>_<fusion>_<head>/ dir, so they can be compared side by side.
#
# Usage:
#   # default 4-variant escalation ladder, in-domain warm-start:
#   ./run_sweep_arch.sh
#
#   # custom combos (each arg is "fusion head"):
#   ./run_sweep_arch.sh "cross_attn perstep" "cross_attn rankk"
#
#   # overrides:
#   ADAPTOR_INIT=runs/droid_inv/train/factorization/adaptor_init.pt \
#   TOTAL_STEPS=30000 RANK_K=4 TAG=droid DRY_RUN=1 ./run_sweep_arch.sh
#
# Watch:  squeue -u $USER
# Then eval each:  see the eval block printed at the end.

set -euo pipefail
cd /scratch/gpfs/AM43/lz3952/Wan2.2

ADAPTOR_INIT=${ADAPTOR_INIT:-runs/droid_inv/train/factorization/adaptor_init.pt}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
RANK_K=${RANK_K:-4}
TAG=${TAG:-droid}
DRY_RUN=${DRY_RUN:-0}

# Default escalation ladder: control -> +fusion -> +rankK -> drop linearity.
COMBOS=(
  "concat     rank1"
  "cross_attn rank1"
  "cross_attn rankk"
  "cross_attn perstep"
)
# Override combos from CLI args, e.g. ./run_sweep_arch.sh "cross_attn perstep"
if [ "$#" -gt 0 ]; then
  COMBOS=("$@")
fi

if [ ! -f "$ADAPTOR_INIT" ]; then
  echo "ERROR: warm-start not found: $ADAPTOR_INIT"
  echo "A bad/missing warm-start sinks every architecture. Re-derive it first:"
  echo "    sbatch run_inversion_droid.sh        # writes that adaptor_init.pt"
  echo "(or set ADAPTOR_INIT=... to point at an existing one)"
  exit 1
fi

echo "=== arch sweep ==="
echo "  warm-start : $ADAPTOR_INIT"
echo "  total_steps: $TOTAL_STEPS   rank_k: $RANK_K   tag: $TAG"
echo "  variants   : ${#COMBOS[@]}"
echo

SUBMITTED=()
for combo in "${COMBOS[@]}"; do
  # shellcheck disable=SC2086
  set -- $combo
  FUSION=$1; HEAD=$2
  NAME="${TAG}_${FUSION}_${HEAD}"
  [ "$HEAD" = "rankk" ] && NAME="${NAME}${RANK_K}"
  OUT="runs/${NAME}"

  # The job's #SBATCH --output dir must exist before the job starts.
  mkdir -p "slurm_outputs/${NAME}"

  echo "  ${NAME}  (fusion=${FUSION} head=${HEAD}) -> ${OUT}"
  if [ "$DRY_RUN" = "1" ]; then
    continue
  fi
  jid=$(sbatch --parsable \
    --job-name="${NAME}" \
    --export=ALL,OUTPUT_DIR="${OUT}",ADAPTOR_INIT="${ADAPTOR_INIT}",ARCH_FUSION="${FUSION}",ARCH_HEAD="${HEAD}",RANK_K="${RANK_K}",TOTAL_STEPS="${TOTAL_STEPS}" \
    run_train_droid.sh)
  echo "      submitted job ${jid}"
  SUBMITTED+=("${NAME}:${jid}")
done

echo
if [ "$DRY_RUN" = "1" ]; then
  echo "[dry run] nothing submitted. Unset DRY_RUN to launch."
  exit 0
fi
echo "submitted ${#SUBMITTED[@]} jobs: ${SUBMITTED[*]}"
echo "watch:  squeue -u ${USER}"
echo
echo "after training, eval each variant on held-out val (no oracle for DROID):"
for combo in "${COMBOS[@]}"; do
  # shellcheck disable=SC2086
  set -- $combo
  HEAD=$2; NAME="${TAG}_$1_${HEAD}"; [ "$HEAD" = "rankk" ] && NAME="${NAME}${RANK_K}"
  echo "  .venv/bin/python scripts/eval_adaptor.py \\"
  echo "      --triplets_root data/droid_cache/val \\"
  echo "      --adaptor_init ${ADAPTOR_INIT} \\"
  echo "      --ckpt_path runs/${NAME}/ckpt_latest.pt \\"
  echo "      --ckpt_dir Wan2.2-TI2V-5B --include_null --max_area 61440 \\"
  echo "      --eval_names \$(ls data/droid_cache/val) \\"
  echo "      --output_dir runs/eval_${NAME}"
done
