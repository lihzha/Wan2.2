#!/bin/bash
# Eval-sweep LAUNCHER (run on the login node). For each architecture
# variant, submits an eval job that auto-fires when its training job
# finishes, via SLURM --dependency=afterok. If a variant's training is
# already done (no job in the queue), the eval is submitted immediately.
#
# Run this AFTER ./run_sweep_arch.sh. It matches training jobs by name, so
# TAG and RANK_K (and the combo list) must match what you trained with.
#
# Usage:
#   ./run_eval_sweep.sh                       # default ladder
#   ./run_eval_sweep.sh "cross_attn perstep"  # subset (each arg "fusion head")
#   TAG=droid RANK_K=4 DRY_RUN=1 ./run_eval_sweep.sh
#
# Watch:  squeue -u $USER     (eval jobs show state "DEPENDENCY" until ready)

set -euo pipefail
cd /scratch/gpfs/AM43/lz3952/Wan2.2

ADAPTOR_INIT=${ADAPTOR_INIT:-runs/droid_inv/train/factorization/adaptor_init.pt}
RANK_K=${RANK_K:-4}
TAG=${TAG:-droid}
DRY_RUN=${DRY_RUN:-0}
# CFG scale for eval. Output dirs are suffixed with it (e.g. _wauto, _w5.0)
# so runs at different scales don't overwrite each other. "auto" matches each
# checkpoint's train_guide_scale (the consistent choice; see docs §3.13).
GUIDE_SCALE=${GUIDE_SCALE:-auto}

# Must match run_sweep_arch.sh's default ladder (override via CLI args).
COMBOS=(
  "concat     rank1"
  "cross_attn rank1"
  "cross_attn rankk"
  "cross_attn perstep"
)
if [ "$#" -gt 0 ]; then
  COMBOS=("$@")
fi

echo "=== eval sweep ==="
echo "  matching training jobs by name (tag=${TAG}, rank_k=${RANK_K})"
echo

for combo in "${COMBOS[@]}"; do
  # shellcheck disable=SC2086
  set -- $combo
  FUSION=$1; HEAD=$2
  NAME="${TAG}_${FUSION}_${HEAD}"
  [ "$HEAD" = "rankk" ] && NAME="${NAME}${RANK_K}"
  CKPT="runs/${NAME}/ckpt_latest.pt"
  OUT="runs/eval_${NAME}_w${GUIDE_SCALE}"
  EVNAME="eval_${NAME}_w${GUIDE_SCALE}"

  # Is the training job still queued/running? (squeue -n matches job name.)
  tid=$(squeue -u "${USER}" -h -n "${NAME}" -o "%i" 2>/dev/null | head -1 || true)

  DEP=""
  if [ -n "${tid}" ]; then
    DEP="--dependency=afterok:${tid}"
    echo "  ${EVNAME}: waits on training job ${tid} (${NAME})"
  elif [ -f "${CKPT}" ]; then
    echo "  ${EVNAME}: training done, ${CKPT} present -> eval now"
  else
    echo "  ${EVNAME}: SKIP — no running train job and no ${CKPT}"
    echo "             (training may have failed; check slurm_outputs/${NAME}/)"
    continue
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    continue
  fi

  mkdir -p "slurm_outputs/${EVNAME}"
  # shellcheck disable=SC2086
  jid=$(sbatch --parsable \
    --job-name="${EVNAME}" \
    ${DEP} \
    --export=ALL,CKPT_PATH="${CKPT}",OUTPUT_DIR="${OUT}",ADAPTOR_INIT="${ADAPTOR_INIT}",GUIDE_SCALE="${GUIDE_SCALE}" \
    run_eval_droid.sh)
  echo "      submitted eval job ${jid}${DEP:+  (${DEP})}"
done

echo
if [ "${DRY_RUN}" = "1" ]; then
  echo "[dry run] nothing submitted. Unset DRY_RUN to launch."
  exit 0
fi
echo "watch:   squeue -u ${USER}"
echo "results: runs/eval_${TAG}_*_w${GUIDE_SCALE}/summary.csv   (sample_ssim_avg vs null_ssim_avg)"
