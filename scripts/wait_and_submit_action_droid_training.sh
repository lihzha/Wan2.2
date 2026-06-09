#!/usr/bin/env bash
# Wait for the DROID window cache, run a smoke train job, then submit full train.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2

TRAIN_SUMMARY=${TRAIN_SUMMARY:-runs/droid_window_plans/train_v0_33f_stride4_cap500g_fp16_top50.summary.json}
VAL_SUMMARY=${VAL_SUMMARY:-runs/droid_window_plans/val_v0_33f_stride4_cap20g_fp16_top50.summary.json}
TRIPLETS_ROOT=${TRIPLETS_ROOT:-data/droid_cache_windows_v0/train}
VAL_TRIPLETS_ROOT=${VAL_TRIPLETS_ROOT:-data/droid_cache_windows_v0/val}
TRAIN_MANIFEST_JSONL=${TRAIN_MANIFEST_JSONL:-runs/droid_window_plans/train_v0_33f_stride4_cap500g_fp16_top50.jsonl}
VAL_MANIFEST_JSONL=${VAL_MANIFEST_JSONL:-runs/droid_window_plans/val_v0_33f_stride4_cap20g_fp16_top50.jsonl}
POLL_SECONDS=${POLL_SECONDS:-900}
LOG_PATH=${LOG_PATH:-runs/droid_window_plans/action_droid_training_submitter.log}

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

json_int() {
  local path=$1
  local key=$2
  .venv/bin/python - <<PY "${path}" "${key}"
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(int(d[sys.argv[2]]))
PY
}

count_dirs() {
  local path=$1
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -mindepth 1 -maxdepth 1 -type d | wc -l
}

wait_for_job_clean() {
  local job_id=$1
  log "waiting for job ${job_id}"
  while squeue -h -j "${job_id}" | grep -q .; do
    squeue -j "${job_id}" -o "%.18i %.9P %.24j %.8T %.20M %.40R" | tee -a "${LOG_PATH}" || true
    sleep "${POLL_SECONDS}"
  done
  local bad
  bad=$(sacct -P -n -j "${job_id}" --format=JobID,State,ExitCode 2>/dev/null \
    | awk -F'|' '
      /\.batch$/ || /\.extern$/ {next}
      $2 != "COMPLETED" || $3 != "0:0" {print}
    ' || true)
  if [[ -n "${bad}" ]]; then
    log "job ${job_id} had non-clean tasks:"
    echo "${bad}" | tee -a "${LOG_PATH}"
    exit 1
  fi
  log "job ${job_id} completed cleanly"
}

submit_sbatch() {
  local out
  out=$(sbatch "$@")
  echo "${out}" | tee -a "${LOG_PATH}"
  echo "${out}" | awk '{print $4}'
}

if [[ ! -f "${TRAIN_SUMMARY}" || ! -f "${VAL_SUMMARY}" ]]; then
  log "missing plan summaries: ${TRAIN_SUMMARY} / ${VAL_SUMMARY}"
  exit 2
fi

expected_train=$(json_int "${TRAIN_SUMMARY}" selected_windows)
expected_val=$(json_int "${VAL_SUMMARY}" selected_windows)
log "waiting for cache train=${expected_train} val=${expected_val}"

while true; do
  train_count=$(count_dirs "${TRIPLETS_ROOT}" | tr -d ' ')
  val_count=$(count_dirs "${VAL_TRIPLETS_ROOT}" | tr -d ' ')
  log "cache counts train=${train_count}/${expected_train} val=${val_count}/${expected_val}"
  if (( train_count >= expected_train && val_count >= expected_val )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

log "cache complete; submitting DROID training smoke"
smoke_job=$(submit_sbatch \
  --time=00:30:00 \
  --job-name=act-droid-win-smoke \
  --export=ALL,TRIPLETS_ROOT="${TRIPLETS_ROOT}",VAL_TRIPLETS_ROOT="${VAL_TRIPLETS_ROOT}",TRAIN_MANIFEST_JSONL="${TRAIN_MANIFEST_JSONL}",VAL_MANIFEST_JSONL="${VAL_MANIFEST_JSONL}",MODE=side_adapter,RUN_TAG=side_bn512h8_L0-29_fresh_25step_window_smoke,TOTAL_STEPS=5,SAMPLING_STEPS=25,SEED=0,NOISE_MODE=fresh,SIDE_ADAPTER_LAYERS=0-29,SIDE_ADAPTER_HIDDEN=512,SIDE_ADAPTER_HEADS=8,LR=5e-5,WARMUP_STEPS=5,LR_MIN_RATIO=0.1,MAX_TRAIN_SAMPLES=128,MAX_VAL_SAMPLES=4,VAL_INTERVAL=5,LOG_INTERVAL=1,CKPT_INTERVAL=5 \
  run_action_conditioned_droid.sh)
wait_for_job_clean "${smoke_job}"

log "smoke clean; submitting full DROID training"
full_job=$(submit_sbatch \
  --time=36:00:00 \
  --job-name=act-droid-win-10k \
  --export=ALL,TRIPLETS_ROOT="${TRIPLETS_ROOT}",VAL_TRIPLETS_ROOT="${VAL_TRIPLETS_ROOT}",TRAIN_MANIFEST_JSONL="${TRAIN_MANIFEST_JSONL}",VAL_MANIFEST_JSONL="${VAL_MANIFEST_JSONL}",MODE=side_adapter,RUN_TAG=side_bn512h8_L0-29_fresh_25step_window_top50_10k_lr5e-5,TOTAL_STEPS=10000,SAMPLING_STEPS=25,SEED=0,NOISE_MODE=fresh,SIDE_ADAPTER_LAYERS=0-29,SIDE_ADAPTER_HIDDEN=512,SIDE_ADAPTER_HEADS=8,LR=5e-5,WARMUP_STEPS=500,LR_MIN_RATIO=0.1,MAX_TRAIN_SAMPLES=0,MAX_VAL_SAMPLES=32,VAL_INTERVAL=1000,LOG_INTERVAL=20,CKPT_INTERVAL=1000 \
  run_action_conditioned_droid.sh)
log "submitted full DROID training job ${full_job}"
