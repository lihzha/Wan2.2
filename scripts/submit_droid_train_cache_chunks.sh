#!/usr/bin/env bash
# Sequentially submit DROID train-cache precompute chunks under array limits.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2

PLAN_JSONL=${PLAN_JSONL:-runs/droid_window_plans/train_v0_33f_stride4_cap500g_fp16_top50.jsonl}
OUT_ROOT=${OUT_ROOT:-data/droid_cache_windows_v0}
TOTAL_SHARDS=${TOTAL_SHARDS:-128}
CHUNK_SIZE=${CHUNK_SIZE:-24}
CONCURRENCY=${CONCURRENCY:-3}
START_SHARD=${START_SHARD:-0}
FIRST_WAIT_JOB=${FIRST_WAIT_JOB:-}
TIME_LIMIT=${TIME_LIMIT:-00:45:00}
SAVE_DTYPE=${SAVE_DTYPE:-float16}
POLL_SECONDS=${POLL_SECONDS:-300}
LOG_PATH=${LOG_PATH:-runs/droid_window_plans/train_cache_chunk_submitter.log}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-run_precompute_droid_window_plan_array.sh}
SBATCH_QOS=${SBATCH_QOS:-}
SBATCH_PARTITION=${SBATCH_PARTITION:-}

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

wait_for_job() {
  local job_id=$1
  if [[ -z "${job_id}" ]]; then
    log "empty job id; refusing to poll all jobs"
    exit 1
  fi
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

submit_chunk() {
  local start=$1
  local end=$2
  local tag=$3
  local out
  local sbatch_args=(
    --time="${TIME_LIMIT}" \
    --job-name="droid-train-cache-${tag}" \
    --array="${start}-${end}%${CONCURRENCY}" \
    --export=ALL,PLAN_JSONL="${PLAN_JSONL}",OUT_ROOT="${OUT_ROOT}",NUM_SHARDS="${TOTAL_SHARDS}",SAVE_DTYPE="${SAVE_DTYPE}",LIMIT=0
  )
  if [[ -n "${SBATCH_QOS}" ]]; then
    sbatch_args+=(--qos="${SBATCH_QOS}")
  fi
  if [[ -n "${SBATCH_PARTITION}" ]]; then
    sbatch_args+=(--partition="${SBATCH_PARTITION}")
  fi
  if ! out=$(sbatch "${sbatch_args[@]}" "${SBATCH_SCRIPT}" 2>&1); then
    log "sbatch failed for shard range ${start}-${end}:"
    echo "${out}" | tee -a "${LOG_PATH}"
    exit 1
  fi
  echo "${out}" >> "${LOG_PATH}"
  local job_id
  job_id=$(echo "${out}" | awk '/Submitted batch job/ {print $4; exit}')
  if [[ -z "${job_id}" ]]; then
    log "could not parse sbatch job id from: ${out}"
    exit 1
  fi
  echo "${job_id}"
}

if [[ ! -f "${PLAN_JSONL}" ]]; then
  log "missing plan ${PLAN_JSONL}"
  exit 2
fi

log "starting chunk submitter plan=${PLAN_JSONL} out=${OUT_ROOT} total_shards=${TOTAL_SHARDS} chunk_size=${CHUNK_SIZE} start=${START_SHARD} script=${SBATCH_SCRIPT} qos=${SBATCH_QOS:-default} partition=${SBATCH_PARTITION:-default}"

if [[ -n "${FIRST_WAIT_JOB}" ]]; then
  wait_for_job "${FIRST_WAIT_JOB}"
fi

chunk_idx=0
start=${START_SHARD}
while (( start < TOTAL_SHARDS )); do
  end=$(( start + CHUNK_SIZE - 1 ))
  if (( end >= TOTAL_SHARDS )); then
    end=$(( TOTAL_SHARDS - 1 ))
  fi
  tag=$(printf "s%03d-%03d" "${start}" "${end}")
  log "submitting shard range ${start}-${end}"
  job_id=$(submit_chunk "${start}" "${end}" "${tag}")
  wait_for_job "${job_id}"
  start=$(( end + 1 ))
  chunk_idx=$(( chunk_idx + 1 ))
done

log "all train-cache chunks submitted and completed"
