#!/bin/bash
#SBATCH --partition=ailab
#SBATCH --qos=ailab
#SBATCH --account=am43
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=embed-sweep
#SBATCH --output=slurm_outputs/%x/out_log_%x_%A_%a.out
#SBATCH --array=0-5
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=lz3952@princeton.edu

# Sweep over scripts/embedding_search.py.
#
# Sweep dims (cartesian product, then drop invalid combos):
#   - triplet : auto-discovered from data/triplets/*/ that contain
#               BOTH I_0.png and I_T.png
#   - seed    : 0, 1, 2
#   - config  : one of
#       embed_empty   : --mode embed   --init empty                (no heuristic)
#       embed_random  : --mode embed   --init random               (no heuristic)
#       noise         : --mode noise   --heuristic "<prompt>"      (needs prompt.txt)
#       null_inv      : --mode null_inversion --heuristic "<prompt>" --guide_scale 5.0
#                                                                  (needs prompt.txt)
#
# Heuristic source: data/triplets/<id>/prompt.txt. Triplets without that file
# only run the embed_empty + embed_random configs.
#
# Each run produces (in OUT_DIR):
#   - I_0.png, I_T_target.png, config.json, loss_log.csv
#   - embed/noise modes : embedding_init.pt|noise_init.pt + *_final.pt + step_*.png
#                          + final_video.mp4 + final_decoded.png  (--validate)
#   - null_inversion    : pivot_clean_video.mp4, inversion_trajectory.pt,
#                          null_embeddings.pt, reconstruction.mp4 + last_frame.png,
#                          heuristic_video.mp4 + last_frame.png   (--validate)
#
# --array sizing: current state has 1 valid triplet (001) with no prompt.txt
#   → 1 × 3 seeds × 2 no-heur configs = 6 tasks → --array=0-5.
# After adding triplets / prompt.txt files, recount and either edit the
# directive above or submit with: `sbatch --array=0-$((N-1)) run_search.sh`.
# Per-triplet count: 6 (no prompt.txt) or 12 (with prompt.txt).
# Out-of-range tasks exit cleanly and print the correct array size.

set -euo pipefail

cd /scratch/gpfs/AM43/lz3952/Wan2.2
mkdir -p slurm_outputs/embed-sweep

# ---------------------------------------------------------------------------
# Configurations (parallel arrays).
# ---------------------------------------------------------------------------
CONFIG_NAMES=(embed_empty embed_random noise null_inv)
CONFIG_MODES=(embed       embed        noise null_inversion)
CONFIG_INITS=(empty       random       empty empty)
CONFIG_NEEDS_HEUR=(0      0            1     1)
CONFIG_GUIDE=(1.0         1.0          1.0   5.0)
CONFIG_GUIDE_VAL=(1.0     1.0          1.0   5.0)

N_CONFIG=${#CONFIG_NAMES[@]}
SEEDS=(0 1 2)
N_SEED=${#SEEDS[@]}

# ---------------------------------------------------------------------------
# Discover triplets and build the (triplet, config_idx, seed) task list,
# skipping configs that need a heuristic when prompt.txt is absent.
# ---------------------------------------------------------------------------
mapfile -t TRIPLETS < <(
    for d in data/triplets/*/; do
        if [[ -f "${d}I_0.png" && -f "${d}I_T.png" ]]; then
            basename "$d"
        fi
    done | sort
)

if [[ ${#TRIPLETS[@]} -eq 0 ]]; then
    echo "[sweep] No valid triplets under data/triplets/ (need I_0.png + I_T.png)." >&2
    exit 1
fi

TASKS=()  # entries: "<triplet>:<config_idx>:<seed>"
for trip in "${TRIPLETS[@]}"; do
    has_prompt=0
    [[ -s "data/triplets/${trip}/prompt.txt" ]] && has_prompt=1
    for ci in $(seq 0 $((N_CONFIG - 1))); do
        if [[ "${CONFIG_NEEDS_HEUR[$ci]}" -eq 1 && "$has_prompt" -eq 0 ]]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            TASKS+=("${trip}:${ci}:${seed}")
        done
    done
done

TOTAL=${#TASKS[@]}
echo "[sweep] discovered triplets=(${TRIPLETS[*]}) → TOTAL=${TOTAL} tasks; recommended --array=0-$((TOTAL - 1))"

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]]; then
    echo "[sweep] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} >= TOTAL=${TOTAL}; nothing to do."
    exit 0
fi

# ---------------------------------------------------------------------------
# Resolve this task.
# ---------------------------------------------------------------------------
IFS=':' read -r TRIPLET CI SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
CONFIG_NAME=${CONFIG_NAMES[$CI]}
MODE=${CONFIG_MODES[$CI]}
INIT=${CONFIG_INITS[$CI]}
NEEDS_HEUR=${CONFIG_NEEDS_HEUR[$CI]}
GUIDE=${CONFIG_GUIDE[$CI]}
GUIDE_VAL=${CONFIG_GUIDE_VAL[$CI]}

OUT_DIR="runs/sweep/${TRIPLET}/${CONFIG_NAME}_seed${SEED}"
mkdir -p "$OUT_DIR"

HEUR_ARG=()
if [[ "$NEEDS_HEUR" -eq 1 ]]; then
    HEURISTIC=$(< "data/triplets/${TRIPLET}/prompt.txt")
    HEUR_ARG=(--heuristic "$HEURISTIC")
    echo "[sweep] heuristic = ${HEURISTIC}"
fi

echo "[sweep] task=${SLURM_ARRAY_TASK_ID}/${TOTAL} triplet=${TRIPLET} config=${CONFIG_NAME} seed=${SEED}"
echo "[sweep] out_dir=${OUT_DIR}"
nvidia-smi || true

# ---------------------------------------------------------------------------
# Common args + per-mode dispatch.
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --start_frame "data/triplets/${TRIPLET}/I_0.png"
    --goal_frame  "data/triplets/${TRIPLET}/I_T.png"
    --ckpt_dir    Wan2.2-TI2V-5B/
    --output_dir  "$OUT_DIR"
    --mode        "$MODE"
    --frames      17
    --sampling_steps 8
    --guide_scale "$GUIDE"
    --max_area    $((480 * 480))
    --shift       5.0
    --seed        "$SEED"
    --validate
    --sampling_steps_val 40
    --guide_scale_val    "$GUIDE_VAL"
)

if [[ "$MODE" == "null_inversion" ]]; then
    # Null-text inversion: no Adam loop on text/noise; per-timestep ∅_t opt.
    EXTRA_ARGS=(
        --null_inner_iters 10
        --null_lr 1e-2
        --target_blend linear
        --inversion_guide_scale 1.0
    )
else
    # embed / noise: outer Adam on text-embed or initial-noise.
    EXTRA_ARGS=(
        --init        "$INIT"
        --L_opt       16
        --loss        lpips
        --num_iters   300
        --lr          1e-2
        --log_every   25
    )
fi

python scripts/embedding_search.py \
    "${COMMON_ARGS[@]}" \
    "${HEUR_ARG[@]}" \
    "${EXTRA_ARGS[@]}"
