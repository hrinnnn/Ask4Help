#!/usr/bin/env bash
# Ordinary source-balanced BC for one airplane expert-data group.
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help-pick-airplane-four-group}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-/root/Ask4Help-online-awbc/RLinf/.venv/bin/python}
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-5100}
ID_REPLAY=${ID_REPLAY:?Set ID_REPLAY to the frozen 98-demo no-180 ID dataset}
NEW_EXPERT_DATASET=${NEW_EXPERT_DATASET:?Set NEW_EXPERT_DATASET to this group's expert dataset}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the immutable airplane step5000 checkpoint}
NORM_STATS=${NORM_STATS:?Set NORM_STATS to the frozen original ID norm asset}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to a new path}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}

if (( MICRO_BATCH_SIZE <= 0 || MICRO_BATCH_SIZE % 2 != 0 )); then
  echo "MICRO_BATCH_SIZE must be a positive even number" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE <= 0 || GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi
test -f "${BASE_CHECKPOINT}/actor/model_state_dict/full_weights.pt"
test -f "${NORM_STATS}/norm_stats.json"

unset CUDA_VISIBLE_DEVICES RAY_ADDRESS
export RLINF_CODE_WORKING_DIR="${RLINF_CODE_WORKING_DIR:-${RLINF_ROOT}}"
export ASK4HELP_RLINF_PLACEMENT="${GPU_ID}-${GPU_ID}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${ASK4HELP_ROOT}:${RLINF_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" "${RLINF_ROOT}/examples/sft/train_vla_sft.py" \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_stackcube_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" \
  runner.max_steps="${MAX_STEPS}" \
  runner.save_interval="${SAVE_INTERVAL}" \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  actor.micro_batch_size="${MICRO_BATCH_SIZE}" \
  actor.seed="${SEED}" \
  "data.train_data_paths=[{dataset_path:${ID_REPLAY},weight:1.0},{dataset_path:${NEW_EXPERT_DATASET},weight:1.0}]" \
  +data.openpi_source_balanced=true \
  +data.openpi_mask_padded_action_targets=true \
  +data.openpi_valid_action_horizon=10 \
  actor.model.model_path="${BASE_CHECKPOINT}" \
  actor.model.openpi_data.repo_id="pick_single_ycb_airplane_id" \
  actor.model.openpi_data.default_prompt="pick up the toy airplane and move it to the green goal" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  awbc.enabled=false
