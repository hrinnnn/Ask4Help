#!/usr/bin/env bash
# Fresh pi0.5 ID-only BC for the controlled PickSingleYCB toy airplane task.
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
GPU_ID=${GPU_ID:-0}
GPU_PLACEMENT=${GPU_PLACEMENT:-"${GPU_ID}-${GPU_ID}"}
SEED=${SEED:-4100}
ID_DATASET=${ID_DATASET:?Set ID_DATASET to the 128-demo ID-only LeRobot dataset}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the pretrained pi0.5 base directory}
NORM_STATS=${NORM_STATS:?Set NORM_STATS to the frozen ID-only norm-stats asset directory}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
MAX_STEPS=${MAX_STEPS:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-32}
RESUME_DIR=${RESUME_DIR:-}

if (( GLOBAL_BATCH_SIZE <= 0 || MICRO_BATCH_SIZE <= 0 || GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be a positive multiple of MICRO_BATCH_SIZE" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${NORM_STATS}/norm_stats.json" ]]; then
  echo "NORM_STATS must be an OpenPI asset directory containing norm_stats.json: ${NORM_STATS}" >&2
  exit 2
fi
resume_args=()
if [[ -n "${RESUME_DIR}" ]]; then
  test -f "${RESUME_DIR}/actor/model_state_dict/full_weights.pt"
  resume_args+=(+runner.resume_dir="${RESUME_DIR}")
fi

unset CUDA_VISIBLE_DEVICES
unset RAY_ADDRESS
export RLINF_CODE_WORKING_DIR="${RLINF_CODE_WORKING_DIR:-${RLINF_ROOT}}"
export ASK4HELP_RLINF_PLACEMENT="${GPU_PLACEMENT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

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
  "data.train_data_paths=[{dataset_path:${ID_DATASET},weight:1.0}]" \
  +data.openpi_mask_padded_action_targets=true \
  +data.openpi_valid_action_horizon=10 \
  actor.model.model_path="${BASE_CHECKPOINT}" \
  actor.model.openpi_data.repo_id="pick_single_ycb_airplane_id" \
  actor.model.openpi_data.default_prompt="pick up the toy airplane and move it to the green goal" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  awbc.enabled=false \
  "${resume_args[@]}"
