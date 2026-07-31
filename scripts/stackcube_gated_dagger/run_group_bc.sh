#!/usr/bin/env bash
# Ordinary BC for one StackCube DAgger group.  The two sources are sampled 1:1
# in every micro-batch; AWBC is explicitly disabled.
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help-online-awbc}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-3000}
ID_REPLAY=${ID_REPLAY:?Set ID_REPLAY to the frozen 128-demo ID dataset}
NEW_EXPERT_DATASET=${NEW_EXPERT_DATASET:?Set NEW_EXPERT_DATASET to this group expert-only dataset}
BASE_CHECKPOINT=${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the immutable member0 step-7000 checkpoint}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
MAX_STEPS=${MAX_STEPS:-500}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}
RESUME_DIR=${RESUME_DIR:-}

unset CUDA_VISIBLE_DEVICES
export RAY_ADDRESS=""
export ASK4HELP_RLINF_PLACEMENT="${GPU_ID}-${GPU_ID}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

resume_args=()
if [[ -n "${RESUME_DIR}" ]]; then
  test -f "${RESUME_DIR}/actor/model_state_dict/full_weights.pt"
  resume_args+=(+runner.resume_dir="${RESUME_DIR}")
fi

"${PYTHON}" "${RLINF_ROOT}/examples/sft/train_vla_sft.py" \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_stackcube_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" \
  runner.max_steps="${MAX_STEPS}" \
  runner.save_interval="${SAVE_INTERVAL}" \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.seed="${SEED}" \
  "data.train_data_paths=[{dataset_path:${ID_REPLAY},weight:1.0},{dataset_path:${NEW_EXPERT_DATASET},weight:1.0}]" \
  +data.openpi_source_balanced=true \
  +data.openpi_exclude_padded_action_targets=true \
  +data.openpi_valid_action_horizon=10 \
  actor.model.model_path="${BASE_CHECKPOINT}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  awbc.enabled=false \
  "${resume_args[@]}"
