#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
GPU_ID=${GPU_ID:?Set GPU_ID to 0 or 1}
SEED=${SEED:?Set SEED}
EXTERNAL_RAY=${EXTERNAL_RAY:-1}
ID_DATASET=${ID_DATASET:?Set ID_DATASET}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
PI05_BASE=${PI05_BASE:?Set PI05_BASE}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
INIT_MODEL_PATH=${INIT_MODEL_PATH:-"${PI05_BASE}"}
RESUME_DIR=${RESUME_DIR:-}
MAX_STEPS=${MAX_STEPS:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}

if [ "${EXTERNAL_RAY}" != "1" ]; then
  "${RLINF_ROOT}/.venv/bin/ray" stop --force || true
fi
unset CUDA_VISIBLE_DEVICES
export RAY_ADDRESS=""
export ASK4HELP_RLINF_PLACEMENT="${GPU_ID}-${GPU_ID}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

resume_args=()
if [ -n "${RESUME_DIR}" ]; then
  test -d "${RESUME_DIR}/actor"
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
  "data.train_data_paths=[{dataset_path:${ID_DATASET},weight:1.0}]" \
  actor.model.model_path="${INIT_MODEL_PATH}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  awbc.enabled=false \
  "${resume_args[@]}"

