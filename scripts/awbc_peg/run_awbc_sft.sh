#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
EXPERT_DATASET=${EXPERT_DATASET:?Set EXPERT_DATASET}
POLICY_DATASET=${POLICY_DATASET:?Set POLICY_DATASET}
PROGRESS_MANIFEST=${PROGRESS_MANIFEST:?Set PROGRESS_MANIFEST}
PI05_WARM_START=${PI05_WARM_START:?Set PI05_WARM_START}
NORM_STATS_PATH=${NORM_STATS_PATH:?Set NORM_STATS_PATH}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
AWBC_MODE=${AWBC_MODE:-arm_paper_exact}
MAX_STEPS=${MAX_STEPS:-500}
EXPERT_SAMPLING_RATIO=${EXPERT_SAMPLING_RATIO:-0.5}

TRAIN_DATA_OVERRIDE="[{dataset_path:${EXPERT_DATASET},weight:1.0},{dataset_path:${POLICY_DATASET},weight:1.0}]"

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" examples/sft/train_vla_sft.py \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_awbc_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" \
  runner.max_steps="${MAX_STEPS}" \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  "data.train_data_paths=${TRAIN_DATA_OVERRIDE}" \
  actor.model.model_path="${PI05_WARM_START}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS_PATH}" \
  awbc.progress_manifest="${PROGRESS_MANIFEST}" \
  awbc.mode="${AWBC_MODE}" \
  awbc.expert_sampling_ratio="${EXPERT_SAMPLING_RATIO}"
