#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
DATASET_PATH=${DATASET_PATH:?Set DATASET_PATH to the local LeRobot Peg smoke dataset}
PI05_BASE=${PI05_BASE:?Set PI05_BASE to the official OpenPI pi0.5 base weights}
NORM_STATS_PATH=${NORM_STATS_PATH:?Set NORM_STATS_PATH to the dataset norm_stats.json}
REPO_ID=${REPO_ID:?Set REPO_ID to the local LeRobot dataset repo id}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the Stage 1 output directory}

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" examples/sft/train_vla_sft.py \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_rlt_stage1_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" \
  runner.max_steps=2 \
  runner.save_interval=1 \
  data.train_data_paths[0].dataset_path="${DATASET_PATH}" \
  actor.model.model_path="${PI05_BASE}" \
  actor.model.openpi_data.repo_id="${REPO_ID}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS_PATH}"
