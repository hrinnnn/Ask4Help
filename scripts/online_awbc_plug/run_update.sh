#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
DATA_PATHS=${DATA_PATHS:?Set DATA_PATHS as a Hydra list of accumulated datasets}
PROGRESS_MANIFEST=${PROGRESS_MANIFEST:?Set PROGRESS_MANIFEST}
PI05_WARM_START=${PI05_WARM_START:?Set PI05_WARM_START}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
MODE=${MODE:-arm_paper_exact}
MAX_STEPS=${MAX_STEPS:-50}
SEED=${SEED:-1000}

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" examples/sft/train_vla_sft.py \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_plug_awbc_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" runner.max_steps="${MAX_STEPS}" \
  runner.save_interval="${MAX_STEPS}" actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.seed="${SEED}" actor.model.model_path="${PI05_WARM_START}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  "data.train_data_paths=${DATA_PATHS}" awbc.enabled=true awbc.mode="${MODE}" \
  awbc.progress_manifest="${PROGRESS_MANIFEST}" awbc.expert_sampling_ratio=null
