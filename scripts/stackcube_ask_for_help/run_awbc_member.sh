#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

GPU_ID=${GPU_ID:?Set GPU_ID}
SEED=${SEED:?Set SEED}
INIT_MODEL_PATH=${INIT_MODEL_PATH:?Set INIT_MODEL_PATH}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
ASSISTED_DATASET=${ASSISTED_DATASET:-"${DATA_ROOT}/stackcube_ood_assisted_smoke_step7000"}
PROGRESS_MANIFEST=${PROGRESS_MANIFEST:-"${RESULT_ROOT}/robodopamine/progress_flux_10step.jsonl"}

export ASK4HELP_RLINF_PLACEMENT="${GPU_ID}-${GPU_ID}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
unset CUDA_VISIBLE_DEVICES
cd "${RLINF_ROOT}"
"${PYTHON}" examples/sft/train_vla_sft.py \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_stackcube_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" runner.max_steps=2 runner.save_interval=2 \
  actor.optim.total_training_steps=2 actor.optim.lr_warmup_steps=0 \
  actor.seed="${SEED}" actor.model.model_path="${INIT_MODEL_PATH}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  "data.train_data_paths=[{dataset_path:${ASSISTED_DATASET},weight:1.0}]" \
  awbc.enabled=true awbc.mode=flux_code \
  awbc.progress_manifest="${PROGRESS_MANIFEST}" \
  awbc.progress_threshold=0.01 awbc.expert_sampling_ratio=null
