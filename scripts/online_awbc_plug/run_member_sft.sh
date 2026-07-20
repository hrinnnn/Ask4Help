#!/usr/bin/env bash
set -euo pipefail

# Run members serially.  RLinf's Cluster first connects to any existing Ray
# cluster, so an old two-GPU Ray head can silently turn a single-GPU command
# into a two-rank FSDP job.  Stopping Ray and constraining CUDA before launch
# makes world_size=1 an explicit precondition.
ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
GPU_ID=${GPU_ID:?Set GPU_ID to 0 or 1}
SEED=${SEED:?Set SEED to 1000 or 1001}
ID_DATASET=${ID_DATASET:?Set ID_DATASET}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
PI05_BASE=${PI05_BASE:?Set PI05_BASE}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}

"${RLINF_ROOT}/.venv/bin/ray" stop --force || true
rm -rf "/tmp/ask4help_ray_member_${SEED}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export RAY_TMPDIR="/tmp/ask4help_ray_member_${SEED}"
export RAY_ADDRESS=""
export EMBODIED_PATH="${RLINF_ROOT}/examples/sft"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

"${PYTHON}" examples/sft/train_vla_sft.py \
  --config-path "${RLINF_ROOT}/examples/sft/config" \
  --config-name maniskill_plug_awbc_sft_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}" \
  runner.max_steps=250 \
  runner.save_interval=50 \
  actor.optim.total_training_steps=250 \
  actor.seed="${SEED}" \
  "data.train_data_paths=[{dataset_path:${ID_DATASET},weight:1.0}]" \
  actor.model.model_path="${PI05_BASE}" \
  actor.model.openpi_data.norm_stats_path="${NORM_STATS}" \
  awbc.enabled=false
