#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
DATASET=${DATASET:?Set DATASET to an absolute LeRobot dataset path}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${RLINF_ROOT}/toolkits/lerobot/collect_maniskill_plug_lerobot_joint.py" \
  --repo-id "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --split id \
  --num-episodes "${NUM_EPISODES:-32}" \
  --seed "${SEED:-0}" \
  --max-attempts "${MAX_ATTEMPTS:-256}" \
  --chunk-size 10 \
  --overwrite
