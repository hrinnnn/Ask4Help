#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
DATASET=${DATASET:?Set DATASET}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
EPISODES=${EPISODES:-128}
SEED=${SEED:-0}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${RLINF_ROOT}/toolkits/lerobot/collect_maniskill_stack_cube_lerobot_joint.py" \
  --repo-id "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-episodes "${EPISODES}" \
  --seed "${SEED}" \
  --max-attempts "$((EPISODES * 8))" \
  --image-size 384 \
  --control-freq 10 \
  --max-episode-steps 100 \
  --save-videos

