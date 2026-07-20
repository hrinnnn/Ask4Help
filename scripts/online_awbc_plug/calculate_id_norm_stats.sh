#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
DATASET=${DATASET:?Set DATASET to the ID-only LeRobot dataset path}
OUTPUT_PATH=${OUTPUT_PATH:?Set OUTPUT_PATH to norm_stats.json}
CONFIG_NAME=${CONFIG_NAME:-pi05_rlt_maniskill_joint}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${RLINF_ROOT}/toolkits/lerobot/calculate_norm_stats.py" \
  --config-name "${CONFIG_NAME}" \
  --repo-id "${DATASET}"
test -s "${DATASET}/norm_stats.json"
mkdir -p "$(dirname "${OUTPUT_PATH}")"
cp "${DATASET}/norm_stats.json" "${OUTPUT_PATH}"
