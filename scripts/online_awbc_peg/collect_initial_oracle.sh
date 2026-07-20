#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
REPO_ID=${REPO_ID:?Set REPO_ID}
NUM_EPISODES=${NUM_EPISODES:-32}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_online_awbc.py" \
  --mode oracle \
  --repo-id "${REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --episodes "${NUM_EPISODES}" \
  --max-attempts "$((NUM_EPISODES * 10))"
