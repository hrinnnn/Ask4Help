#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
MEMBER_0=${MEMBER_0:?Set MEMBER_0}
MEMBER_1=${MEMBER_1:?Set MEMBER_1}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
THRESHOLD_PATH=${THRESHOLD_PATH:?Set THRESHOLD_PATH}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
REPO_ID=${REPO_ID:?Set REPO_ID}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_online_awbc.py" \
  --mode online \
  --member-0 "${MEMBER_0}" \
  --member-1 "${MEMBER_1}" \
  --norm-stats "${NORM_STATS}" \
  --threshold-path "${THRESHOLD_PATH}" \
  --repo-id "${REPO_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --episodes "${EPISODES_PER_ROUND:-2}" \
  --num-action-samples "${VFD_SAMPLES:-5}"
