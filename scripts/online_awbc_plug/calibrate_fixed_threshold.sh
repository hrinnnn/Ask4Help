#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
MEMBER_0=${MEMBER_0:?Set MEMBER_0}
MEMBER_1=${MEMBER_1:?Set MEMBER_1}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_online_awbc.py" \
  --mode calibrate --task plug --split id \
  --member-0 "${MEMBER_0}" --member-1 "${MEMBER_1}" --norm-stats "${NORM_STATS}" \
  --output-dir "${OUTPUT_DIR}" --seed 10000 --successes 20 --max-attempts 200 \
  --samples-per-episode 5 --quantile 0.95 --num-action-samples 5
