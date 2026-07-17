#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help-vfd}
RLINF_ROOT=${RLINF_ROOT:-${ASK4HELP_ROOT}/RLinf}
PYTHON=${PYTHON:-${RLINF_ROOT}/.venv/bin/python}
MEMBER_0=${MEMBER_0:?Set MEMBER_0 to the reference pi0.5 checkpoint}
MEMBER_1=${MEMBER_1:?Set MEMBER_1 to the comparison pi0.5 checkpoint}
NORM_STATS_PATH=${NORM_STATS_PATH:?Set NORM_STATS_PATH to norm_stats.json}
OUTPUT_PATH=${OUTPUT_PATH:-/mnt/data/ask4help/results/pi05_vfd_stage3/result.json}

export ASK4HELP_ROOT
export RLINF_ROOT
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_smoke.py" \
  --member-0 "${MEMBER_0}" \
  --member-1 "${MEMBER_1}" \
  --norm-stats "${NORM_STATS_PATH}" \
  --output "${OUTPUT_PATH}" \
  "$@"

