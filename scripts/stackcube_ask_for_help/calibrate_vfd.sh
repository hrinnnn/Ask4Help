#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="${RESULT_ROOT}/calibration"
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES=${GPU_ID:-0} "${PYTHON}" \
  "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_online_awbc.py" \
  --mode calibrate --task stack --split id \
  --output-dir "${OUT}" --member-0 "${MEMBER_0}" --member-1 "${MEMBER_1}" \
  --pi05-base "${PI05_BASE}" --norm-stats "${NORM_STATS}" \
  --successes 5 --samples-per-episode 5 --quantile 0.95 \
  --chunk-size 5 --num-action-samples 5 --seed 10000 --max-attempts 20
