#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

MEMBER=${MEMBER:?Set MEMBER to 0 or 1}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT}
CUDA_VISIBLE_DEVICES=${GPU_ID:-0} "${PYTHON}" \
  "${ASK4HELP_ROOT}/tools/stackcube_checkpoint_forward_smoke.py" \
  --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
  --norm-stats "${NORM_STATS}" \
  --output "${RESULT_ROOT}/awbc_member_${MEMBER}/reload_forward.json" \
  --seed "$((30000 + MEMBER))"
