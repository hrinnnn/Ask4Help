#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
require_path "${LIBERO_SPATIAL_DEMO_DIR}" "LIBERO spatial demos"
mkdir -p "$(dirname "${GRM_GOAL_BANK_DIR}")"

python "${RLINF_DIR}/tools/libero_extract_grm_goal_bank.py" \
  --demo-root "${LIBERO_SPATIAL_DEMO_DIR}" \
  --output-dir "${GRM_GOAL_BANK_DIR}" \
  --task-suite-name libero_spatial

find "${GRM_GOAL_BANK_DIR}" -maxdepth 2 -type f | sort

