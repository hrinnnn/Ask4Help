#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
require_path "${LIBERO_GOAL_DEMO_DIR}" "LIBERO Goal demos"
mkdir -p "$(dirname "${GRM_LIBERO_GOAL_BANK_DIR}")"

python "${RLINF_DIR}/tools/libero_extract_grm_goal_bank.py" \
  --demo-root "${LIBERO_GOAL_DEMO_DIR}" \
  --output-dir "${GRM_LIBERO_GOAL_BANK_DIR}" \
  --task-suite-name libero_goal

find "${GRM_LIBERO_GOAL_BANK_DIR}" -maxdepth 2 -type f | sort
