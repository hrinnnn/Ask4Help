#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ASSISTED_DATASET=${ASSISTED_DATASET:-"${DATA_ROOT}/stackcube_ood_assisted_smoke_step7000"}
OUT="${RESULT_ROOT}/robodopamine"
GOAL_BANK="${OUT}/goal_bank"
MANIFEST="${OUT}/progress_flux_10step.jsonl"
mkdir -p "${OUT}"

"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_extract_grm_goal_bank.py" \
  --dataset "${EXPERT_DATASET}" --output-dir "${GOAL_BANK}" \
  --episode-index 0 --task-id 0
printf '{"successful_episodes":[]}\n' >"${OUT}/successful_episodes.json"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_annotate_awbc_progress.py" \
  --dataset "${ASSISTED_DATASET}" --goal-bank-dir "${GOAL_BANK}" \
  --grm-endpoint "${GRM_ENDPOINT}" --model-name "${GRM_MODEL_NAME}" \
  --output "${MANIFEST}" --source policy \
  --source-manifest "${RESULT_ROOT}/ood_assisted/progress.jsonl" \
  --stride-steps 5 --lookahead-steps 10 \
  --successful-episodes "${OUT}/successful_episodes.json"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/validate_flux_awbc_manifest.py" \
  --manifest "${MANIFEST}" --output "${OUT}/flux_weight_summary.json"
