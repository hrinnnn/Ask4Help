#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
EXPERT_DATASET=${EXPERT_DATASET:?Set EXPERT_DATASET}
POLICY_DATASET=${POLICY_DATASET:?Set POLICY_DATASET}
POLICY_SUCCESS_EPISODES=${POLICY_SUCCESS_EPISODES:?Set POLICY_SUCCESS_EPISODES}
GRM_ENDPOINT=${GRM_ENDPOINT:?Set GRM_ENDPOINT}
GRM_MODEL_NAME=${GRM_MODEL_NAME:?Set GRM_MODEL_NAME}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT}

GOAL_BANK="${OUTPUT_ROOT}/goal_bank"
EXPERT_MANIFEST="${OUTPUT_ROOT}/expert_progress.jsonl"
POLICY_MANIFEST="${OUTPUT_ROOT}/policy_progress.jsonl"
COMBINED_MANIFEST="${OUTPUT_ROOT}/combined_progress.jsonl"
mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_extract_grm_goal_bank.py" \
  --dataset "${EXPERT_DATASET}" \
  --output-dir "${GOAL_BANK}" \
  --episode-index 0 \
  --task-id 0

"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_annotate_awbc_progress.py" \
  --dataset "${EXPERT_DATASET}" \
  --goal-bank-dir "${GOAL_BANK}" \
  --grm-endpoint "${GRM_ENDPOINT}" \
  --model-name "${GRM_MODEL_NAME}" \
  --output "${EXPERT_MANIFEST}" \
  --source expert \
  --stride-steps 5 \
  --assume-all-success

"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_annotate_awbc_progress.py" \
  --dataset "${POLICY_DATASET}" \
  --goal-bank-dir "${GOAL_BANK}" \
  --grm-endpoint "${GRM_ENDPOINT}" \
  --model-name "${GRM_MODEL_NAME}" \
  --output "${POLICY_MANIFEST}" \
  --source policy \
  --stride-steps 5 \
  --successful-episodes "${POLICY_SUCCESS_EPISODES}"

"${PYTHON}" "${ASK4HELP_ROOT}/tools/combine_awbc_manifests.py" \
  --input "${EXPERT_MANIFEST}" \
  --input "${POLICY_MANIFEST}" \
  --output "${COMBINED_MANIFEST}"

printf '%s\n' "${COMBINED_MANIFEST}"
