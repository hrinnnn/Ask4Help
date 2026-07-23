#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

OUT="${RESULT_ROOT}/ood_assisted"
REPO_ID=${REPO_ID:-stackcube_ood_assisted_smoke_step7000}
DURABLE_DATASET=${DURABLE_DATASET:-"${DATA_ROOT}/${REPO_ID}"}
mkdir -p "${OUT}"
CUDA_VISIBLE_DEVICES=${GPU_ID:-0} "${PYTHON}" \
  "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_online_awbc.py" \
  --mode online --task stack --split ood \
  --output-dir "${OUT}" --repo-id "${REPO_ID}" \
  --member-0 "${MEMBER_0}" --member-1 "${MEMBER_1}" \
  --pi05-base "${PI05_BASE}" --norm-stats "${NORM_STATS}" \
  --threshold-path "${RESULT_ROOT}/calibration/fixed_vfd_threshold.json" \
  --episodes 2 --max-attempts 10 --seed 20000 \
  --chunk-size 5 --num-action-samples 5 --require-expert-trigger

SOURCE_DATASET="${HOME}/.cache/huggingface/lerobot/${REPO_ID}"
test -d "${SOURCE_DATASET}"
if [[ ! -e "${DURABLE_DATASET}" ]]; then
  cp -a "${SOURCE_DATASET}" "${DURABLE_DATASET}"
fi
test -d "${DURABLE_DATASET}/data"
