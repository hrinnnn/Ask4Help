#!/usr/bin/env bash
set -euo pipefail

ROOT="${ASK4HELP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${XVLA_PYTHON:-/data/zhaozhixuan/envs/xvla_official_5090/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/data/zhaozhixuan/X-VLA}"
STACK_ROOT="${XVLA_STACK_ROOT:-/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1}"
BASE_ROOT="${STACK_ROOT}/temporal_mask_v2"
BASE_CHECKPOINT="${BASE_ROOT}/id_sft_from3500_to10000_official_2gpu_retry1/ckpt-7500"
DETECTOR_ROOT="${BASE_ROOT}/failure_detection_ckpt7500_100id100ood_v1"
RESULT="${XVLA_STACKCUBE_STAGE2_RESULT:-${BASE_ROOT}/stage2_target_ood_v1}"

export PYTHONPATH="${ROOT}:${ROOT}/RLinf${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" "${ROOT}/tools/run_xvla_stackcube_stage2_pipeline.py" \
  --repo "${ROOT}" \
  --python "${PYTHON}" \
  --xvla-root "${XVLA_ROOT}" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --result "${RESULT}" \
  --internal-assets "${DETECTOR_ROOT}/assets_internal/multilayer_detector_assets.pt" \
  --external-assets "${DETECTOR_ROOT}/assets_external_retry1/external_detector_assets.pt" \
  --calibration "${DETECTOR_ROOT}/calibration_q95.json" \
  --id-calibration-summary "${DETECTOR_ROOT}/calibration_id25/summary.json" \
  --id-detector-summary "${DETECTOR_ROOT}/eval_id_100/summary.json" \
  --id-meta "${STACK_ROOT}/manifests/panda_stackcube_id_128.json" \
  --expert-action-budget 2500 \
  --training-steps 2500 \
  --eval-episodes 100
