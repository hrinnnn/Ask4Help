#!/usr/bin/env bash
set -euo pipefail

ROOT="${ASK4HELP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON="${XVLA_PYTHON:-/data/zhaozhixuan/envs/xvla_official_5090/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/data/zhaozhixuan/X-VLA}"
STACK_ROOT="${XVLA_STACK_ROOT:-/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_stackcube_v1}"
BASE_ROOT="${STACK_ROOT}/temporal_mask_v2"
BASE_CHECKPOINT="${XVLA_STACKCUBE_BASE_CHECKPOINT:-${BASE_ROOT}/id_sft_from3500_to10000_official_2gpu_retry1/ckpt-7500}"
ID_META="${XVLA_STACKCUBE_ID_META:-${STACK_ROOT}/manifests/panda_stackcube_id_128.json}"
RESULT="${XVLA_STACKCUBE_STAGE2_RESULT:-${BASE_ROOT}/stackcube_target_ood_timing_v1}"

GPU_ARGS=()
if [[ -n "${XVLA_STAGE2_GPUS:-}" ]]; then
  read -r -a GPU_IDS <<<"${XVLA_STAGE2_GPUS}"
  GPU_ARGS=(--gpus "${GPU_IDS[@]}")
fi

export PYTHONPATH="${ROOT}:${ROOT}/RLinf${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON}" "${ROOT}/tools/run_xvla_stackcube_stage2_pipeline.py" \
  --repo "${ROOT}" \
  --python "${PYTHON}" \
  --xvla-root "${XVLA_ROOT}" \
  --checkpoint "${BASE_CHECKPOINT}" \
  --result "${RESULT}" \
  --id-meta "${ID_META}" \
  --expert-action-budget 2000 \
  --pool-action-target 2200 \
  --training-steps 10000 \
  --cohort-screen-episodes 300 \
  --cohort-size 200 \
  "${GPU_ARGS[@]}"
