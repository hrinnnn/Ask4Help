#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ASK4HELP_ROOT=${ASK4HELP_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
RESULT_ROOT=${RESULT_ROOT:-/mnt/data/ask4help/results/stackcube_ask_for_help/robodopamine_adapted_buffer_v1}
DATA_ROOT=${DATA_ROOT:-/mnt/data/ask4help/datasets/lerobot/local}
PI05_BASE=${PI05_BASE:-/mnt/data/ask4help/models/pi05_base_torch}
NORM_STATS=${NORM_STATS:-/mnt/data/ask4help/datasets/lerobot/local/stackcube_id_128_visual_v1_20260722/norm_stats_id.json}
MEMBER_0=${MEMBER_0:-/mnt/data/ask4help/results/stackcube_id_sft/pi05_id_sft_10000_visual_v1_20260722_dual/member_0/maniskill_stackcube_pi05_id_sft/checkpoints/global_step_7000}
MEMBER_1=${MEMBER_1:-/mnt/data/ask4help/results/stackcube_id_sft/pi05_id_sft_10000_visual_v1_20260722_dual/member_1/maniskill_stackcube_pi05_id_sft/checkpoints/global_step_7000}
EXPERT_DATASET=${EXPERT_DATASET:-/mnt/data/ask4help/datasets/lerobot/local/stackcube_id_128_visual_v1_20260722}
GRM_MODEL=${GRM_MODEL:-/mnt/data/ask4help/models/Robo-Dopamine-GRM-2.0-4B-Preview}
GRM_ADAPTER_NAME=${GRM_ADAPTER_NAME:-stackcube-grm-lora}
GRM_ADAPTER_DIR=${GRM_ADAPTER_DIR:-${RESULT_ROOT}/robodopamine_adaptation/stackcube-grm-lora}
GRM_MODEL_NAME=${GRM_MODEL_NAME:-${GRM_ADAPTER_NAME}}
GRM_ENDPOINT=${GRM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
mkdir -p "${RESULT_ROOT}"
