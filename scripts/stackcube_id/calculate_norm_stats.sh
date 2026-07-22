#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
DATASET=${DATASET:?Set DATASET}
OUTPUT_PATH=${OUTPUT_PATH:-"${DATASET}/norm_stats_id.json"}
DATASET="${DATASET}" OUTPUT_PATH="${OUTPUT_PATH}" \
  bash "${ASK4HELP_ROOT}/scripts/online_awbc_plug/calculate_id_norm_stats.sh"

