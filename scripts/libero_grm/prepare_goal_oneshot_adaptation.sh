#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
require_path "${LIBERO_GOAL_DEMO_DIR}" "LIBERO Goal demos"
require_path "${ROBO_DOPAMINE_DIR}/dataset/utils/0_preprocess_data.py" "official Robo-Dopamine checkout"

TASK_ID="${ONESHOT_TASK_ID:-0}"
RAW_DIR="${GRM_ONESHOT_ROOT}/libero_goal/task_$(printf '%03d' "${TASK_ID}")/raw_data"
TRAIN_DIR="${GRM_ONESHOT_ROOT}/libero_goal/task_$(printf '%03d' "${TASK_ID}")/train_data"

python "${RLINF_DIR}/tools/libero_prepare_dopamine_oneshot.py" \
  --demo-root "${LIBERO_GOAL_DEMO_DIR}" \
  --output-dir "${RAW_DIR}" \
  --task-suite libero_goal \
  --task-id "${TASK_ID}" \
  --segment-count "${ONESHOT_SEGMENT_COUNT:-8}"

cd "${ROBO_DOPAMINE_DIR}/dataset"
python -m utils.0_preprocess_data --raw_dir "${RAW_DIR}" --cvt_dir "${TRAIN_DIR}" --sample_factor "${ONESHOT_SAMPLE_FACTOR:-20}"
python -m utils.1_generate_data --base-dir "${TRAIN_DIR}" --score-bins 25 --gap-bins 4 --oversample-factor 100 --zero-ratio 0.05 --max_sample_num "${ONESHOT_MAX_SAMPLES:-1000}" --workers "${ONESHOT_WORKERS:-4}"
python -m utils.2_posprocess_data \
  --root-dir "${TRAIN_DIR}" \
  --merged-json "${TRAIN_DIR}/train_jsons/finetune_data_wo_replace.json" \
  --final-json "${TRAIN_DIR}/train_jsons/finetune_data_final.json" \
  --replace-prob 0.75 \
  --seed 42

sample_count="$(${RLINF_DIR}/.venv/bin/python -c 'import json, sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${TRAIN_DIR}/train_jsons/finetune_data_final.json")"
if [[ "${sample_count}" -le 0 ]]; then
  echo "One-shot preparation produced zero training samples." >&2
  exit 1
fi

echo "Prepared one-shot adaptation data: ${TRAIN_DIR}/train_jsons/finetune_data_final.json"
