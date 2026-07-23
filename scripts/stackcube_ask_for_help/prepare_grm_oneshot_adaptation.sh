#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ROBO_DOPAMINE_DIR=${ROBO_DOPAMINE_DIR:-"${ASK4HELP_ROOT}/external/Robo-Dopamine"}
RAW_ROOT="${RESULT_ROOT}/robodopamine_adaptation/raw_data"
TRAIN_ROOT="${RESULT_ROOT}/robodopamine_adaptation/train_data"
EPISODE_INDEX=${ONESHOT_EPISODE_INDEX:-0}
EPISODE_SEED=${ONESHOT_EPISODE_SEED:-0}
PRIVILEGED_EVENTS=${ONESHOT_PRIVILEGED_EVENTS:?Set ONESHOT_PRIVILEGED_EVENTS to the successful ID privileged event sidecar.}

test -f "${ROBO_DOPAMINE_DIR}/dataset/utils/0_preprocess_data.py"
test -f "${PRIVILEGED_EVENTS}"
mkdir -p "${RAW_ROOT}"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/export_stackcube_grm_oneshot.py" \
  --dataset "${EXPERT_DATASET}" --privileged-events "${PRIVILEGED_EVENTS}" \
  --episode-index "${EPISODE_INDEX}" --seed "${EPISODE_SEED}" --output-dir "${RAW_ROOT}"

cd "${ROBO_DOPAMINE_DIR}/dataset"
python -m utils.0_preprocess_data --raw_dir "${RAW_ROOT}" --cvt_dir "${TRAIN_ROOT}" --sample_factor 20
python -m utils.1_generate_data --base-dir "${TRAIN_ROOT}" --score-bins 25 --gap-bins 4 --oversample-factor 100 --zero-ratio 0.05 --max_sample_num 1000 --workers 4
python -m utils.2_posprocess_data --root-dir "${TRAIN_ROOT}" --merged-json "${TRAIN_ROOT}/train_jsons/finetune_data_wo_replace.json" --final-json "${TRAIN_ROOT}/train_jsons/finetune_data_final.json" --replace-prob 0.75 --seed 42
test -s "${TRAIN_ROOT}/train_jsons/finetune_data_final.json"
cp -a "${RAW_ROOT}/oneshot_manifest.json" "${RESULT_ROOT}/robodopamine_adaptation/"
