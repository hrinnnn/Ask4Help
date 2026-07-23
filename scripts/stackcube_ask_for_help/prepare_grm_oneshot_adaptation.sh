#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ROBO_DOPAMINE_DIR=${ROBO_DOPAMINE_DIR:-"${ASK4HELP_ROOT}/external/Robo-Dopamine"}
GRM_PYTHON=${GRM_PYTHON:-/opt/conda/envs/robo-dopamine/bin/python}
RAW_ROOT="${RESULT_ROOT}/robodopamine_adaptation/raw_data_local_video_v2"
TRAIN_ROOT="${RESULT_ROOT}/robodopamine_adaptation/train_data_local_video_v2"
EPISODE_INDEX=${ONESHOT_EPISODE_INDEX:-0}
EPISODE_SEED=${ONESHOT_EPISODE_SEED:-0}
PRIVILEGED_EVENTS=${ONESHOT_PRIVILEGED_EVENTS:-"${RESULT_ROOT}/robodopamine_adaptation/oneshot_privileged_events.jsonl"}

test -f "${ROBO_DOPAMINE_DIR}/dataset/utils/0_preprocess_data.py"
test -x "${GRM_PYTHON}"
mkdir -p "${RAW_ROOT}"
if [[ ! -f "${PRIVILEGED_EVENTS}" ]]; then
  "${PYTHON}" "${ASK4HELP_ROOT}/tools/replay_stackcube_privileged_events.py" \
    --dataset "${EXPERT_DATASET}" --episode-index "${EPISODE_INDEX}" --seed "${EPISODE_SEED}" --output "${PRIVILEGED_EVENTS}"
fi
"${PYTHON}" "${ASK4HELP_ROOT}/tools/export_stackcube_grm_oneshot.py" \
  --dataset "${EXPERT_DATASET}" --privileged-events "${PRIVILEGED_EVENTS}" \
  --episode-index "${EPISODE_INDEX}" --seed "${EPISODE_SEED}" --output-dir "${RAW_ROOT}" --local-staging-dir /root

cd "${ROBO_DOPAMINE_DIR}/dataset"
"${GRM_PYTHON}" -m utils.0_preprocess_data --raw_dir "${RAW_ROOT}" --cvt_dir "${TRAIN_ROOT}" --sample_factor 20
"${GRM_PYTHON}" -m utils.1_generate_data --base-dir "${TRAIN_ROOT}" --score-bins 25 --gap-bins 4 --oversample-factor 100 --zero-ratio 0.05 --max_sample_num 1000 --workers 4
"${GRM_PYTHON}" -m utils.2_posprocess_data --root-dir "${TRAIN_ROOT}" --merged-json "${TRAIN_ROOT}/train_jsons/finetune_data_wo_replace.json" --final-json "${TRAIN_ROOT}/train_jsons/finetune_data_final.json" --replace-prob 0.75 --seed 42
test -s "${TRAIN_ROOT}/train_jsons/finetune_data_final.json"
sample_count="$("${GRM_PYTHON}" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${TRAIN_ROOT}/train_jsons/finetune_data_final.json")"
if [[ "${sample_count}" -le 0 ]]; then
  echo "Official Robo-Dopamine pair generation produced zero one-shot samples." >&2
  exit 1
fi
printf '%s\n' "${sample_count}" >"${RESULT_ROOT}/robodopamine_adaptation/official_pair_count.txt"
cp -a "${RAW_ROOT}/oneshot_manifest.json" "${RESULT_ROOT}/robodopamine_adaptation/"
