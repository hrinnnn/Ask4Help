#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

SOCKET="${RESULT_ROOT}/worker.sock"
WORKER_LOG="${RESULT_ROOT}/resident_worker.log"
PROGRESS="${RESULT_ROOT}/robodopamine/progress_10step.jsonl"
mkdir -p "${RESULT_ROOT}" "${RESULT_ROOT}/robodopamine"
if [[ -e "${SOCKET}" ]]; then
  echo "Refusing to reuse an existing worker socket: ${SOCKET}" >&2
  exit 1
fi

if [[ "${PREPARE_ADAPTATION:-1}" == 1 ]]; then
  "${ASK4HELP_ROOT}/scripts/stackcube_ask_for_help/prepare_grm_oneshot_adaptation.sh"
  "${ASK4HELP_ROOT}/scripts/stackcube_ask_for_help/finetune_grm_lora_oneshot.sh"
fi
"${ASK4HELP_ROOT}/scripts/stackcube_ask_for_help/start_grm_endpoint.sh"

CUDA_VISIBLE_DEVICES=0 nohup "${PYTHON}" "${ASK4HELP_ROOT}/tools/stackcube_online_awbc_worker.py" \
  --socket "${SOCKET}" --member0 "${MEMBER_0}" --member1 "${MEMBER_1}" \
  --pi05-base "${PI05_BASE}" --norm-stats "${NORM_STATS}" --expert-dataset "${EXPERT_DATASET}" --output-dir "${RESULT_ROOT}" \
  >"${WORKER_LOG}" 2>&1 &
echo $! >"${RESULT_ROOT}/resident_worker.pid"
for _ in $(seq 1 120); do
  [[ -S "${SOCKET}" ]] && break
  sleep 5
done
[[ -S "${SOCKET}" ]]

client() { "${PYTHON}" "${ASK4HELP_ROOT}/tools/stackcube_online_awbc_client.py" --socket "${SOCKET}" --command "$1" --args-json "${2:-{}}"; }
client preflight
client calibrate '{"seeds":[10000,10001,10002,10003,10004,10005,10006,10007,10008,10009,10010,10011,10012,10013,10014,10015,10016,10017,10018,10019],"successes":5,"samples_per_episode":5,"quantile":0.95}'
client collect '{"seeds":[20000,20001,20002,20003,20004,20005,20006,20007,20008,20009],"trajectories":2}'

RAW_DATASET="${RESULT_ROOT}/raw_online_archive/dataset"
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_extract_grm_goal_bank.py" --dataset "${EXPERT_DATASET}" --output-dir "${RESULT_ROOT}/robodopamine/goal_bank" --episode-index "${ONESHOT_EPISODE_INDEX:-0}" --task-id 0
CUDA_VISIBLE_DEVICES='' "${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_annotate_awbc_progress.py" \
  --dataset "${RAW_DATASET}" --goal-bank-dir "${RESULT_ROOT}/robodopamine/goal_bank" --grm-endpoint "${GRM_ENDPOINT}" --model-name "${GRM_ADAPTER_NAME}" \
  --output "${PROGRESS}" --cache-path "${RESULT_ROOT}/robodopamine/grm_cache.jsonl" --source policy \
  --source-manifest "${RESULT_ROOT}/raw_online_archive/progress_privileged.jsonl" --stride-steps 10 --lookahead-steps 10 --successful-episodes "${RESULT_ROOT}/raw_online_archive/successful_episodes.json" --request-workers 3
client admit_quality_buffer "{\"progress_path\":\"${PROGRESS}\",\"minimum_weight\":0.1}"
client update
client forward_smoke
client shutdown
