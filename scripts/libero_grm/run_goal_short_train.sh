#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
export_rll_env
require_path "${PI05_MODEL_PATH}" "pi0.5 model"
require_path "${GRM_LIBERO_GOAL_BANK_DIR}/task_000/meta.json" "LIBERO Goal GRM goal bank"

GOAL_NUM_ENVS="${GOAL_NUM_ENVS:-8}"
GOAL_ROLLOUT_EPOCH="${GOAL_ROLLOUT_EPOCH:-4}"
GOAL_MAX_STEPS="${GOAL_MAX_STEPS:-20}"
GOAL_MAX_EPOCHS="${GOAL_MAX_EPOCHS:-20}"
GOAL_MAX_EPISODE_STEPS="${GOAL_MAX_EPISODE_STEPS:-320}"
GOAL_MICRO_BATCH="${GOAL_MICRO_BATCH:-8}"
GOAL_GLOBAL_BATCH="${GOAL_GLOBAL_BATCH:-32}"
GOAL_REWARD_WEIGHT="${GOAL_REWARD_WEIGHT:-1.0}"
GOAL_GRM_INTERVAL_CHUNKS="${GOAL_GRM_INTERVAL_CHUNKS:-2}"
GOAL_TASK_ID_FILTER="${GOAL_TASK_ID_FILTER:-0}"

CONFIG_NAME="libero_goal_ppo_openpi_pi05_dopamine_runtime"
CONFIG_PATH="${RLINF_DIR}/examples/embodiment/config/${CONFIG_NAME}.yaml"
RUN_ID="$(timestamp)"
LOG="${GOAL_TRAIN_LOG:-/root/rlinf_pi05_libero_goal_dopamine_${RUN_ID}.log}"
LOGDIR="${GOAL_TRAIN_LOGDIR:-${RLINF_DIR}/logs/libero_goal_dopamine_${RUN_ID}}"
python "${SCRIPT_DIR}/write_runtime_config.py" \
  --output "${CONFIG_PATH}" \
  --task-suite libero_goal \
  --experiment-name "libero_goal_ppo_openpi_pi05_dopamine" \
  --pi05-model-path "${PI05_MODEL_PATH}" \
  --goal-bank-dir "${GRM_LIBERO_GOAL_BANK_DIR}" \
  --grm-model-name "${GRM_MODEL_NAME}" \
  --grm-endpoint "${GRM_ENDPOINT}" \
  --metrics-log-path "${LOGDIR}/grm_metrics.jsonl" \
  --train-gpu-rank "${TRAIN_GPU_RANK}" \
  --num-envs "${GOAL_NUM_ENVS}" \
  --rollout-epoch "${GOAL_ROLLOUT_EPOCH}" \
  --max-epochs "${GOAL_MAX_EPOCHS}" \
  --max-steps "${GOAL_MAX_STEPS}" \
  --max-episode-steps "${GOAL_MAX_EPISODE_STEPS}" \
  --micro-batch-size "${GOAL_MICRO_BATCH}" \
  --global-batch-size "${GOAL_GLOBAL_BATCH}" \
  --reward-weight "${GOAL_REWARD_WEIGHT}" \
  --grm-interval-chunks "${GOAL_GRM_INTERVAL_CHUNKS}" \
  --task-id-filter "${GOAL_TASK_ID_FILTER}"
cd "${RLINF_DIR}"
python examples/embodiment/train_embodied_agent.py \
  --config-path "${RLINF_DIR}/examples/embodiment/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOGDIR}" 2>&1 | tee "${LOG}"

DEST="${RESULTS_DIR}/$(date +%Y%m%d_%H%M)_libero_goal_dopamine_${GOAL_NUM_ENVS}env_${GOAL_MAX_STEPS}step"
mkdir -p "${DEST}/server_runtime_config"
cp -a "${LOG}" "${DEST}/"
cp -a "${LOGDIR}" "${DEST}/"
cp -a "${CONFIG_PATH}" "${DEST}/server_runtime_config/"
echo "Saved LIBERO Goal Dopamine-RL result to ${DEST}"
