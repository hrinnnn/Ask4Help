#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
export_rll_env
require_path "${PI05_MODEL_PATH}" "pi0.5 model"

# Keep these defaults aligned with run_short_train.sh. Override them together for a fair pair.
SHORT_NUM_ENVS="${SHORT_NUM_ENVS:-8}"
SHORT_ROLLOUT_EPOCH="${SHORT_ROLLOUT_EPOCH:-4}"
SHORT_MAX_STEPS="${SHORT_MAX_STEPS:-10}"
SHORT_MAX_EPOCHS="${SHORT_MAX_EPOCHS:-10}"
SHORT_MAX_EPISODE_STEPS="${SHORT_MAX_EPISODE_STEPS:-120}"
SHORT_MICRO_BATCH="${SHORT_MICRO_BATCH:-8}"
SHORT_GLOBAL_BATCH="${SHORT_GLOBAL_BATCH:-32}"

CONFIG_NAME="libero_spatial_ppo_openpi_pi05_sparse_short_train_runtime"
CONFIG_PATH="${RLINF_DIR}/examples/embodiment/config/${CONFIG_NAME}.yaml"
RUN_ID="$(timestamp)"
LOG="${SPARSE_TRAIN_LOG:-/root/rlinf_pi05_libero_sparse_short_train_${RUN_ID}.log}"
LOGDIR="${SPARSE_TRAIN_LOGDIR:-${RLINF_DIR}/logs/sparse_short_train_${RUN_ID}}"
python "${SCRIPT_DIR}/write_runtime_config.py" \
  --output "${CONFIG_PATH}" \
  --experiment-name "libero_spatial_ppo_openpi_pi05_sparse_short_train" \
  --pi05-model-path "${PI05_MODEL_PATH}" \
  --goal-bank-dir "${GRM_GOAL_BANK_DIR}" \
  --grm-model-name "${GRM_MODEL_NAME}" \
  --grm-endpoint "${GRM_ENDPOINT}" \
  --train-gpu-rank "${TRAIN_GPU_RANK}" \
  --num-envs "${SHORT_NUM_ENVS}" \
  --rollout-epoch "${SHORT_ROLLOUT_EPOCH}" \
  --max-epochs "${SHORT_MAX_EPOCHS}" \
  --max-steps "${SHORT_MAX_STEPS}" \
  --max-episode-steps "${SHORT_MAX_EPISODE_STEPS}" \
  --micro-batch-size "${SHORT_MICRO_BATCH}" \
  --global-batch-size "${SHORT_GLOBAL_BATCH}" \
  --disable-reward-model
cd "${RLINF_DIR}"
python examples/embodiment/train_embodied_agent.py \
  --config-path "${RLINF_DIR}/examples/embodiment/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOGDIR}" 2>&1 | tee "${LOG}"

DEST="${RESULTS_DIR}/$(date +%Y%m%d_%H%M)_sparse_short_train_${SHORT_NUM_ENVS}env_${SHORT_MAX_STEPS}step"
mkdir -p "${DEST}/server_runtime_config"
cp -a "${LOG}" "${DEST}/"
cp -a "${LOGDIR}" "${DEST}/"
cp -a "${CONFIG_PATH}" "${DEST}/server_runtime_config/"
echo "Saved sparse baseline result to ${DEST}"
