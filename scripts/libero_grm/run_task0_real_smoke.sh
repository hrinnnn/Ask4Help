#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
export_rll_env
require_path "${PI05_MODEL_PATH}" "pi0.5 model"
require_path "${GRM_GOAL_BANK_DIR}/task_000/meta.json" "GRM goal bank"

CONFIG_NAME="libero_spatial_ppo_openpi_pi05_grm_task0_real_smoke_runtime"
CONFIG_PATH="${RLINF_DIR}/examples/embodiment/config/${CONFIG_NAME}.yaml"
python "${SCRIPT_DIR}/write_runtime_config.py" \
  --output "${CONFIG_PATH}" \
  --experiment-name "libero_spatial_ppo_openpi_pi05_grm_task0_real_smoke" \
  --pi05-model-path "${PI05_MODEL_PATH}" \
  --goal-bank-dir "${GRM_GOAL_BANK_DIR}" \
  --grm-model-name "${GRM_MODEL_NAME}" \
  --grm-endpoint "${GRM_ENDPOINT}" \
  --train-gpu-rank "${TRAIN_GPU_RANK}" \
  --num-envs 1 \
  --max-epochs 1 \
  --max-steps 1 \
  --max-episode-steps 20 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --task-id-filter 0

RUN_ID="$(timestamp)"
LOG="${TASK0_SMOKE_LOG:-/root/rlinf_pi05_libero_grm_task0_real_smoke_${RUN_ID}.log}"
LOGDIR="${TASK0_SMOKE_LOGDIR:-${RLINF_DIR}/logs/grm_task0_real_smoke_${RUN_ID}}"

cd "${RLINF_DIR}"
python examples/embodiment/train_embodied_agent.py \
  --config-path "${RLINF_DIR}/examples/embodiment/config" \
  --config-name "${CONFIG_NAME}" \
  runner.logger.log_path="${LOGDIR}" 2>&1 | tee "${LOG}"

DEST="${RESULTS_DIR}/$(date +%Y%m%d_%H%M)_grm_task0_real_smoke"
mkdir -p "${DEST}/server_runtime_config"
cp -a "${LOG}" "${DEST}/"
cp -a "${LOGDIR}" "${DEST}/"
cp -a "${CONFIG_PATH}" "${DEST}/server_runtime_config/"
echo "Saved smoke result to ${DEST}"

