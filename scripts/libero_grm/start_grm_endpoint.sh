#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_path "${GRM_MODEL_PATH}" "Robo-Dopamine GRM model"
require_path "${ROBO_DOPAMINE_CONDA}" "conda activation script"

if curl -fsS "http://127.0.0.1:${GRM_PORT}/v1/models" >/dev/null 2>&1; then
  echo "GRM endpoint already responds on port ${GRM_PORT}."
  exit 0
fi

LOG="${GRM_LOG:-/root/grm_vllm_endpoint_$(timestamp)_gpu${GRM_GPU}.log}"
# shellcheck disable=SC1090
source "${ROBO_DOPAMINE_CONDA}"
conda activate "${ROBO_DOPAMINE_ENV}"

CUDA_VISIBLE_DEVICES="${GRM_GPU}" nohup python -m vllm.entrypoints.openai.api_server \
  --model "${GRM_MODEL_PATH}" \
  --served-model-name "${GRM_MODEL_NAME}" \
  --host "${GRM_HOST}" \
  --port "${GRM_PORT}" \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image":8}' \
  --max-model-len 8192 \
  --gpu-memory-utilization "${GRM_GPU_MEMORY_UTILIZATION:-0.45}" \
  > "${LOG}" 2>&1 &

echo $! > /tmp/ask4help_grm_endpoint.pid
echo "${LOG}" > /tmp/ask4help_grm_endpoint.logpath
echo "Started GRM endpoint pid=$(cat /tmp/ask4help_grm_endpoint.pid) log=${LOG}"

