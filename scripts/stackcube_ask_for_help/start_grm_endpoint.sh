#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

GRM_PYTHON=${GRM_PYTHON:-/opt/conda/envs/robo-dopamine/bin/python}
test -x "${GRM_PYTHON}"
test -d "${GRM_MODEL}"
if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "GRM endpoint already healthy"
  exit 0
fi
LOG="${RESULT_ROOT}/grm_endpoint.log"
CUDA_VISIBLE_DEVICES=${GRM_GPU:-1} nohup "${GRM_PYTHON}" \
  -m vllm.entrypoints.openai.api_server \
  --model "${GRM_MODEL}" --served-model-name "${GRM_MODEL_NAME}" \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --limit-mm-per-prompt '{"image":8}' --max-model-len 8192 \
  --gpu-memory-utilization "${GRM_GPU_MEMORY_UTILIZATION:-0.45}" \
  >"${LOG}" 2>&1 &
echo $! >"${RESULT_ROOT}/grm_endpoint.pid"
