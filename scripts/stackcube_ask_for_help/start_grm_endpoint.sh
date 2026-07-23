#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
export RESULT_ROOT GRM_MODEL GRM_MODEL_NAME GRM_ADAPTER_NAME GRM_ADAPTER_DIR

GRM_PYTHON=${GRM_PYTHON:-/opt/conda/envs/robo-dopamine/bin/python}
test -x "${GRM_PYTHON}"
test -d "${GRM_MODEL}"
test -f "${GRM_ADAPTER_DIR}/adapter_config.json"
if curl -fsS http://127.0.0.1:8000/v1/models >"${RESULT_ROOT}/grm_models.json" 2>/dev/null; then
  "${PYTHON}" - <<'PY'
import json, os
payload = json.load(open(os.environ["RESULT_ROOT"] + "/grm_models.json"))
ids = {item.get("id") for item in payload.get("data", [])}
expected = {os.environ["GRM_MODEL_NAME"], os.environ["GRM_ADAPTER_NAME"]}
if not expected <= ids:
    raise SystemExit(f"healthy endpoint lacks adapted adapter: ids={sorted(ids)} expected={sorted(expected)}")
PY
  echo "Adapted GRM endpoint already healthy"
  exit 0
fi
LOG=${GRM_LOG:-/root/stackcube_smoke_grm_endpoint.log}
CUDA_VISIBLE_DEVICES=${GRM_GPU:-1} nohup "${GRM_PYTHON}" \
  -m vllm.entrypoints.openai.api_server \
  --model "${GRM_MODEL}" --served-model-name "${GRM_MODEL}" \
  --enable-lora --max-loras 1 --lora-modules "${GRM_ADAPTER_NAME}=${GRM_ADAPTER_DIR}" \
  --default-mm-loras "${GRM_ADAPTER_NAME}" \
  --host 127.0.0.1 --port 8000 --trust-remote-code \
  --limit-mm-per-prompt '{"image":8}' --max-model-len 8192 \
  --gpu-memory-utilization "${GRM_GPU_MEMORY_UTILIZATION:-0.45}" \
  >"${LOG}" 2>&1 &
echo $! >"${RESULT_ROOT}/grm_endpoint.pid"
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/v1/models >"${RESULT_ROOT}/grm_models.json" 2>/dev/null; then
    break
  fi
  sleep 5
done
"${PYTHON}" - <<'PY'
import hashlib, json, os
root = os.environ["RESULT_ROOT"]
adapter = os.environ["GRM_ADAPTER_DIR"]
payload = json.load(open(root + "/grm_models.json"))
ids = {item.get("id") for item in payload.get("data", [])}
expected = {os.environ["GRM_MODEL"], os.environ["GRM_ADAPTER_NAME"]}
if not expected <= ids:
    raise SystemExit(f"endpoint did not expose both base and adapter: ids={sorted(ids)}")
weight = next((os.path.join(adapter, name) for name in ("adapter_model.safetensors", "adapter_model.bin") if os.path.isfile(os.path.join(adapter, name))), None)
if weight is None:
    raise SystemExit("LoRA adapter weight is missing")
with open(weight, "rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
json.dump({"base_model": os.environ["GRM_MODEL"], "adapter_name": os.environ["GRM_ADAPTER_NAME"], "adapter_path": adapter, "adapter_sha256": digest, "models": sorted(ids)}, open(root + "/grm_endpoint_manifest.json", "w"), indent=2)
PY
