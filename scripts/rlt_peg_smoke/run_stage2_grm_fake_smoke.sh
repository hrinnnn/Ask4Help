#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
FAKE_GRM_HOST=${FAKE_GRM_HOST:-127.0.0.1}
FAKE_GRM_PORT=${FAKE_GRM_PORT:-18000}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the Stage 2 output directory}
FAKE_GRM_LOG=${FAKE_GRM_LOG:-"${OUTPUT_DIR}/fake_grm_server.log"}

mkdir -p "${OUTPUT_DIR}"
"${PYTHON}" "${RLINF_ROOT}/toolkits/fake_dopamine_grm_server.py" \
  --host "${FAKE_GRM_HOST}" \
  --port "${FAKE_GRM_PORT}" >"${FAKE_GRM_LOG}" 2>&1 &
FAKE_GRM_PID=$!
trap 'kill "${FAKE_GRM_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 50); do
  if ! kill -0 "${FAKE_GRM_PID}" 2>/dev/null; then
    echo "Fake GRM server exited before becoming ready" >&2
    exit 1
  fi
  if "${PYTHON}" - "${FAKE_GRM_HOST}" "${FAKE_GRM_PORT}" <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1:]
body = json.dumps(
    {"model": "fake-grm", "messages": [{"role": "user", "content": "ping"}]}
).encode("utf-8")
request = urllib.request.Request(
    f"http://{host}:{port}/v1/chat/completions",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=0.5) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  sleep 0.2
done

GRM_ENDPOINT="http://${FAKE_GRM_HOST}:${FAKE_GRM_PORT}/v1/chat/completions" \
GRM_MODEL_NAME="fake-grm" \
  "$(dirname "$0")/run_stage2_grm_smoke.sh"
