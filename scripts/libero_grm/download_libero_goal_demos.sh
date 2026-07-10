#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
mkdir -p "${LIBERO_DEMO_ROOT}"
mkdir -p "${LIBERO_GOAL_DEMO_DIR}"

count="$(find "${LIBERO_GOAL_DEMO_DIR}" -maxdepth 1 -name '*.hdf5' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${count}" == "10" ]]; then
  echo "LIBERO Goal demos already complete at ${LIBERO_GOAL_DEMO_DIR}."
  exit 0
fi

LOG="${LIBERO_DOWNLOAD_LOG:-/root/libero_goal_download_$(timestamp).log}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" nohup python -u \
  "${RLINF_DIR}/.venv/libero/benchmark_scripts/download_libero_datasets.py" \
  --download-dir "${LIBERO_DEMO_ROOT}" \
  --datasets libero_goal \
  --use-huggingface > "${LOG}" 2>&1 &

echo $! > /tmp/ask4help_libero_goal_download.pid
echo "${LOG}" > /tmp/ask4help_libero_goal_download.logpath
echo "Started LIBERO Goal download pid=$(cat /tmp/ask4help_libero_goal_download.pid) log=${LOG}"
