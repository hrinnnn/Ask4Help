#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

activate_rlinf
mkdir -p "${LIBERO_DEMO_ROOT}"

count="$(find "${LIBERO_SPATIAL_DEMO_DIR}" -maxdepth 1 -name '*.hdf5' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${count}" == "10" ]]; then
  echo "LIBERO spatial demos already complete at ${LIBERO_SPATIAL_DEMO_DIR}."
  exit 0
fi

LOG="${LIBERO_DOWNLOAD_LOG:-/root/libero_spatial_download_$(timestamp).log}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" nohup python -u \
  "${RLINF_DIR}/.venv/libero/benchmark_scripts/download_libero_datasets.py" \
  --download-dir "${LIBERO_DEMO_ROOT}" \
  --datasets libero_spatial \
  --use-huggingface > "${LOG}" 2>&1 &

echo $! > /tmp/ask4help_libero_spatial_download.pid
echo "${LOG}" > /tmp/ask4help_libero_spatial_download.logpath
echo "Started LIBERO spatial download pid=$(cat /tmp/ask4help_libero_spatial_download.pid) log=${LOG}"

