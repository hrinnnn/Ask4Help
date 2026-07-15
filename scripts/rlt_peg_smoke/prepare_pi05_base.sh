#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
DOWNLOAD_ROOT=${DOWNLOAD_ROOT:-/root/ask4help_model_downloads/openpi}
OUTPUT_DIR=${OUTPUT_DIR:-/root/ask4help_model_downloads/pi05_base_torch}
PERSIST_DIR=${PERSIST_DIR:-/mnt/data/ask4help/models/pi05_base_torch}
OPENPI_CONVERT_CONFIG=${OPENPI_CONVERT_CONFIG:-pi05_droid}

[[ -x "${PYTHON}" ]] || {
  echo "RLinf Python does not exist: ${PYTHON}" >&2
  exit 1
}
[[ -f "${RLINF_ROOT}/rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py" ]] || {
  echo "RLinf conversion utility does not exist under ${RLINF_ROOT}" >&2
  exit 1
}

export OPENPI_DATA_HOME="${DOWNLOAD_ROOT}"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base"))'

JAX_CHECKPOINT="${DOWNLOAD_ROOT}/openpi-assets/checkpoints/pi05_base"
[[ -f "${JAX_CHECKPOINT}/params/commit_success.txt" ]] || {
  echo "Incomplete pi05_base download: ${JAX_CHECKPOINT}" >&2
  exit 1
}

if [[ ! -f "${OUTPUT_DIR}/model.safetensors" ]]; then
  "${PYTHON}" "${RLINF_ROOT}/rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py" \
    --checkpoint-dir "${JAX_CHECKPOINT}" \
    --config-name "${OPENPI_CONVERT_CONFIG}" \
    --output-path "${OUTPUT_DIR}" \
    --precision bfloat16
fi

if [[ -n "${PERSIST_DIR}" ]]; then
  mkdir -p "${PERSIST_DIR}"
  rsync -a --delete "${OUTPUT_DIR}/" "${PERSIST_DIR}/"
  echo "Persisted converted pi05_base to ${PERSIST_DIR}"
fi

echo "Prepared pi05_base PyTorch checkpoint at ${OUTPUT_DIR}"
