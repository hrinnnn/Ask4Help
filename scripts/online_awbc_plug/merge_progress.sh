#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
OUTPUT=${OUTPUT:?Set OUTPUT to combined progress.jsonl}

# Each INPUT item is NAME:MANIFEST.  Keep this order identical to
# data.train_data_paths so ConcatDataset indices and sidecar indices agree.
: "${INPUTS:?Set INPUTS as whitespace-separated NAME:MANIFEST entries}"
args=()
for item in ${INPUTS}; do
  name=${item%%:*}
  path=${item#*:}
  args+=(--input "${name}" "${path}")
done
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" "${RLINF_ROOT}/toolkits/lerobot/merge_awbc_progress.py" "${args[@]}" --output "${OUTPUT}"
