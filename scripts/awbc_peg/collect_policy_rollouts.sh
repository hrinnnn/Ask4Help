#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
PI05_MODEL_PATH=${PI05_MODEL_PATH:?Set PI05_MODEL_PATH to the shared warm-start}
NORM_STATS_PATH=${NORM_STATS_PATH:?Set NORM_STATS_PATH}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
MERGED_DATASET=${MERGED_DATASET:-"${OUTPUT_DIR}/policy_rollouts"}

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export PI05_MODEL_PATH
export NORM_STATS_PATH

"${PYTHON}" examples/embodiment/train_embodied_agent.py \
  --config-path "${RLINF_ROOT}/examples/embodiment/config" \
  --config-name maniskill_awbc_collect_openpi_pi05 \
  runner.logger.log_path="${OUTPUT_DIR}"

rm -rf "${MERGED_DATASET}"
"${PYTHON}" toolkits/lerobot/merge_lerobot_datasets.py \
  --source-dir "${OUTPUT_DIR}/policy_rollouts_raw" \
  --output-dir "${MERGED_DATASET}"

"${PYTHON}" - "${MERGED_DATASET}" "${OUTPUT_DIR}/successful_episodes.json" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

dataset = Path(sys.argv[1])
output = Path(sys.argv[2])
successful = []
for parquet_path in sorted(dataset.glob("data/chunk-*/episode_*.parquet")):
    table = pq.read_table(parquet_path, columns=["is_success"])
    values = table.column("is_success").to_pylist()
    if any(bool(value[0] if isinstance(value, list) else value) for value in values):
        successful.append(int(parquet_path.stem.split("_")[-1]))
output.write_text(json.dumps({"successful_episodes": successful}, indent=2) + "\n")
print(output)
PY
