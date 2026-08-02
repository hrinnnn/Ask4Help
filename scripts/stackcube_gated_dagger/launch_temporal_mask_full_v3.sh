#!/usr/bin/env bash
# Start the clean two-GPU StackCube BC rerun from the immutable step-7000 weights.
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/root/Ask4Help-online-awbc-code}
RUNNER=${CODE_ROOT}/scripts/stackcube_gated_dagger/run_group_bc.sh
PYTHON=${PYTHON:-/root/Ask4Help-online-awbc/RLinf/.venv/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/mnt/data/ask4help/results/stackcube_gated_dagger/full_v3/temporal_mask_original_id_norm_5k}
BASE_CHECKPOINT=/mnt/data/ask4help/results/stackcube_id_sft/pi05_id_sft_10000_visual_v1_20260722_dual/member_0/maniskill_stackcube_pi05_id_sft/checkpoints/global_step_7000
ID_REPLAY=/mnt/data/ask4help/datasets/lerobot/local/stackcube_id_128_visual_v1_20260722
ORIGINAL_NORM=/mnt/data/ask4help/datasets/lerobot/local/stackcube_id_128_visual_v1_20260722/norm_stats_id.json
KNN_DATASET=/mnt/data/ask4help/results/stackcube_gated_dagger/full_v2/bridge_knn_successful_experts_100_full_suffix_retry1/dataset
OFFLINE_DATASET=/mnt/data/ask4help/results/stackcube_gated_dagger/full_v2/two_gpu_lockstep_v1/offline_oracle_100_full_rebuilt/dataset

if [[ -e "${RESULT_ROOT}" ]]; then
  echo "refusing to reuse existing result directory: ${RESULT_ROOT}" >&2
  exit 2
fi
for required in "${RUNNER}" "${BASE_CHECKPOINT}/actor/model_state_dict/full_weights.pt" \
  "${ORIGINAL_NORM}" "${KNN_DATASET}/meta/info.json" "${OFFLINE_DATASET}/meta/info.json"; do
  test -e "${required}"
done

mkdir -p "${RESULT_ROOT}/logs"
cp "${BASH_SOURCE[0]}" "${RESULT_ROOT}/launch_command.sh"
sha256sum "${ORIGINAL_NORM}" > "${RESULT_ROOT}/original_id_norm_stats.sha256"
"${PYTHON}" - <<PY
import hashlib
import json
from pathlib import Path

root = Path("${RESULT_ROOT}")
manifest = {
    "base_checkpoint": "${BASE_CHECKPOINT}",
    "id_replay": "${ID_REPLAY}",
    "knn_dataset": "${KNN_DATASET}",
    "offline_dataset": "${OFFLINE_DATASET}",
    "norm_stats": "${ORIGINAL_NORM}",
    "norm_sha256": hashlib.sha256(Path("${ORIGINAL_NORM}").read_bytes()).hexdigest(),
    "padded_action_mode": "temporal_mask",
    "action_horizon": 10,
    "global_batch_size": 128,
    "micro_batch_size": 32,
    "max_steps": 5000,
    "save_interval": 500,
    "awbc_enabled": False,
    "source_mix": "1:1 original_id:new_expert",
    "code_commit": "$(git -C "${CODE_ROOT}" rev-parse HEAD)",
    "rlinf_commit": "$(git -C "${CODE_ROOT}/RLinf" rev-parse HEAD)",
}
(root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

ray stop --force || true

ASK4HELP_ROOT="${CODE_ROOT}" RLINF_ROOT="${CODE_ROOT}/RLinf" PYTHON="${PYTHON}" \
GPU_ID=0 SEED=4000 ID_REPLAY="${ID_REPLAY}" NEW_EXPERT_DATASET="${KNN_DATASET}" \
BASE_CHECKPOINT="${BASE_CHECKPOINT}" NORM_STATS="${ORIGINAL_NORM}" \
OUTPUT_DIR="${RESULT_ROOT}/bridge_knn" MAX_STEPS=5000 SAVE_INTERVAL=500 \
GLOBAL_BATCH_SIZE=128 MICRO_BATCH_SIZE=32 PADDED_ACTION_MODE=mask \
nohup bash "${RUNNER}" > "${RESULT_ROOT}/logs/bridge_knn.log" 2>&1 &
echo $! > "${RESULT_ROOT}/bridge_knn.pid"

sleep 15
ASK4HELP_ROOT="${CODE_ROOT}" RLINF_ROOT="${CODE_ROOT}/RLinf" PYTHON="${PYTHON}" \
RAY_ADDRESS=auto GPU_ID=1 SEED=4001 ID_REPLAY="${ID_REPLAY}" NEW_EXPERT_DATASET="${OFFLINE_DATASET}" \
BASE_CHECKPOINT="${BASE_CHECKPOINT}" NORM_STATS="${ORIGINAL_NORM}" \
OUTPUT_DIR="${RESULT_ROOT}/offline_oracle" MAX_STEPS=5000 SAVE_INTERVAL=500 \
GLOBAL_BATCH_SIZE=128 MICRO_BATCH_SIZE=32 PADDED_ACTION_MODE=mask \
nohup bash "${RUNNER}" > "${RESULT_ROOT}/logs/offline_oracle.log" 2>&1 &
echo $! > "${RESULT_ROOT}/offline_oracle.pid"

echo "launched $(cat "${RESULT_ROOT}/bridge_knn.pid") and $(cat "${RESULT_ROOT}/offline_oracle.pid")"
