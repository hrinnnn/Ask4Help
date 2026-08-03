#!/usr/bin/env bash
# Launch the two-worker full LIBERO-10 feature bank on dedicated idle GPUs.
set -euo pipefail

REPO_ROOT="${ASK4HELP_ROOT:-/data/zhaozhixuan/Ask4Help}"
DATA_ROOT="${LIBERO_FAILURE_ROOT:-/data/zhaozhixuan/libero_plus_failure}"
DATASET_ROOT="${LIBERO_DATASET_ROOT:-${DATA_ROOT}/datasets/physical-intelligence_libero}"
CHECKPOINT="${PI05_LIBERO_CHECKPOINT:-/data/zhaozhixuan/libero_pi05_eval/models/openpi/openpi-assets/checkpoints/pi05_libero}"
PYTHON="${OPENPI_SERVER_PYTHON:-/data/zhaozhixuan/libero_pi05_eval/envs/openpi_server/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/results/libero10_all_observation_reference_v1}"
GPU_IDS="${GPU_IDS:-2,3}"
PORTS="${PORTS:-8022,8023}"

IFS=',' read -r GPU_A GPU_B <<<"${GPU_IDS}"
IFS=',' read -r PORT_A PORT_B <<<"${PORTS}"
if [[ -z "${GPU_A}" || -z "${GPU_B}" || -z "${PORT_A}" || -z "${PORT_B}" || "${GPU_A}" == "${GPU_B}" ]]; then
  echo "GPU_IDS and PORTS must provide two distinct comma-separated values" >&2
  exit 2
fi

export TMPDIR="${DATA_ROOT}/tmp/full_reference_bank"
export XDG_CACHE_HOME="${DATA_ROOT}/cache"
export HF_HOME="${DATA_ROOT}/cache/huggingface"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${OUTPUT_ROOT}/logs"

# A listed compute PID means the card belongs to another workload. Driver-only
# memory does not appear here and is safe.
for gpu in "${GPU_A}" "${GPU_B}"; do
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F, -v wanted="${gpu}" '$1 ~ "^" wanted " *$" {gsub(/ /, "", $2); print $2}')"
  if [[ -z "${uuid}" ]]; then
    echo "GPU ${gpu} does not exist" >&2
    exit 3
  fi
  if nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -Fq "${uuid}"; then
    echo "GPU ${gpu} has an existing compute process; refusing to interfere" >&2
    exit 3
  fi
done

if [[ -e "${OUTPUT_ROOT}/validation.json" ]]; then
  echo "full reference bank is already sealed: ${OUTPUT_ROOT}" >&2
  exit 4
fi

cd "${REPO_ROOT}"
launch_policy() {
  local worker="$1" gpu="$2" port="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON}" tools/libero_plus_failure/serve_pi05_internal_features.py \
    --checkpoint "${CHECKPOINT}" --port "${port}" \
    >"${OUTPUT_ROOT}/logs/policy_worker_${worker}.log" 2>&1 &
  echo $! >"${OUTPUT_ROOT}/policy_worker_${worker}.pid"
}
launch_policy 0 "${GPU_A}" "${PORT_A}"
launch_policy 1 "${GPU_B}" "${PORT_B}"
for worker in 0 1; do
  log="${OUTPUT_ROOT}/logs/policy_worker_${worker}.log"
  ready=0
  for _ in $(seq 1 60); do
    if grep -q "Serving pi05 internal probes" "${log}"; then
      ready=1
      break
    fi
    pid="$(cat "${OUTPUT_ROOT}/policy_worker_${worker}.pid")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 5
  done
  if [[ "${ready}" != 1 ]]; then
    echo "policy worker ${worker} failed to start; inspect ${log}" >&2
    exit 5
  fi
done

launch_extractor() {
  local worker="$1" port="$2"
  nohup "${PYTHON}" tools/libero_plus_failure/build_all_observation_feature_bank.py \
    --dataset-root "${DATASET_ROOT}" --output-root "${OUTPUT_ROOT}" \
    --worker-index "${worker}" --worker-count 2 --port "${port}" --checkpoint "${CHECKPOINT}" \
    >"${OUTPUT_ROOT}/logs/extractor_worker_${worker}.log" 2>&1 &
  echo $! >"${OUTPUT_ROOT}/extractor_worker_${worker}.pid"
}
launch_extractor 0 "${PORT_A}"
launch_extractor 1 "${PORT_B}"
echo "Started two full-bank extractors. PIDs and logs are under ${OUTPUT_ROOT}."
