#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?protocol-gate root is required}
PYTHON=${2:?python path is required}
REPO_ROOT=${3:?repo root is required}
XVLA_ROOT=${4:?X-VLA root is required}
CHECKPOINT=${5:?checkpoint is required}
SUPERVISOR=${ROOT}/prefix_gate_supervisor
mkdir -p "${SUPERVISOR}"

while true; do
    if [[ -f "${ROOT}/PROTOCOL_GATES_COMPLETE" ]]; then
        break
    fi
    if [[ -f "${ROOT}/PROTOCOL_GATES_FAILED" ]]; then
        printf '%s\n' 'protocol gates failed' > "${SUPERVISOR}/FAILED"
        exit 2
    fi
    sleep 300
done

OUTPUT=${ROOT}/prefix_gate
if [[ -e "${OUTPUT}" ]]; then
    printf '%s\n' 'prefix gate output already exists' > "${SUPERVISOR}/FAILED"
    exit 3
fi

"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_stage_locality_gate.py" \
    --output-root "${OUTPUT}" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --python "${PYTHON}" \
    --seed-manifest "${ROOT}/seed_manifest.json" \
    --cpu-sets 0-7 8-15 \
    > "${SUPERVISOR}/run.log" 2>&1

printf '%s\n' 'complete' > "${SUPERVISOR}/COMPLETE"
