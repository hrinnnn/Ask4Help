#!/usr/bin/env bash
set -euo pipefail

V1_ROOT=${1:?v1 protocol-gate root is required}
V2_ROOT=${2:?v2 root is required}
PYTHON=${3:?python path is required}
REPO_ROOT=${4:?repo root is required}
XVLA_ROOT=${5:?X-VLA root is required}
MODEL=${6:?base checkpoint is required}
ID_H5=${7:?ID H5 is required}
MANIFEST=${8:?v2 seed manifest is required}

if [[ -e "${V2_ROOT}" ]]; then
    printf '%s\n' "refusing to reuse v2 root: ${V2_ROOT}" >&2
    exit 2
fi
mkdir -p "${V2_ROOT}"
printf '%s\n' 'phase=waiting_for_v1_locality_diagnostic' > "${V2_ROOT}/pipeline_state.txt"

while true; do
    if [[ -f "${V1_ROOT}/prefix_gate/STAGE_LOCALITY_GATE_DIAGNOSTIC" ]]; then
        break
    fi
    if [[ -f "${V1_ROOT}/prefix_gate/STAGE_LOCALITY_GATE_COMPLETE" ]]; then
        printf '%s\n' 'v1 locality gate passed; v2 recovery is unnecessary' > "${V2_ROOT}/V2_NOT_NEEDED"
        printf '%s\n' 'phase=not_needed' > "${V2_ROOT}/pipeline_state.txt"
        exit 0
    fi
    sleep 300
done

export STACKPYRAMID_OOD_GEOMETRY=v2
printf '%s\n' 'phase=v2_protocol_gates' > "${V2_ROOT}/pipeline_state.txt"
"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_timing_protocol_gates.py" \
    --output-root "${V2_ROOT}/protocol_gates" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --checkpoint "${MODEL}" \
    --python "${PYTHON}" \
    --cpu-sets 0-7 8-15 \
    --seed-manifest "${MANIFEST}" \
    > "${V2_ROOT}/protocol_gates.log" 2>&1

printf '%s\n' 'phase=v2_locality_gate' > "${V2_ROOT}/pipeline_state.txt"
"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_stage_locality_gate.py" \
    --output-root "${V2_ROOT}/protocol_gates/prefix_gate" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --checkpoint "${MODEL}" \
    --python "${PYTHON}" \
    --seed-manifest "${V2_ROOT}/protocol_gates/seed_manifest.json" \
    --cpu-sets 0-7 8-15 \
    > "${V2_ROOT}/prefix_gate.log" 2>&1

printf '%s\n' 'phase=v2_formal_timing' > "${V2_ROOT}/pipeline_state.txt"
"${REPO_ROOT}/tools/run_stackpyramid_formal_timing_after_gates.sh" \
    "${V2_ROOT}/protocol_gates" \
    "${V2_ROOT}/formal_timing" \
    "${PYTHON}" \
    "${REPO_ROOT}" \
    "${XVLA_ROOT}" \
    "${MODEL}" \
    "${ID_H5}" \
    > "${V2_ROOT}/formal_timing.log" 2>&1

printf '%s\n' 'complete' > "${V2_ROOT}/V2_RECOVERY_COMPLETE"
printf '%s\n' 'phase=complete' > "${V2_ROOT}/pipeline_state.txt"
