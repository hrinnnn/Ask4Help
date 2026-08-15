#!/usr/bin/env bash
set -euo pipefail

V2_ROOT=${1:?v2 diagnostic root is required}
V3_ROOT=${2:?v3 root is required}
PYTHON=${3:?python path is required}
REPO_ROOT=${4:?repo root is required}
XVLA_ROOT=${5:?X-VLA root is required}
MODEL=${6:?base checkpoint is required}
ID_H5=${7:?ID H5 is required}
MANIFEST=${8:?v3 seed manifest is required}

if [[ -e "${V3_ROOT}" ]]; then
    printf '%s\n' "refusing to reuse v3 root: ${V3_ROOT}" >&2
    exit 2
fi
mkdir -p "${V3_ROOT}"
printf '%s\n' 'phase=waiting_for_v2_diagnostic' > "${V3_ROOT}/pipeline_state.txt"

while true; do
    if [[ -f "${V2_ROOT}/protocol_gates.log" ]]; then
        break
    fi
    if [[ -f "${V2_ROOT}/V2_RECOVERY_COMPLETE" ]]; then
        printf '%s\n' 'v2 unexpectedly completed; v3 recovery is unnecessary' > "${V3_ROOT}/V3_NOT_NEEDED"
        printf '%s\n' 'phase=not_needed' > "${V3_ROOT}/pipeline_state.txt"
        exit 0
    fi
    sleep 300
done

export STACKPYRAMID_OOD_GEOMETRY=v3
printf '%s\n' 'phase=v3_protocol_gates' > "${V3_ROOT}/pipeline_state.txt"
"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_timing_protocol_gates.py" \
    --output-root "${V3_ROOT}/protocol_gates" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --checkpoint "${MODEL}" \
    --python "${PYTHON}" \
    --cpu-sets 0-7 8-15 \
    --seed-manifest "${MANIFEST}" \
    > "${V3_ROOT}/protocol_gates.log" 2>&1

printf '%s\n' 'phase=v3_locality_gate' > "${V3_ROOT}/pipeline_state.txt"
"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_stage_locality_gate.py" \
    --output-root "${V3_ROOT}/protocol_gates/prefix_gate" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --checkpoint "${MODEL}" \
    --python "${PYTHON}" \
    --seed-manifest "${V3_ROOT}/protocol_gates/seed_manifest.json" \
    --geometry v3 \
    --cpu-sets 0-7 8-15 \
    > "${V3_ROOT}/prefix_gate.log" 2>&1

printf '%s\n' 'phase=v3_formal_timing' > "${V3_ROOT}/pipeline_state.txt"
"${REPO_ROOT}/tools/run_stackpyramid_formal_timing_after_gates.sh" \
    "${V3_ROOT}/protocol_gates" \
    "${V3_ROOT}/formal_timing" \
    "${PYTHON}" \
    "${REPO_ROOT}" \
    "${XVLA_ROOT}" \
    "${MODEL}" \
    "${ID_H5}" \
    "${V3_ROOT}/protocol_gates/seed_manifest.json" \
    > "${V3_ROOT}/formal_timing.log" 2>&1

printf '%s\n' 'complete' > "${V3_ROOT}/V3_RECOVERY_COMPLETE"
printf '%s\n' 'phase=complete' > "${V3_ROOT}/pipeline_state.txt"
