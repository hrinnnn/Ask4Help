#!/usr/bin/env bash
set -euo pipefail

GATE_ROOT=${1:?protocol-gate root is required}
ROOT=${2:?formal output root is required}
PYTHON=${3:?python path is required}
REPO_ROOT=${4:?repo root is required}
XVLA_ROOT=${5:?X-VLA root is required}
MODEL=${6:?base checkpoint is required}
ID_H5=${7:?ID H5 is required}

STATE=${ROOT}/formal_pipeline_state.json
write_state() {
    printf '%s\n' "$1" > "${STATE}"
}

if [[ -e "${ROOT}" ]]; then
    printf '%s\n' "refusing to reuse formal output root: ${ROOT}" >&2
    exit 2
fi
mkdir -p "${ROOT}"
write_state 'phase=waiting_for_protocol_gates'

while true; do
    if [[ -f "${GATE_ROOT}/prefix_gate/STAGE_LOCALITY_GATE_COMPLETE" ]]; then
        break
    fi
    if [[ -f "${GATE_ROOT}/prefix_gate/STAGE_LOCALITY_GATE_DIAGNOSTIC" ]]; then
        printf '%s\n' 'stage locality gate failed; formal timing was not started' > "${ROOT}/FORMAL_TIMING_DIAGNOSTIC"
        write_state 'phase=diagnostic_locality_gate_failed'
        exit 3
    fi
    if [[ -f "${GATE_ROOT}/PROTOCOL_GATES_FAILED" ]]; then
        printf '%s\n' 'protocol gate failed; formal timing was not started' > "${ROOT}/FORMAL_TIMING_DIAGNOSTIC"
        write_state 'phase=diagnostic_protocol_gate_failed'
        exit 4
    fi
    sleep 300
done

write_state 'phase=boundary_audit'
"${PYTHON}" "${REPO_ROOT}/tools/audit_stackpyramid_stage_boundaries.py" \
    --xvla-root "${XVLA_ROOT}" \
    --task-root "${REPO_ROOT}" \
    --output "${ROOT}/audit" \
    --episodes 100 \
    --seed-manifest "${GATE_ROOT}/seed_manifest.json" \
    --sim-backend cpu \
    --render-backend cpu
test -f "${ROOT}/audit/AUDIT_COMPLETE"

run_collection() {
    local stage=$1 condition=$2 gpu=$3 cpu_set=$4 start_seed=$5
    local output="${ROOT}/collection/${stage}/${condition}"
    mkdir -p "${ROOT}/collection/${stage}"
    CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=8 taskset -c "${cpu_set}" \
        "${PYTHON}" "${REPO_ROOT}/tools/collect_stackpyramid_timing_sweep.py" \
        --checkpoint "${MODEL}" \
        --xvla-root "${XVLA_ROOT}" \
        --task-root "${REPO_ROOT}" \
        --audit "${ROOT}/audit" \
        --output "${output}" \
        --split "${stage}" \
        --condition "${condition}" \
        --target 20 \
        --start-seed "${start_seed}" \
        --max-attempts 100 \
        --pre-offset 25 \
        --boundary-offset 5 \
        --recovery-delay 25 \
        --flow-steps 5 \
        --sim-backend cpu \
        --render-backend cpu \
        > "${ROOT}/collection/${stage}/${condition}.log" 2>&1
    test -f "${output}/COLLECTION_COMPLETE"
}

write_state 'phase=collection'
for stage_index in 0 1 2; do
    case "${stage_index}" in
        0) stage=stage1_ood; start_seed=63000 ;;
        1) stage=stage2_ood; start_seed=63100 ;;
        2) stage=stage3_ood; start_seed=63200 ;;
    esac
    run_collection "${stage}" immediate 0 0-7 "${start_seed}" & p0=$!
    run_collection "${stage}" pre_stage 1 8-15 "${start_seed}" & p1=$!
    wait "${p0}" "${p1}"
    run_collection "${stage}" capability_boundary 0 0-7 "${start_seed}" & p0=$!
    run_collection "${stage}" failure_recovery 1 8-15 "${start_seed}" & p1=$!
    wait "${p0}" "${p1}"
done
printf '%s\n' 'complete' > "${ROOT}/FORMAL_COLLECTION_COMPLETE"

write_state 'phase=budget_selection'
for stage in stage1_ood stage2_ood stage3_ood; do
    "${PYTHON}" "${REPO_ROOT}/tools/prepare_stackpyramid_timing_sweep.py" \
        --output "${ROOT}/selected/${stage}" \
        --stage "${stage}" \
        --max-budget 2002 \
        --source "immediate=${ROOT}/collection/${stage}/immediate/accepted_suffixes.h5" \
        --source "pre_stage=${ROOT}/collection/${stage}/pre_stage/accepted_suffixes.h5" \
        --source "capability_boundary=${ROOT}/collection/${stage}/capability_boundary/accepted_suffixes.h5" \
        --source "failure_recovery=${ROOT}/collection/${stage}/failure_recovery/accepted_suffixes.h5"
done
printf '%s\n' 'complete' > "${ROOT}/BUDGET_SELECTION_COMPLETE"

write_state 'phase=training'
"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_timing_training.py" \
    --output-root "${ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --python "${PYTHON}" \
    --base-model "${MODEL}" \
    --id-h5 "${ID_H5}" \
    --gpus 0,1 \
    --cpu-sets 0-7,8-15 \
    --steps 2000 \
    --batch-size 8 \
    --seed 9200
test -f "${ROOT}/TIMING_TRAINING_COMPLETE"

write_state 'phase=evaluation'
"${PYTHON}" "${REPO_ROOT}/tools/run_stackpyramid_timing_evaluation.py" \
    --output-root "${ROOT}" \
    --repo-root "${REPO_ROOT}" \
    --xvla-root "${XVLA_ROOT}" \
    --python "${PYTHON}" \
    --gpus 0,1 \
    --cpu-sets 0-7,8-15
test -f "${ROOT}/TIMING_EVALUATION_COMPLETE"

write_state 'phase=summary'
"${PYTHON}" - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = {"format": "stackpyramid_formal_timing_summary_v1", "stages": {}}
for stage in ("stage1_ood", "stage2_ood", "stage3_ood"):
    rows = {}
    budget = json.loads((root / "selected" / stage / "budget_manifest.json").read_text())
    for condition in ("immediate", "pre_stage", "capability_boundary", "failure_recovery"):
        collection = json.loads((root / "collection" / stage / condition / "summary.json").read_text())
        evaluation = json.loads((root / "evaluation" / stage / condition / stage / "summary.json").read_text())
        id_eval = json.loads((root / "evaluation" / stage / condition / "id" / "summary.json").read_text())
        rows[condition] = {
            "accepted_episodes": collection["accepted_total"],
            "raw_attempts": collection["raw_attempts"],
            "raw_successes": collection["raw_successes"],
            "selected_expert_actions": budget["conditions"][condition]["selected_expert_action_steps"],
            "ood_strict_success": evaluation["strict_success"],
            "ood_episodes": evaluation["episodes"],
            "ood_success_rate": evaluation["strict_success"] / evaluation["episodes"],
            "id_strict_success": id_eval["strict_success"],
            "id_episodes": id_eval["episodes"],
            "id_success_rate": id_eval["strict_success"] / id_eval["episodes"],
        }
    report["stages"][stage] = {"common_budget": budget["common_expert_action_budget"], "conditions": rows}
(root / "timing_summary.json").write_text(json.dumps(report, indent=2) + "\n")
PY
printf '%s\n' 'complete' > "${ROOT}/FORMAL_TIMING_COMPLETE"
write_state 'phase=complete'
