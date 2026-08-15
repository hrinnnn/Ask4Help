#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:?set MODEL}"
XVLA_ROOT="${XVLA_ROOT:?set XVLA_ROOT}"
TASK_ROOT="${TASK_ROOT:?set TASK_ROOT}"
AUDIT="${AUDIT:?set AUDIT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
PYTHON="${PYTHON:?set PYTHON}"

if [[ -e "$OUTPUT_ROOT" ]]; then
    echo "refusing to reuse timing collection output: $OUTPUT_ROOT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' 'format=stackpyramid_timing_collection_controller_v1' > "$OUTPUT_ROOT/pipeline_state.txt"

run_condition() {
    local split="$1"
    local condition="$2"
    local start_seed="$3"
    local output="$OUTPUT_ROOT/$split/$condition"
    if [[ -e "$output" ]]; then
        echo "refusing to reuse condition output: $output" >&2
        exit 2
    fi
    mkdir -p "$OUTPUT_ROOT/$split"
    printf 'stage=%s condition=%s status=running\n' "$split" "$condition" > "$OUTPUT_ROOT/pipeline_state.txt"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 "$PYTHON" \
        "$TASK_ROOT/tools/collect_stackpyramid_timing_sweep.py" \
        --checkpoint "$MODEL" \
        --xvla-root "$XVLA_ROOT" \
        --task-root "$TASK_ROOT" \
        --audit "$AUDIT" \
        --output "$output" \
        --split "$split" \
        --condition "$condition" \
        --target 20 \
        --max-attempts 40 \
        --start-seed "$start_seed" \
        --pre-offset 25 \
        --boundary-offset 5 \
        --recovery-delay 25 \
        --flow-steps 5 \
        --sim-backend cpu \
        --render-backend cpu
    test -f "$output/COLLECTION_COMPLETE"
    printf 'stage=%s condition=%s status=complete\n' "$split" "$condition" > "$OUTPUT_ROOT/pipeline_state.txt"
}

run_condition stage1_ood immediate 63000
run_condition stage1_ood pre_stage 63000
run_condition stage1_ood capability_boundary 63000
run_condition stage1_ood failure_recovery 63000

run_condition stage2_ood immediate 63100
run_condition stage2_ood pre_stage 63100
run_condition stage2_ood capability_boundary 63100
run_condition stage2_ood failure_recovery 63100

run_condition stage3_ood immediate 63200
run_condition stage3_ood pre_stage 63200
run_condition stage3_ood capability_boundary 63200
run_condition stage3_ood failure_recovery 63200

printf '%s\n' 'status=complete' > "$OUTPUT_ROOT/pipeline_state.txt"
printf '%s\n' 'complete' > "$OUTPUT_ROOT/FORMAL_COLLECTION_COMPLETE"
