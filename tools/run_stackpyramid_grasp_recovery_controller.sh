#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
BASE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000"
BASELINE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/final_checkpoint_formal_id_gate_100_retry3"
ROOT_OUT="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1"
WORK="/tmp/stackpyramid_grasp_recovery_v1"
STATE="$ROOT_OUT/pipeline_state.json"
LOG="/root/ask4help_stage2_logs/stackpyramid_grasp_recovery_v1_controller.log"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$ROOT_OUT" "$WORK"

state() {
  printf '{"format":"stackpyramid_grasp_recovery_controller_v1","stage":"%s","status":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$(date '+%Y-%m-%dT%H:%M:%S%z')" > "$STATE"
}

if [[ ! -f "$BASELINE/EVAL_COMPLETE" ]]; then
  state baseline_failure_audit FAILED_MISSING_BASELINE
  exit 2
fi

if [[ ! -f "$ROOT_OUT/baseline_failure_audit.json" ]]; then
  state baseline_failure_audit RUNNING
  "$PY" "$ROOT/tools/audit_stackpyramid_grasp_recovery.py" \
    --root "$BASELINE" --output "$ROOT_OUT/baseline_failure_audit.json" >> "$LOG" 2>&1
  state baseline_failure_audit PASSED
fi

HORIZON_LOCAL="$WORK/horizon_450_diagnostic_20_retry2"
HORIZON_OUT="$ROOT_OUT/horizon_450_diagnostic_20_retry2"
if [[ ! -f "$HORIZON_OUT/EVAL_COMPLETE" ]]; then
  state horizon_450_diagnostic_20 RUNNING
  rm -rf "$HORIZON_LOCAL"
  "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$BASE" --xvla-root "$XVLA_ROOT" --output "$HORIZON_LOCAL" \
    --split id --episodes 20 --start-seed 84400 --max-episode-steps 450 \
    --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
    --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode >> "$LOG" 2>&1
  cp -a "$HORIZON_LOCAL" "$HORIZON_OUT"
  printf 'diagnostic only; official 300-step protocol unchanged\n' > "$HORIZON_OUT/HORIZON_450_DIAGNOSTIC"
  state horizon_450_diagnostic_20 PASSED
fi

if [[ ! -f "$ROOT_OUT/adapter_gripper_audit.json" ]]; then
  state adapter_gripper_audit RUNNING
  "$PY" "$ROOT/tools/audit_stackpyramid_grasp_adapter.py" \
    --baseline-root "$BASELINE" --output "$ROOT_OUT/adapter_gripper_audit.json" >> "$LOG" 2>&1
  state adapter_gripper_audit PASSED
fi

REC_LOCAL="$WORK/recovery_collection_128_retry1"
REC_OUT="$ROOT_OUT/recovery_collection_128_retry1"
if [[ ! -f "$REC_OUT/COLLECTION_COMPLETE" ]]; then
  state same_id_recovery_collection RUNNING
  rm -rf "$REC_LOCAL"
  "$PY" "$ROOT/tools/collect_stackpyramid_xvla_dagger.py" \
    --method offline_oracle --checkpoint "$BASE" --xvla-root "$XVLA_ROOT" \
    --output-dir "$REC_LOCAL" --split id --target 128 --seed-start 887000 \
    --max-attempts 180 --fresh-env-per-episode --full-planner \
    --sim-backend gpu --render-backend gpu >> "$LOG" 2>&1
  "$PY" "$ROOT/tools/audit_stackpyramid_id_collection.py" \
    --collection-root "$REC_LOCAL" --output "$WORK/recovery_collection_128_audit" \
    --task-spec "$ROOT/configs/stackpyramid_v4_task_spec.json" --expected-episodes 128 >> "$LOG" 2>&1
  cp -a "$REC_LOCAL" "$REC_OUT"
  cp -a "$WORK/recovery_collection_128_audit" "$ROOT_OUT/recovery_collection_128_audit"
  state same_id_recovery_collection PASSED
fi

state recovery_training NEEDS_USER_DECISION
printf 'recovery collection audited; recovery training launch requires explicit exposure review\n' > "$ROOT_OUT/NEEDS_USER_DECISION"
