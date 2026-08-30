#!/usr/bin/env bash
set -u

# End-to-end controller after the reviewed direct-grasp Oracle is approved:
# wait for the new formal collection, select a new exact common budget, run
# the bounded adaptive trainer, then launch the audited 100-ID/100-OOD eval.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
ID_DATASET=${OPEN_DRAWER_TIMING_ID_DATASET:?set OPEN_DRAWER_TIMING_ID_DATASET}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
FORMAL_RUN=${OPEN_DRAWER_DIRECT_ORACLE_FORMAL_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1}
FORMAL_ROOT=$FORMAL_RUN/formal
BUDGET_ROOT=${OPEN_DRAWER_DIRECT_ORACLE_BUDGET_ROOT:-$FORMAL_RUN/formal_budget}
RUN=${OPEN_DRAWER_DIRECT_ORACLE_ADAPTIVE_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_adaptive_retry1}
FORMAL_MARKER=$FORMAL_RUN/TIMING_COLLECTION_COMPLETE
GPU_POOL=${OPEN_DRAWER_TIMING_GPU_POOL:-"0 1 2 3 4 5 6 7"}
RAY_TMP_ROOT=${OPEN_DRAWER_TIMING_RAY_TMP_ROOT:-/sdd/od_direct_oracle_adaptive_ray_retry1}
TMP_ROOT=${OPEN_DRAWER_TIMING_TMP_ROOT:-/sdd/od_direct_oracle_adaptive_tmp_retry1}
STATE=$RUN/direct_oracle_adaptive_total_state.json
LOG=$RUN/direct_oracle_adaptive_total_controller.log

mkdir -p "$RUN" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_direct_oracle_adaptive_total_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

fail() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "stage=$1 detail=${2:-}" > "$RUN/DIRECT_ORACLE_ADAPTIVE_FAILED"
  exit 1
}

write_state waiting_for_formal_collection running "marker=$FORMAL_MARKER"
while [[ ! -f "$FORMAL_MARKER" ]]; do
  if [[ -f "$FORMAL_RUN/DIRECT_ORACLE_FORMAL_COLLECTION_FAILED" ]]; then
    fail formal_collection "formal_collection_failed"
  fi
  sleep 300
done
[[ -f "$FORMAL_ROOT/AUDIT_PASS" ]] || fail formal_collection_audit "missing direct Oracle formal audit pass"

if [[ ! -f "$BUDGET_ROOT/BUDGET_AUDIT_PASS" ]]; then
  if [[ -e "$BUDGET_ROOT" && -n "$(find "$BUDGET_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail exact_budget_selection "refusing partial budget root: $BUDGET_ROOT"
  fi
  write_state exact_budget_selection running "maximum common whole-episode budget from new direct Oracle data"
  args=(--output-root "$BUDGET_ROOT" --budget auto_max_common)
  for anchor in 0 50 80 120 160 220; do
    args+=(--condition "anchor_${anchor}=$FORMAL_ROOT/anchor_${anchor}/lerobot_dataset")
  done
  "$PY" -u "$ROOT/tools/prepare_open_drawer_timing_budget.py" "${args[@]}" \
    > "$RUN/logs/direct_oracle_budget.log" 2>&1 || fail exact_budget_selection "budget_selector_failed"
fi
[[ -f "$BUDGET_ROOT/BUDGET_AUDIT_PASS" ]] || fail exact_budget_selection "budget audit marker missing"

write_state budget_audit running "new direct Oracle budget root=$BUDGET_ROOT"
set +e
"$PY" - "$BUDGET_ROOT" "$FORMAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

budget = Path(sys.argv[1])
formal = Path(sys.argv[2])
manifest = json.loads((budget / "budget_manifest.json").read_text())
selected = manifest.get("selected_expert_actions", {})
if len(selected) != 6 or len(set(int(v) for v in selected.values())) != 1:
    raise SystemExit(f"selected budgets are not common: {selected}")
if manifest.get("budget_selection_rule") != "maximum_common_reachable_whole_episode_sum":
    raise SystemExit(f"unexpected budget rule: {manifest.get('budget_selection_rule')}")
for anchor in (0, 50, 80, 120, 160, 220):
    summary = json.loads((formal / f"anchor_{anchor}" / "summary.json").read_text())
    if summary.get("oracle_mode") != "direct_grasp":
        raise SystemExit(f"non-direct summary in anchor_{anchor}")
print(json.dumps({"common_budget": next(iter(selected.values())), "rule": manifest["budget_selection_rule"]}))
PY
budget_audit_rc=$?
set -e
[[ "$budget_audit_rc" -eq 0 ]] || fail budget_audit "budget_manifest_audit_failed"

if [[ ! -f "$RUN/ADAPTIVE_TIMING_TRAINING_COMPLETE" ]]; then
  write_state adaptive_training running "one model per anchor; up to four audited idle GPUs"
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_RLINF_ROOT="$RL" \
    OPEN_DRAWER_PYTHON="$PY" OPEN_DRAWER_TIMING_ROOT="$RUN" \
    OPEN_DRAWER_TIMING_CHECKPOINT="$MODEL" OPEN_DRAWER_TIMING_PI05_BASE="$PI05_BASE" \
    OPEN_DRAWER_TIMING_ID_DATASET="$ID_DATASET" OPEN_DRAWER_TIMING_NORM="$NORM" \
    OPEN_DRAWER_TIMING_FORMAL_ROOT="$FORMAL_ROOT" OPEN_DRAWER_TIMING_BUDGET_ROOT="$BUDGET_ROOT" \
    OPEN_DRAWER_TIMING_GPU_POOL="$GPU_POOL" OPEN_DRAWER_TIMING_RAY_TMP_ROOT="$RAY_TMP_ROOT" \
    OPEN_DRAWER_TIMING_TMP_ROOT="$TMP_ROOT" OPEN_DRAWER_RAY_OBJECT_STORE_MEMORY=$((100 * 1024**3)) \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    "$PY" -u "$ROOT/tools/run_open_drawer_adaptive_parallel_controller.py" \
    > "$RUN/logs/direct_oracle_adaptive_parallel.log" 2>&1 || fail adaptive_training "adaptive_parallel_controller_failed"
fi
[[ -f "$RUN/ADAPTIVE_TIMING_TRAINING_COMPLETE" ]] || fail adaptive_training "training completion marker missing"

if [[ ! -f "$RUN/PIPELINE_COMPLETE" ]]; then
  write_state formal_evaluation running "100-ID and 100-Grasp-OOD per anchor"
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_RLINF_ROOT="$RL" \
    OPEN_DRAWER_PYTHON="$PY" OPEN_DRAWER_TIMING_ROOT="$RUN" \
    OPEN_DRAWER_TIMING_CHECKPOINT="$MODEL" OPEN_DRAWER_TIMING_PI05_BASE="$PI05_BASE" \
    OPEN_DRAWER_TIMING_NORM="$NORM" OPEN_DRAWER_TIMING_FORMAL_ROOT="$FORMAL_ROOT" \
    OPEN_DRAWER_TIMING_BUDGET_ROOT="$BUDGET_ROOT" OPEN_DRAWER_TIMING_FORMAL_MARKER="$FORMAL_MARKER" \
    OPEN_DRAWER_TIMING_GPU_POOL="$GPU_POOL" OPEN_DRAWER_TIMING_EVALUATOR="$ROOT/tools/evaluate_open_drawer_id_pi05.py" \
    OPEN_DRAWER_TIMING_EVAL_WRAPPER="$ROOT/tools/run_open_drawer_timing_eval.py" \
    OPEN_DRAWER_TIMING_RECONCILER="$ROOT/tools/summarize_open_drawer_adaptive_timing.py" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    bash "$ROOT/tools/run_open_drawer_adaptive_formal_eval_controller.sh" \
    > "$RUN/logs/direct_oracle_adaptive_formal_eval.log" 2>&1 || fail formal_evaluation "formal_eval_controller_failed"
fi
[[ -f "$RUN/INDEPENDENT_RECONCILIATION_COMPLETE" ]] || fail independent_reconciliation "missing reconciliation marker"
[[ -f "$RUN/final_report.json" && -f "$RUN/final_report.md" ]] || fail independent_reconciliation "missing final report"
printf '%s\n' 'direct Oracle adaptive timing pipeline complete after independent reconciliation' > "$RUN/PIPELINE_COMPLETE"
write_state pipeline_complete complete "new direct Oracle data, common budget, adaptive training and formal evaluation audited"
echo OPEN_DRAWER_DIRECT_ORACLE_ADAPTIVE_PIPELINE_COMPLETE
