#!/usr/bin/env bash
set -u

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:?set OPEN_DRAWER_TIMING_ROOT}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
ID_DATASET=${OPEN_DRAWER_TIMING_ID_DATASET:?set OPEN_DRAWER_TIMING_ID_DATASET}
STATE=$RUN/pipeline_supervisor_state.json
LOG=$RUN/pipeline_supervisor.log
FORMAL=$RUN/formal
FORMAL=${OPEN_DRAWER_TIMING_FORMAL_ROOT:-$RUN/formal}
BUDGET=${OPEN_DRAWER_TIMING_BUDGET_ROOT:-$RUN/formal_budget}
POLICY_ONLY=${OPEN_DRAWER_TIMING_POLICY_ONLY_ROOT:-$RUN/policy_only_grasp_ood}
FORMAL_MARKER=${OPEN_DRAWER_TIMING_FORMAL_MARKER:-$RUN/TIMING_COLLECTION_COMPLETE}
BUDGET_VALUE=${OPEN_DRAWER_TIMING_BUDGET:-5006}
ANCHORS=(0 50 80 120 160 220)
TRAIN_SEEDS=(9301 9302 9303)
EVAL_GPU=4
EVAL_CPU=80-99

mkdir -p "$RUN" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_grasp_timing_pipeline_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

fail() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "stage=$1 detail=${2:-}" > "$RUN/PIPELINE_FAILED"
  exit 1
}

write_state waiting_for_formal_collection running marker_pending
while [[ ! -e "$FORMAL_MARKER" ]]; do
  if [[ -e "$RUN/PIPELINE_FAILED" ]]; then fail formal_collection "collection_controller_failed"; fi
  if [[ -e "$RUN/formal/AUDIT_FAILED" ]]; then fail formal_collection "formal_audit_failed"; fi
  echo "$(date -Is) waiting for formal TIMING_COLLECTION_COMPLETE"
  sleep 300
done

write_state formal_collection_audit running denominator_and_evidence
"$PY" -u "$ROOT/tools/audit_open_drawer_grasp_timing.py" --root "$FORMAL" --anchors "${ANCHORS[@]}" --target 30 > "$RUN/formal_audit.log" 2>&1 || fail formal_collection_audit "audit_failed"
if [[ ! -e "$POLICY_ONLY/POLICY_ONLY_COMPLETE" ]]; then
  if [[ -e "$POLICY_ONLY" && -n "$(find "$POLICY_ONLY" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail policy_only_timeline "partial_policy_only_output_exists"
  fi
  write_state policy_only_timeline running "20 Grasp-OOD policy-only episodes"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
    ASK4HELP_RLINF_ROOT="$ROOT/RLinf" PYTHONPATH="$ROOT:$ROOT/RLinf" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    taskset -c 0-19 "$PY" -u "$ROOT/tools/collect_open_drawer_policy_only_timeline.py" \
      --checkpoint "$MODEL" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
      --output-root "$POLICY_ONLY" --episodes 20 --seed 78700 > "$RUN/policy_only.log" 2>&1 || true
  if ! "$PY" - "$POLICY_ONLY" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
summary=root/"summary.json"
if not summary.is_file(): raise SystemExit("missing policy-only summary")
d=json.loads(summary.read_text())
if d.get("episodes") != 20 or len(d.get("rows",[])) != 20 or len(list((root/"videos").glob("*.mp4"))) != 20:
    raise SystemExit("policy-only denominator/artifact mismatch")
(root/"POLICY_ONLY_COMPLETE").write_text("20 policy-only task-state timelines complete\n")
PY
  then
    fail policy_only_timeline "denominator_or_artifact_audit_failed"
  fi
fi
write_state d_path_analysis running prefix_and_reference
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$ROOT/tools/analyze_open_drawer_grasp_timing.py" --root "$FORMAL" --reference-anchor 0 --anchors "${ANCHORS[@]}" --policy-only-root "$POLICY_ONLY" --output "$FORMAL/d_path_summary.json" > "$RUN/d_path_analysis.log" 2>&1 || fail d_path_analysis "analysis_failed"

if [[ ! -e "$BUDGET/BUDGET_AUDIT_PASS" ]]; then
  write_state exact_budget_selection running all_complete_anchor_datasets
  if [[ -e "$BUDGET" && -n "$(find "$BUDGET" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail exact_budget_selection "partial_budget_output_exists"
  fi
  args=(--output-root "$BUDGET" --budget "$BUDGET_VALUE")
  for step in "${ANCHORS[@]}"; do
    args+=(--condition "anchor_${step}=$FORMAL/anchor_${step}/lerobot_dataset")
  done
  "$PY" -u "$ROOT/tools/prepare_open_drawer_timing_budget.py" "${args[@]}" > "$RUN/formal_budget.log" 2>&1 || fail exact_budget_selection "no_exact_whole_episode_budget"
else
  echo "exact budget already audited: $BUDGET"
fi

if [[ ! -e "$RUN/TIMING_TRAINING_COMPLETE" ]]; then
  write_state matched_budget_training running three_seeds_per_anchor
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_RLINF_ROOT="$ROOT/RLinf" OPEN_DRAWER_PYTHON="$PY" \
    OPEN_DRAWER_TIMING_ROOT="$RUN" OPEN_DRAWER_TIMING_CHECKPOINT="$MODEL" \
    OPEN_DRAWER_TIMING_ID_DATASET="$ID_DATASET" OPEN_DRAWER_TIMING_NORM="$NORM" \
    OPEN_DRAWER_TIMING_BUDGET_ROOT="$BUDGET" OPEN_DRAWER_TIMING_TRAIN_STEPS=2500 \
    bash "$ROOT/tools/run_open_drawer_grasp_timing_training.sh" || fail matched_budget_training training_controller_failed
fi

if [[ ! -e "$RUN/TIMING_EVALUATION_COMPLETE" ]]; then
  write_state matched_budget_evaluation running id_and_grasp_ood
  for anchor in "${ANCHORS[@]}"; do
    for index in "${!TRAIN_SEEDS[@]}"; do
      seed=${TRAIN_SEEDS[$index]}
      gpu=$EVAL_GPU
      cpuset=$EVAL_CPU
      out="$RUN/evaluation/anchor_${anchor}/seed_${seed}"
      checkpoint="$RUN/training/anchor_${anchor}/seed_${seed}/run/checkpoints/global_step_2500"
      if [[ -e "$out/EVAL_COMPLETE" ]]; then
        continue
      fi
      if [[ ! -f "$checkpoint/actor/model_state_dict/full_weights.pt" ]]; then
        fail matched_budget_evaluation "missing_checkpoint_anchor_${anchor}_seed_${seed}"
      fi
      if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        fail matched_budget_evaluation "partial_eval_output_anchor_${anchor}_seed_${seed}"
      fi
      mkdir -p "$(dirname "$out")"
      env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_PYTHON="$PY" \
        OPEN_DRAWER_TIMING_ROOT="$RUN" taskset -c "$cpuset" "$PY" -u "$ROOT/tools/run_open_drawer_timing_eval.py" \
        --root "$ROOT" --checkpoint "$checkpoint" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
        --evaluator "$ROOT/tools/evaluate_open_drawer_id_pi05.py" --output-root "$out" \
        --python "$PY" --episodes 100 --id-seed 78500 --ood-seed 78600 --gpu "$gpu" --cpu-set "$cpuset" \
        > "$RUN/logs/eval_anchor_${anchor}_seed_${seed}.log" 2>&1
      if [[ ! -e "$out/EVAL_COMPLETE" ]]; then
        fail matched_budget_evaluation "evaluation_anchor_${anchor}_seed_${seed}_failed"
      fi
    done
  done
  printf '%s\n' 'all timing anchors and training seeds evaluated on 100 ID/100 Grasp-OOD episodes' > "$RUN/TIMING_EVALUATION_COMPLETE"
fi

write_state independent_reconciliation running metrics_denominator_and_artifact_audit
"$PY" -u "$ROOT/tools/summarize_open_drawer_grasp_timing.py" --root "$RUN" --anchors "${ANCHORS[@]}" --seeds "${TRAIN_SEEDS[@]}" --output "$RUN/final_report.json" > "$RUN/final_report.log" 2>&1 || fail independent_reconciliation "final_report_audit_failed"
if [[ ! -e "$RUN/INDEPENDENT_RECONCILIATION_COMPLETE" ]]; then
  fail independent_reconciliation "reconciliation_marker_missing"
fi
printf '%s\n' 'OpenDrawer Grasp-OOD timing sweep complete after independent reconciliation.' > "$RUN/PIPELINE_COMPLETE"
write_state pipeline_complete complete final_report_and_reconciliation_present
echo OPEN_DRAWER_GRASP_TIMING_PIPELINE_COMPLETE
