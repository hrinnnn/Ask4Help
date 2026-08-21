#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
BASE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000"
OLD_H5="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v3/id_training_collection_512_external_links_retry1/id/accepted_suffixes.h5"
REC_H5="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/recovery_collection_128_retry1/accepted_suffixes.h5"
PIPE_ROOT="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1"
RUN_ROOT="$PIPE_ROOT/recovery_training_50k_retry1"
WORK="/tmp/stackpyramid_grasp_recovery_50k_retry1"
MERGED="$WORK/merged_640_external_links"
SMOKE="$WORK/train_smoke_2step"
TRAIN="$RUN_ROOT/training"
SELECTION="$RUN_ROOT/selection"
FORMAL="$RUN_ROOT/formal_id_gate_100"
STATE="$PIPE_ROOT/pipeline_state.json"
STATE_LOCAL="$WORK/pipeline_state.json"
LOG="/root/ask4help_stage2_logs/stackpyramid_grasp_recovery_50k_retry1_controller.log"
TRAIN_PID_FILE="/root/ask4help_stage2_logs/stackpyramid_grasp_recovery_50k_retry1_train.pid"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORK" "$PIPE_ROOT" "$RUN_ROOT"

state() {
  local stage="$1" status="$2"
  printf '{"format":"stackpyramid_grasp_recovery_50k_controller_v1","stage":"%s","status":"%s","updated_at":"%s"}\n' "$stage" "$status" "$(date '+%Y-%m-%dT%H:%M:%S%z')" > "$STATE_LOCAL"
  cp "$STATE_LOCAL" "$STATE"
}

fail() {
  state "$1" "$2"
  exit 1
}

[[ -d "$BASE" ]] || fail preflight FAILED_MISSING_BASE
[[ -f "$REC_H5" ]] || fail preflight FAILED_MISSING_RECOVERY_H5
[[ -f "$OLD_H5" ]] || fail preflight FAILED_MISSING_EXISTING_H5
[[ -f "$PIPE_ROOT/recovery_collection_128_audit/ID_COLLECTION_AUDIT_PASS" ]] || fail preflight FAILED_MISSING_RECOVERY_AUDIT

if [[ ! -f "$MERGED/MERGE_COMPLETE" ]]; then
  state merged_data_audit RUNNING
  rm -rf "$MERGED"
  "$PY" "$ROOT/tools/merge_stackpyramid_id_external_links.py" \
    --original-h5 "$OLD_H5" --additional-h5 "$REC_H5" \
    --output-root "$MERGED" --manifest "$ROOT/configs/stackpyramid_v4_task_spec.json" \
    --expected-episodes 640 >> "$LOG" 2>&1
  "$PY" "$ROOT/tools/audit_stackpyramid_recovery_merge.py" \
    --collection-root "$MERGED" --output "$WORK/merged_640_audit.json" \
    --expected-episodes 640 >> "$LOG" 2>&1
  cp -a "$MERGED" "$RUN_ROOT/merged_640_external_links"
  cp "$WORK/merged_640_audit.json" "$RUN_ROOT/merged_640_audit.json"
  state merged_data_audit PASSED
fi

if [[ ! -f "$SMOKE/RELOAD_SMOKE_COMPLETE" ]]; then
  state train_reload_smoke RUNNING
  rm -rf "$SMOKE"
  "$PY" "$ROOT/tools/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA_ROOT" --model "$BASE" --collection-root "$MERGED" \
    --split id --target-episodes 640 --output "$SMOKE" --smoke-only \
    --batch-size 8 --source-balanced --existing-source-fraction 0.8 \
    --learning-rate 1e-4 --learning-coef 0.1 --freeze-steps 0 \
    --warmup-steps 2000 --dtype bf16 >> "$LOG" 2>&1
  cp -a "$SMOKE" "$RUN_ROOT/train_smoke_2step"
  state train_reload_smoke PASSED
fi

if [[ ! -f "$TRAIN/TRAINING_COMPLETE" ]]; then
  state recovery_training RUNNING
  rm -rf "$TRAIN"
  nohup "$PY" "$ROOT/tools/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA_ROOT" --model "$BASE" --collection-root "$MERGED" \
    --split id --target-episodes 640 --output "$TRAIN" --steps 50000 \
    --save-interval 5000 --batch-size 8 --source-balanced \
    --existing-source-fraction 0.8 --learning-rate 1e-4 \
    --learning-coef 0.1 --freeze-steps 0 --warmup-steps 2000 \
    --dtype bf16 > "$RUN_ROOT/training.log" 2>&1 &
  TRAIN_PID=$!
  printf '%s\n' "$TRAIN_PID" > "$TRAIN_PID_FILE"
  printf '{"pid":%s,"output":"%s","steps":50000}\n' "$TRAIN_PID" "$TRAIN" > "$RUN_ROOT/TRAINING_LAUNCHED.json"
else
  TRAIN_PID=""
fi

while [[ ! -f "$TRAIN/TRAINING_COMPLETE" ]]; do
  if [[ -n "$TRAIN_PID" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    fail recovery_training FAILED_TRAINING_PROCESS_EXITED
  fi
  sleep 60
done
state recovery_training PASSED

if [[ ! -f "$SELECTION/SELECTION_COMPLETE" ]]; then
  state checkpoint_selection_20_id RUNNING
  mkdir -p "$SELECTION"
  for STEP in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000; do
    OUT="$SELECTION/step-$STEP"
    if [[ -f "$OUT/EVAL_COMPLETE" ]]; then continue; fi
    "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
      --checkpoint "$TRAIN/ckpt-$STEP" --xvla-root "$XVLA_ROOT" --output "$OUT" \
      --split id --episodes 20 --start-seed 88500 --max-episode-steps 300 \
      --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
      --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode \
      > "$SELECTION/step-$STEP.log" 2>&1
  done
  "$PY" - "$SELECTION" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("step-*/summary.json"), key=lambda item: int(item.parent.name.split("-")[1])):
    data = json.loads(path.read_text())
    rows.append({"step": int(path.parent.name.split("-")[1]), "strict_success": int(data.get("strict_success", 0)), "episodes": int(data.get("episodes", 0))})
if len(rows) != 10 or any(row["episodes"] != 20 for row in rows):
    raise SystemExit("incomplete checkpoint selection")
best = max(rows, key=lambda row: (row["strict_success"], -row["step"]))
report = {"format": "stackpyramid_recovery_checkpoint_selection_v1", "rule": "max strict_success, earliest checkpoint on ties", "rows": rows, "selected": best, "seed_manifest": "88500--88519"}
(root / "selection_summary.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "SELECTED_STEP").write_text(str(best["step"]) + "\n")
(root / "SELECTION_COMPLETE").write_text("complete\n")
PY
  state checkpoint_selection_20_id PASSED
fi

SELECTED_STEP="$(cat "$SELECTION/SELECTED_STEP")"
SELECTED_CKPT="$TRAIN/ckpt-$SELECTED_STEP"
if [[ ! -f "$FORMAL/EVAL_COMPLETE" ]]; then
  state formal_gate_100_id RUNNING
  rm -rf "$FORMAL"
  "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$SELECTED_CKPT" --xvla-root "$XVLA_ROOT" --output "$FORMAL" \
    --split id --episodes 100 --start-seed 84400 --max-episode-steps 300 \
    --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
    --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode \
    > "$RUN_ROOT/formal_id_gate_100.log" 2>&1
fi

SUCCESS="$($PY - "$FORMAL/summary.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1])).get("strict_success", 0)))
PY
)"
VIDEO_COUNT="$(find "$FORMAL/videos" -type f -name '*.mp4' | wc -l)"
if [[ "$SUCCESS" -ge 80 ]] && [[ "$VIDEO_COUNT" -eq 100 ]]; then
  printf 'selected_step=%s strict_success=%s threshold=80\n' "$SELECTED_STEP" "$SUCCESS" > "$RUN_ROOT/ID_BASE_VALIDATED"
  state result_registration PIPELINE_COMPLETE
else
  printf 'selected_step=%s strict_success=%s threshold=80\n' "$SELECTED_STEP" "$SUCCESS" > "$RUN_ROOT/ID_BASE_NOT_ACCEPTED"
  state result_registration ID_BASE_NOT_ACCEPTED
fi
