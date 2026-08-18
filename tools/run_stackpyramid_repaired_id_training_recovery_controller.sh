#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/Ask4Help-xvla-stackpyramid-v4"
PY="/root/.venvs/xvla-h20/bin/python"
XVLA_ROOT="/root/X-VLA"
WORK="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1"
RUN_NAME="id_sft_10000_retry4"
LOCAL_OUTPUT="$WORK/$RUN_NAME"
PERSIST_OUTPUT="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v1/$RUN_NAME"
TRAIN_PID_FILE="$WORK/${RUN_NAME}.pid"
STATE_LOCAL="$WORK/${RUN_NAME}_recovery_pipeline_state.json"
GATE_SEED=887000

write_state() {
  printf '{"stage":"%s","status":"%s","output":"%s"}\n' "$1" "$2" "$PERSIST_OUTPUT" > "$STATE_LOCAL"
  mkdir -p "$PERSIST_OUTPUT" 2>/dev/null || true
  cp "$STATE_LOCAL" "$PERSIST_OUTPUT/pipeline_state_recovery.json" 2>/dev/null || true
}

sync_tree() {
  local source="$1"
  local name="$(basename "$source")"
  mkdir -p "$PERSIST_OUTPUT" 2>/dev/null || true
  if [[ -e "$PERSIST_OUTPUT/$name" ]]; then
    return 0
  fi
  cp -a "$source" "$PERSIST_OUTPUT/"
}

test -f "$TRAIN_PID_FILE"
TRAIN_PID="$(cat "$TRAIN_PID_FILE")"
kill -0 "$TRAIN_PID" 2>/dev/null
test -d "$LOCAL_OUTPUT/ckpt-1000"
mkdir -p "$LOCAL_OUTPUT/id_gate"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"

write_state "id_gate_step_1000" "RECOVERY_READY"
if [[ -f "$LOCAL_OUTPUT/id_gate/step-1000/summary.json" ]]; then
  SUCCESS=$($PY - "$LOCAL_OUTPUT/id_gate/step-1000/summary.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1])).get("strict_success", 0)))
PY
)
else
  echo "missing completed step-1000 summary" >&2
  exit 1
fi
printf '{"step":1000,"episodes":20,"strict_successes":%s,"threshold":14}\n' "$SUCCESS" > "$LOCAL_OUTPUT/id_gate/step-1000/decision.json"
printf 'step=1000 strict_successes=%s threshold=14\n' "$SUCCESS" > "$LOCAL_OUTPUT/id_gate/step-1000/ID_GATE_NOT_PASSED"
sync_tree "$LOCAL_OUTPUT/ckpt-1000"
sync_tree "$LOCAL_OUTPUT/id_gate/step-1000"

kill -CONT "$TRAIN_PID"
write_state "fresh_id_training" "RUNNING_FROM_STEP_1000_RECOVERY"

for STEP in 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
  CKPT="$LOCAL_OUTPUT/ckpt-$STEP"
  while [[ ! -d "$CKPT" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      write_state "fresh_id_training" "FAILED_BEFORE_CKPT_$STEP"
      exit 1
    fi
    sleep 30
  done
  sleep 10
  write_state "id_gate_step_$STEP" "RUNNING"
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -STOP "$TRAIN_PID"
  elif [[ "$STEP" != "10000" ]]; then
    echo "training exited before expected checkpoint gate $STEP" >&2
    write_state "fresh_id_training" "FAILED_PID_EXITED_BEFORE_GATE_$STEP"
    exit 1
  fi
  GATE_OUT="$LOCAL_OUTPUT/id_gate/step-$STEP"
  mkdir -p "$LOCAL_OUTPUT/id_gate"
  "$PY" tools/evaluate_stackpyramid_xvla.py \
    --checkpoint "$CKPT" --xvla-root "$XVLA_ROOT" --output "$GATE_OUT" \
    --split id --episodes 20 --start-seed "$GATE_SEED" \
    --max-episode-steps 300 --execute-horizon 5 --flow-steps 5 \
    --device cuda --sim-backend gpu --render-backend gpu > "$LOCAL_OUTPUT/id_gate/step-$STEP.log" 2>&1
  SUCCESS=$($PY - "$GATE_OUT/summary.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1])).get("strict_success", 0)))
PY
)
  printf '{"step":%s,"episodes":20,"strict_successes":%s,"threshold":14}\n' "$STEP" "$SUCCESS" > "$GATE_OUT/decision.json"
  if (( SUCCESS >= 14 )); then
    printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$SUCCESS" > "$GATE_OUT/ID_GATE_PASSED"
  else
    printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$SUCCESS" > "$GATE_OUT/ID_GATE_NOT_PASSED"
  fi
  sync_tree "$CKPT"
  sync_tree "$GATE_OUT"
  rm -rf "$CKPT"
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -CONT "$TRAIN_PID"
  fi
  write_state "fresh_id_training" "RUNNING_AFTER_GATE_$STEP"
done

if kill -0 "$TRAIN_PID" 2>/dev/null; then
  wait "$TRAIN_PID"
fi
if [[ ! -f "$LOCAL_OUTPUT/TRAINING_COMPLETE" ]]; then
  write_state "fresh_id_training" "FAILED_NO_COMPLETION_MARKER"
  exit 1
fi
cp "$LOCAL_OUTPUT/TRAINING_COMPLETE" "$PERSIST_OUTPUT/TRAINING_COMPLETE" 2>/dev/null || true
cp "$LOCAL_OUTPUT/training_config.json" "$PERSIST_OUTPUT/training_config.json" 2>/dev/null || true
cp "$LOCAL_OUTPUT/anchor_report.json" "$PERSIST_OUTPUT/anchor_report.json" 2>/dev/null || true
cp "$LOCAL_OUTPUT/train.jsonl" "$PERSIST_OUTPUT/train.jsonl" 2>/dev/null || true
FINAL_SUCCESS=$($PY - "$LOCAL_OUTPUT/id_gate/step-10000/decision.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["strict_successes"]))
PY
)
if (( FINAL_SUCCESS >= 14 )); then
  printf 'step=10000 strict_successes=%s threshold=14\n' "$FINAL_SUCCESS" > "$PERSIST_OUTPUT/ID_BASE_VALIDATED"
  write_state "id_base_validated" "PASSED"
else
  printf 'step=10000 strict_successes=%s threshold=14\n' "$FINAL_SUCCESS" > "$PERSIST_OUTPUT/ID_BASE_NOT_ACCEPTED"
  write_state "id_base_validated" "FAILED"
fi
