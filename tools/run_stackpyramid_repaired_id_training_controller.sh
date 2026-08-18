#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/Ask4Help-xvla-stackpyramid-v4"
PY="/root/.venvs/xvla-h20/bin/python"
XVLA_ROOT="/root/X-VLA"
BASE_MODEL="/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4"
INPUT_ROOT="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1/fresh_id_training_input_256"
WORK="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1"
RUN_NAME="${RUN_NAME:-id_sft_10000_retry1}"
OUTPUT="${OUTPUT:-/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v1/$RUN_NAME}"
TRAIN_LOG="$WORK/${RUN_NAME}.log"
TRAIN_PID_FILE="$WORK/${RUN_NAME}.pid"
GATE_SEED=887000

write_state() {
  local stage="$1"
  local status="$2"
  printf '{"stage":"%s","status":"%s","output":"%s"}\n' "$stage" "$status" "$OUTPUT" > "$OUTPUT/pipeline_state.json"
}

if [[ -e "$OUTPUT" ]]; then
  echo "refusing to overwrite $OUTPUT" >&2
  exit 2
fi
test -f "$INPUT_ROOT/id/accepted_suffixes.h5"
test -f "$WORK/fresh_id_train_smoke_canonical_base_retry1/RELOAD_SMOKE_COMPLETE"
mkdir -p "$WORK"
rm -rf "$OUTPUT"
mkdir -p "$OUTPUT"

write_state "fresh_id_training" "STARTING"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"

nohup "$PY" tools/run_stackpyramid_xvla_training.py \
  --xvla-root "$XVLA_ROOT" \
  --model "$BASE_MODEL" \
  --collection-root "$INPUT_ROOT" \
  --split id \
  --target-episodes 256 \
  --output "$OUTPUT" \
  --steps 10000 \
  --save-interval 1000 \
  --batch-size 8 \
  --learning-rate 2.5e-5 \
  --learning-coef 0.1 \
  --weight-decay 0.0 \
  --freeze-steps 1000 \
  --warmup-steps 2000 \
  --log-interval 20 \
  --seed 886000 \
  --dtype bf16 > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$TRAIN_PID_FILE"
write_state "fresh_id_training" "RUNNING"

for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
  CKPT="$OUTPUT/ckpt-$STEP"
  while [[ ! -d "$CKPT" ]]; do
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "training exited before ckpt-$STEP" >&2
      write_state "fresh_id_training" "FAILED_BEFORE_CKPT_$STEP"
      exit 1
    fi
    sleep 30
  done
  sleep 10
  write_state "id_gate_step_$STEP" "RUNNING"
  kill -STOP "$TRAIN_PID"
  GATE_OUT="$OUTPUT/id_gate/step-$STEP"
  if [[ ! -e "$GATE_OUT" ]]; then
    "$PY" tools/evaluate_stackpyramid_xvla.py \
      --checkpoint "$CKPT" \
      --xvla-root "$XVLA_ROOT" \
      --output "$GATE_OUT" \
      --split id \
      --episodes 20 \
      --start-seed "$GATE_SEED" \
      --max-episode-steps 300 \
      --execute-horizon 5 \
      --flow-steps 5 \
      --device cpu \
      --sim-backend cpu \
      --render-backend cpu > "$OUTPUT/id_gate/step-$STEP.log" 2>&1
  fi
  GATE_SUCCESS=$("$PY" - "$GATE_OUT/summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(int(data.get("strict_success", 0)))
PY
)
  printf '{"step":%s,"episodes":20,"strict_successes":%s,"threshold":14}\n' "$STEP" "$GATE_SUCCESS" > "$OUTPUT/id_gate/step-$STEP/decision.json"
  if (( GATE_SUCCESS >= 14 )); then
    printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$GATE_SUCCESS" > "$OUTPUT/id_gate/step-$STEP/ID_GATE_PASSED"
  else
    printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$GATE_SUCCESS" > "$OUTPUT/id_gate/step-$STEP/ID_GATE_NOT_PASSED"
  fi
  kill -CONT "$TRAIN_PID"
  write_state "fresh_id_training" "RUNNING_AFTER_GATE_$STEP"
done

wait "$TRAIN_PID"
if [[ ! -f "$OUTPUT/TRAINING_COMPLETE" ]]; then
  echo "training exited without TRAINING_COMPLETE" >&2
  write_state "fresh_id_training" "FAILED_NO_COMPLETION_MARKER"
  exit 1
fi

FINAL_SUCCESS=$(cat "$OUTPUT/id_gate/step-10000/decision.json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["strict_successes"])')
if (( FINAL_SUCCESS >= 14 )); then
  printf 'step=10000 strict_successes=%s threshold=14\n' "$FINAL_SUCCESS" > "$OUTPUT/ID_BASE_VALIDATED"
  write_state "id_base_validated" "PASSED"
else
  printf 'step=10000 strict_successes=%s threshold=14\n' "$FINAL_SUCCESS" > "$OUTPUT/ID_BASE_NOT_ACCEPTED"
  write_state "id_base_validated" "FAILED"
fi
