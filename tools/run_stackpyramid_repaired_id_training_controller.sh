#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/Ask4Help-xvla-stackpyramid-v4"
PY="/root/.venvs/xvla-h20/bin/python"
XVLA_ROOT="/root/X-VLA"
BASE_MODEL="/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4"
INPUT_ROOT="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1/fresh_id_training_input_256"
WORK="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1"
RUN_NAME="${RUN_NAME:-id_sft_10000_generic_base_retry1}"
PERSIST_OUTPUT="${OUTPUT:-/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v1/$RUN_NAME}"
LOCAL_OUTPUT="$WORK/$RUN_NAME"
TRAIN_LOG="$WORK/${RUN_NAME}.log"
TRAIN_PID_FILE="$WORK/${RUN_NAME}.pid"
STATE_LOCAL="$WORK/${RUN_NAME}_pipeline_state.json"
GATE_SEED=887000
SMOKE_ROOT="${SMOKE_ROOT:-$WORK/fresh_id_train_smoke_canonical_base_retry1}"

write_state() {
  local stage="$1"
  local status="$2"
  printf '{"stage":"%s","status":"%s","output":"%s"}\n' "$stage" "$status" "$PERSIST_OUTPUT" > "$STATE_LOCAL"
  mkdir -p "$PERSIST_OUTPUT" 2>/dev/null || true
  cp "$STATE_LOCAL" "$PERSIST_OUTPUT/pipeline_state.json" 2>/dev/null || echo "warning: persistent state sync unavailable" >&2
}

sync_tree() {
  local source="$1"
  local name
  name="$(basename "$source")"
  mkdir -p "$PERSIST_OUTPUT" 2>/dev/null || true
  if [[ -e "$PERSIST_OUTPUT/$name" ]]; then
    echo "refusing to overwrite $PERSIST_OUTPUT/$name" >&2
    exit 2
  fi
  cp -a "$source" "$PERSIST_OUTPUT/"
}

if [[ -e "$PERSIST_OUTPUT" ]]; then
  echo "refusing to overwrite $PERSIST_OUTPUT" >&2
  exit 2
fi
test -f "$INPUT_ROOT/id/accepted_suffixes.h5"
if [[ -f "$SMOKE_ROOT/RELOAD_SMOKE_COMPLETE" ]]; then
  SMOKE_MARKER="$SMOKE_ROOT/RELOAD_SMOKE_COMPLETE"
elif [[ -f "$SMOKE_ROOT/smoke/RELOAD_SMOKE_COMPLETE" ]]; then
  SMOKE_MARKER="$SMOKE_ROOT/smoke/RELOAD_SMOKE_COMPLETE"
else
  echo "missing canonical corrected smoke marker under $SMOKE_ROOT" >&2
  exit 3
fi

case "$BASE_MODEL" in
  */xvla_stage2_inputs_priority/*|*/ckpt-7500)
    echo "forbidden StackCube Stage2 base: $BASE_MODEL" >&2
    exit 4
    ;;
esac
test -f "$BASE_MODEL/config.json"
test -f "$BASE_MODEL/model.safetensors"
"$PY" - "$BASE_MODEL/config.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if config.get("model_type") != "XVLA":
    raise SystemExit(f"unexpected base model_type: {config.get('model_type')!r}")
PY

if pgrep -af 'xvla_stage2_inputs_priority/ckpt-7500|run_stackpyramid_xvla_training.py.*ckpt-7500' >/dev/null 2>&1; then
  echo "refusing to start while a forbidden StackCube-base process is alive" >&2
  exit 5
fi
mkdir -p "$WORK"
rm -rf "$LOCAL_OUTPUT"
cat > "$WORK/${RUN_NAME}_formal_launch_manifest.json" <<EOF
{
  "base_model": "$BASE_MODEL",
  "collection_root": "$INPUT_ROOT",
  "collection_h5": "$INPUT_ROOT/id/accepted_suffixes.h5",
  "smoke_marker": "$SMOKE_MARKER",
  "steps": 10000,
  "save_interval": 1000,
  "id_gate_episodes": 20,
  "id_gate_seed_start": $GATE_SEED,
  "id_gate_minimum_successes": 14,
  "ood_started": false
}
EOF
printf 'base_model=%s\ncollection_root=%s\ncollection_h5=%s\nsmoke_marker=%s\n' \
  "$BASE_MODEL" "$INPUT_ROOT" "$INPUT_ROOT/id/accepted_suffixes.h5" "$SMOKE_MARKER" > "$WORK/${RUN_NAME}_formal_training_locks"

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
  --output "$LOCAL_OUTPUT" \
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
printf 'pid=%s\nmodel=%s\ncollection_root=%s\nlog=%s\n' \
  "$TRAIN_PID" "$BASE_MODEL" "$INPUT_ROOT" "$TRAIN_LOG" > "$WORK/${RUN_NAME}_formal_training_started"
while [[ ! -d "$LOCAL_OUTPUT" ]]; do sleep 1; done
cp "$WORK/${RUN_NAME}_formal_launch_manifest.json" "$LOCAL_OUTPUT/formal_launch_manifest.json"
cp "$WORK/${RUN_NAME}_formal_training_locks" "$LOCAL_OUTPUT/FORMAL_TRAINING_LOCKS"
cp "$WORK/${RUN_NAME}_formal_training_started" "$LOCAL_OUTPUT/FORMAL_TRAINING_STARTED"
write_state "fresh_id_training" "RUNNING"

resume_training() {
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -CONT "$TRAIN_PID" 2>/dev/null || true
  fi
}
trap resume_training EXIT INT TERM

for STEP in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000; do
  CKPT="$LOCAL_OUTPUT/ckpt-$STEP"
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
  mkdir -p "$LOCAL_OUTPUT/id_gate"
  GATE_OUT="$LOCAL_OUTPUT/id_gate/step-$STEP"
  if [[ ! -e "$GATE_OUT" ]]; then
    STACKPYRAMID_OOD_GEOMETRY=v4 PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}" "$PY" tools/evaluate_stackpyramid_xvla.py \
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
      --render-backend cpu > "$LOCAL_OUTPUT/id_gate/step-$STEP.log" 2>&1
  fi
  GATE_SUCCESS=$("$PY" - "$GATE_OUT/summary.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print(int(data.get("strict_success", 0)))
PY
)
  printf '{"step":%s,"episodes":20,"strict_successes":%s,"threshold":14}\n' "$STEP" "$GATE_SUCCESS" > "$GATE_OUT/decision.json"
  if (( GATE_SUCCESS >= 14 )); then
    printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$GATE_SUCCESS" > "$GATE_OUT/ID_GATE_PASSED"
  else
    printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$GATE_SUCCESS" > "$GATE_OUT/ID_GATE_NOT_PASSED"
  fi
  printf 'step=%s strict_successes=%s threshold=14\n' "$STEP" "$GATE_SUCCESS" > "$GATE_OUT/ID_GATE_STEP_COMPLETE"
  sync_tree "$CKPT"
  sync_tree "$GATE_OUT"
  rm -rf "$CKPT"
  kill -CONT "$TRAIN_PID"
  write_state "fresh_id_training" "RUNNING_AFTER_GATE_$STEP"
done

wait "$TRAIN_PID"
if [[ ! -f "$LOCAL_OUTPUT/TRAINING_COMPLETE" ]]; then
  echo "training exited without TRAINING_COMPLETE" >&2
  write_state "fresh_id_training" "FAILED_NO_COMPLETION_MARKER"
  exit 1
fi

cp "$LOCAL_OUTPUT/TRAINING_COMPLETE" "$PERSIST_OUTPUT/TRAINING_COMPLETE" 2>/dev/null || true
cp "$LOCAL_OUTPUT/training_config.json" "$PERSIST_OUTPUT/training_config.json" 2>/dev/null || true
cp "$LOCAL_OUTPUT/anchor_report.json" "$PERSIST_OUTPUT/anchor_report.json" 2>/dev/null || true
cp "$LOCAL_OUTPUT/train.jsonl" "$PERSIST_OUTPUT/train.jsonl" 2>/dev/null || true
FINAL_SUCCESS=$(cat "$LOCAL_OUTPUT/id_gate/step-10000/decision.json" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["strict_successes"])')
if (( FINAL_SUCCESS >= 14 )); then
  printf 'step=10000 strict_successes=%s threshold=14\n' "$FINAL_SUCCESS" > "$PERSIST_OUTPUT/ID_BASE_VALIDATED"
  write_state "id_base_validated" "PASSED"
else
  printf 'step=10000 strict_successes=%s threshold=14\n' "$FINAL_SUCCESS" > "$PERSIST_OUTPUT/ID_BASE_NOT_ACCEPTED"
  write_state "id_base_validated" "FAILED"
fi
