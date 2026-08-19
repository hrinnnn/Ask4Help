#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
BASE="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1/id_sft_10000_retry4/ckpt-10000"
DATA="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v3/id_training_collection_512_external_links_retry1"
ROOT_OUT="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1"
TRAIN_OUT="$ROOT_OUT/training"
FORMAL_OUT="$ROOT_OUT/formal_id_gate_100"
LOG="/root/ask4help_stage2_logs/stackpyramid_continuation_50k_retry1.log"
PID_FILE="/root/ask4help_stage2_logs/stackpyramid_continuation_50k_retry1.pid"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
export STACKPYRAMID_OOD_GEOMETRY=v4

test -f "$ROOT_OUT/CUSTOM_ADAPTER_ACCEPTED_FOR_CONTINUATION"
test -f "$ROOT_OUT/CONTINUATION_CONFIG_RECONCILED"
test -f "$BASE/config.json"
test -f "$DATA/id/accepted_suffixes.h5"
if [[ -e "$TRAIN_OUT/TRAINING_COMPLETE" || -e "$ROOT_OUT/CONTINUATION_FAILED" ]]; then
  echo "continuation already has terminal marker" >&2
  exit 2
fi
mkdir -p "$ROOT_OUT" "/root/ask4help_stage2_logs"

if [[ ! -f "$TRAIN_OUT/TRAINING_COMPLETE" ]]; then
  nohup "$PY" "$ROOT/tools/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA_ROOT" --model "$BASE" --collection-root "$DATA" \
    --split id --target-episodes 512 --output "$TRAIN_OUT" \
    --steps 40000 --save-interval 5000 --batch-size 8 \
    --learning-rate 1e-4 --learning-coef 0.1 --weight-decay 0.0 \
    --freeze-steps 0 --warmup-steps 2000 --log-interval 20 \
    --seed 886300 --dtype bf16 > "$LOG" 2>&1 &
  TRAIN_PID=$!
  printf '%s\n' "$TRAIN_PID" > "$PID_FILE"
  printf 'pid=%s\nbase=%s\ndata=%s\noutput=%s\n' "$TRAIN_PID" "$BASE" "$DATA" "$TRAIN_OUT" > "$ROOT_OUT/TRAINING_STARTED"
else
  TRAIN_PID=0
fi

if [[ "$TRAIN_PID" != 0 ]]; then
  while [[ ! -f "$TRAIN_OUT/TRAINING_COMPLETE" ]]; do
    kill -0 "$TRAIN_PID" 2>/dev/null || { printf 'training process exited before completion\n' > "$ROOT_OUT/CONTINUATION_FAILED"; exit 10; }
    sleep 60
  done
fi
test -d "$TRAIN_OUT/ckpt-40000"

if [[ ! -f "$FORMAL_OUT/EVAL_COMPLETE" ]]; then
  "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$TRAIN_OUT/ckpt-40000" --xvla-root "$XVLA_ROOT" \
    --output "$FORMAL_OUT" --split id --episodes 100 --start-seed 84400 \
    --max-episode-steps 300 --execute-horizon 5 --flow-steps 5 \
    --device cuda --sim-backend gpu --render-backend gpu \
    --formal-evidence --geometry v4 --fresh-env-per-episode
  "$PY" "$ROOT/tools/audit_stackpyramid_formal_id_gate.py" \
    --root "$FORMAL_OUT" --expected 100 --minimum-successes 80 \
    --output "$FORMAL_OUT/formal_audit.json" || true
fi

if "$PY" - "$FORMAL_OUT/formal_audit.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["audit_pass"] else 1)
PY
then
  printf 'formal ID gate passed at total_steps=50000\n' > "$ROOT_OUT/ID_BASE_VALIDATED_50K"
  printf 'continuation complete; OOD remains locked pending review\n' > "$ROOT_OUT/CONTINUATION_COMPLETE"
else
  printf 'formal ID gate failed at total_steps=50000\n' > "$ROOT_OUT/ID_BASE_NOT_ACCEPTED_50K"
  printf 'continuation complete with failed ID gate; OOD remains locked\n' > "$ROOT_OUT/CONTINUATION_COMPLETE"
fi
