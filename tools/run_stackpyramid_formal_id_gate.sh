#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/Ask4Help-xvla-stackpyramid-v4"
PY="/root/.venvs/xvla-h20/bin/python"
XVLA_ROOT="/root/X-VLA"
WORK="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1"
CHECKPOINT="$WORK/id_sft_10000_retry4/ckpt-10000"
TRAINING_COMPLETE="$WORK/id_sft_10000_retry4/TRAINING_COMPLETE"
LOCAL="$WORK/formal_id_gate_100_retry1"
PERSIST="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v1/formal_id_gate_100_retry1"

if [[ ! -f "$TRAINING_COMPLETE" || ! -d "$CHECKPOINT" ]]; then
  echo "missing completed training or ckpt-10000" >&2
  exit 2
fi
if [[ -e "$LOCAL" || -e "$PERSIST" ]]; then
  echo "refusing to overwrite formal gate output" >&2
  exit 2
fi
rm -rf "$LOCAL"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"

"$PY" tools/evaluate_stackpyramid_xvla.py \
  --checkpoint "$CHECKPOINT" --xvla-root "$XVLA_ROOT" --output "$LOCAL" \
  --split id --episodes 100 --start-seed 888000 \
  --max-episode-steps 300 --execute-horizon 5 --flow-steps 5 \
  --device cuda --sim-backend gpu --render-backend gpu --formal-evidence \
  --geometry v4

"$PY" tools/audit_stackpyramid_formal_id_gate.py \
  --root "$LOCAL" --expected 100 --minimum-successes 80 \
  --output "$LOCAL/formal_audit.json"

mkdir -p "$PERSIST"
cp -a "$LOCAL"/. "$PERSIST"/
printf '{"stage":"formal_id_gate_100","status":"AUDIT_COMPLETE","seed_start":888000,"episodes":100,"strict_threshold":80}\n' > "$LOCAL/pipeline_state.json"
cp "$LOCAL/pipeline_state.json" "$PERSIST/pipeline_state.json" 2>/dev/null || true
if "$PY" - "$LOCAL/formal_audit.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["audit_pass"] else 1)
PY
then
  printf 'formal_id_gate=100/100 evidence_passed strict_threshold=80\n' > "$PERSIST/ID_BASE_VALIDATED"
  printf 'formal_id_gate_100=PASSED\n' > "$PERSIST/FORMAL_ID_GATE_PASSED"
else
  printf 'formal_id_gate failed; OOD remains locked\n' > "$PERSIST/ID_BASE_NOT_ACCEPTED"
  printf 'formal_id_gate_100=FAILED\n' > "$PERSIST/FORMAL_ID_GATE_FAILED"
  exit 1
fi
