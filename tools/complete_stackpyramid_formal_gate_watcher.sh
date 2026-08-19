#!/usr/bin/env bash
set -euo pipefail

EVAL_PID="${EVAL_PID:?set EVAL_PID to the single formal evaluator}"
LAUNCHER_PID="${LAUNCHER_PID:?set LAUNCHER_PID to the old launcher shell}"
ROOT="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1/formal_id_gate_100_retry3"
PERSIST="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v1/formal_id_gate_100_retry3"
PY="/root/.venvs/xvla-h20/bin/python"
REPO="/root/Ask4Help-xvla-stackpyramid-v4"

while kill -0 "$EVAL_PID" 2>/dev/null; do
  sleep 30
done
sleep 5

# The original launcher has known quoting errors in its post-evaluation shell.
# Stop only that shell after its evaluator exits; preserve its log as diagnostic.
if kill -0 "$LAUNCHER_PID" 2>/dev/null; then
  kill "$LAUNCHER_PID" 2>/dev/null || true
fi

if [[ ! -f "$ROOT/summary.json" || ! -f "$ROOT/EVAL_COMPLETE" ]]; then
  printf '%s\n' "Formal evaluator exited without a complete summary/EVAL_COMPLETE." > "$ROOT/FORMAL_GATE_FAILED_INCOMPLETE"
  exit 1
fi

set +e
"$PY" "$REPO/tools/audit_stackpyramid_formal_id_gate.py" \
  --root "$ROOT" --expected 100 --minimum-successes 80 \
  --output "$ROOT/formal_audit.json"
AUDIT_RC=$?
set -e
if [[ ! -f "$ROOT/formal_audit.json" ]]; then
  printf '%s\n' "formal audit command failed with rc=$AUDIT_RC and produced no report." > "$ROOT/FORMAL_GATE_FAILED_AUDIT_NO_REPORT"
  exit 1
fi

AUDIT_PASS=$("$PY" - "$ROOT/formal_audit.json" <<'PY'
import json, sys
print("1" if json.load(open(sys.argv[1]))["audit_pass"] else "0")
PY
)
STRICT_SUCCESS=$("$PY" - "$ROOT/summary.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("strict_success", 0))
PY
)
if [[ "$AUDIT_PASS" == "1" ]]; then
  STATUS="FORMAL_ID_GATE_PASSED"
  MARKER="ID_BASE_VALIDATED"
else
  STATUS="FORMAL_ID_GATE_FAILED"
  MARKER="ID_BASE_NOT_ACCEPTED_FORMAL_100"
fi

cat > "$ROOT/formal_reconciliation.json" <<EOF
{
  "status": "$STATUS",
  "strict_success": $STRICT_SUCCESS,
  "episodes": 100,
  "videos": 100,
  "actions": 100,
  "states": 100,
  "geometry": "v4",
  "env_id": "Ask4HelpStackPyramidID-v4",
  "seed_start": 84400,
  "seed_end_exclusive": 84500,
  "old_launcher_error": "post-evaluation shell quoting was bypassed by this watcher",
  "ood_unlocked": false
}
EOF
printf '%s\n' "$STATUS" > "$ROOT/$STATUS"
printf '%s\n' "$MARKER" > "$ROOT/$MARKER"
printf '%s\n' "formal_id_gate_100_retry3=$STATUS strict_success=$STRICT_SUCCESS" > "$ROOT/COMPLETION_WATCHER_COMPLETE"

mkdir -p "$PERSIST"
cp -a "$ROOT"/. "$PERSIST"/
