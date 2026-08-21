#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
DURABLE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/ood_plan_v1/oracle_smoke_stage2_stage3_retry1"
WORK="/tmp/stackpyramid_ood_oracle_smoke_stage2_stage3_retry1"
LOG="/root/ask4help_stage2_logs/stackpyramid_ood_oracle_smoke_stage2_stage3_retry1.log"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORK" "$DURABLE"

run_condition() {
  local name="$1" split="$2" seed="$3"
  local local_out="$WORK/$name" durable_out="$DURABLE/$name"
  rm -rf "$local_out" "$durable_out"
  "$PY" "$ROOT/tools/run_stackpyramid_oracle_gate.py" \
    --repo-root "$ROOT" --xvla-root "$XVLA_ROOT" --output "$local_out" \
    --split "$split" --episodes 20 --start-seed "$seed" \
    --sim-backend gpu --render-backend gpu --max-episode-steps 600 \
    --fresh-env-per-episode --formal-evidence >> "$LOG" 2>&1
  cp -a "$local_out" "$DURABLE/"
  "$PY" - "$durable_out" "$local_out" <<'PY'
import json
import sys
from pathlib import Path
durable, local = Path(sys.argv[1]), sys.argv[2]
for filename in ("summary.json", "episodes.jsonl"):
    path = durable / filename
    if not path.exists(): continue
    if filename.endswith(".jsonl"):
        lines = path.read_text().splitlines()
        values = [json.loads(line) for line in lines if line.strip()]
        content = "".join(json.dumps(value).replace(local, str(durable)) + "\n" for value in values)
        path.write_text(content)
    else:
        value = json.loads(path.read_text())
        path.write_text(json.dumps(value, indent=2).replace(local, str(durable)) + "\n")
PY
  if [[ ! -f "$durable_out/ORACLE_GATE_COMPLETE" ]]; then
    printf 'Oracle smoke failed; preserve complete diagnostic output\n' > "$durable_out/ORACLE_SMOKE_FAILED"
    exit 1
  fi
  [[ "$(find "$durable_out/videos" -type f -printf '%f\n' | wc -l)" -eq 20 ]] || exit 2
  [[ "$(find "$durable_out/actions" -type f -printf '%f\n' | wc -l)" -eq 20 ]] || exit 2
  [[ "$(find "$durable_out/states" -type f -printf '%f\n' | wc -l)" -eq 20 ]] || exit 2
  rm -rf "$local_out"
}

run_condition stage2_id id 892000
run_condition stage2_ood stage2_ood 892000
run_condition stage3_id id 893000
run_condition stage3_ood stage3_ood 893000

printf 'complete; Stage2 and Stage3 Oracle smoke passed\n' > "$DURABLE/ORACLE_SMOKE_COMPLETE"
