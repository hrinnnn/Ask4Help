#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
TRAIN="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/recovery_training_50k_retry1/training"
BASE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000"
DURABLE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/horizon_600_20_id_890000_retry1"
WORK="/tmp/stackpyramid_horizon600_20_id_890000_retry1"
LOG="/root/ask4help_stage2_logs/stackpyramid_horizon600_20_id_890000_retry1.log"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORK" "$DURABLE"

declare -a NAMES=(baseline_ckpt40000 recovery_ckpt5000 recovery_ckpt10000 recovery_ckpt15000 recovery_ckpt20000)
declare -a MODELS=("$BASE" "$TRAIN/ckpt-5000" "$TRAIN/ckpt-10000" "$TRAIN/ckpt-15000" "$TRAIN/ckpt-20000")

for MODEL in "${MODELS[@]}"; do
  [[ -f "$MODEL/model.safetensors" ]] || { echo "missing $MODEL" >&2; exit 2; }
done

for INDEX in "${!NAMES[@]}"; do
  NAME="${NAMES[$INDEX]}"
  MODEL="${MODELS[$INDEX]}"
  LOCAL_OUT="$WORK/$NAME"
  DUR_OUT="$DURABLE/$NAME"
  if [[ -f "$DUR_OUT/HORIZON_600_DIAGNOSTIC_ONLY" ]]; then continue; fi
  rm -rf "$LOCAL_OUT" "$DUR_OUT"
  "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$MODEL" --xvla-root "$XVLA_ROOT" --output "$LOCAL_OUT" \
    --split id --episodes 20 --start-seed 890000 --max-episode-steps 600 \
    --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
    --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode \
    >> "$LOG" 2>&1
  [[ -f "$LOCAL_OUT/EVAL_COMPLETE" ]] || exit 3
  cp -a "$LOCAL_OUT" "$DURABLE/"
  "$PY" - "$DUR_OUT" "$LOCAL_OUT" <<'PY'
import json
import sys
from pathlib import Path

durable = Path(sys.argv[1])
local = sys.argv[2]
source = durable / "episodes.jsonl"
rows = []
for line in source.read_text().splitlines():
    if not line.strip(): continue
    row = json.loads(line)
    def rewrite(value):
        if isinstance(value, str) and value.startswith(local):
            return value.replace(local, str(durable), 1)
        if isinstance(value, dict): return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list): return [rewrite(item) for item in value]
        return value
    rows.append(rewrite(row))
tmp = Path("/tmp") / (durable.name + "_episodes_rewritten.jsonl")
tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
(durable / "episodes.jsonl").write_text(tmp.read_text())
tmp.unlink()
PY
  VIDEO_COUNT="$(find "$DUR_OUT/videos" -type f -printf '%f\n' | wc -l)"
  ACTION_COUNT="$(find "$DUR_OUT/actions" -type f -printf '%f\n' | wc -l)"
  STATE_COUNT="$(find "$DUR_OUT/states" -type f -printf '%f\n' | wc -l)"
  [[ "$VIDEO_COUNT" -eq 20 && "$ACTION_COUNT" -eq 20 && "$STATE_COUNT" -eq 20 ]] || exit 4
  "$PY" - "$DUR_OUT/summary.json" "$DUR_OUT/horizon600_metrics.json" "$NAME" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
rows = []
for row in summary.get("rows", []):
    rows.append({
        "episode_index": row.get("episode_index"),
        "seed": row.get("seed"),
        "steps": row.get("steps"),
        "timeout": bool(int(row.get("steps", 0)) >= 600 and not row.get("strict_success", False)),
        "strict_success": bool(row.get("strict_success", False)),
        "ever_grasped": bool(row.get("ever_grasped", False)),
        "ever_base_completed": bool(row.get("ever_base_completed", False)),
        "stage_events": row.get("stage_events", {}),
        "stage_event_steps": row.get("stage_event_steps", {}),
    })
report = {
    "format": "stackpyramid_horizon600_diagnostic_metrics_v1",
    "checkpoint": sys.argv[3],
    "episodes": len(rows),
    "strict_success": sum(row["strict_success"] for row in rows),
    "ever_grasped": sum(row["ever_grasped"] for row in rows),
    "ever_base_completed": sum(row["ever_base_completed"] for row in rows),
    "timeouts": sum(row["timeout"] for row in rows),
    "rows": rows,
    "diagnostic_only": True,
}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
PY
  printf '600-step timeout diagnostic; formal 300-step gate unchanged\n' > "$DUR_OUT/HORIZON_600_DIAGNOSTIC_ONLY"
  rm -rf "$LOCAL_OUT"
done

"$PY" - "$DURABLE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
reports = [json.loads(path.read_text()) for path in sorted(root.glob("*/horizon600_metrics.json"))]
if len(reports) != 5 or any(report["episodes"] != 20 for report in reports):
    raise SystemExit("incomplete horizon600 diagnostic")
summary = {
    "format": "stackpyramid_horizon600_diagnostic_v1",
    "seed_manifest": "890000--890019",
    "episodes_per_checkpoint": 20,
    "geometry": "v4",
    "fresh_env_per_episode": True,
    "max_episode_steps": 600,
    "execute_horizon": 5,
    "reports": reports,
    "diagnostic_only": True,
    "formal_300_step_gate_unchanged": True,
    "ood_started": False,
}
(root / "horizon600_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
(root / "HORIZON_600_COMPLETE").write_text("complete; diagnostic only\n")
PY
