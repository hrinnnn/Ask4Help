#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
TRAIN="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/recovery_training_50k_retry1/training"
PROBE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/early_probe_20_id_888000_retry1"
LOG="/root/ask4help_stage2_logs/stackpyramid_recovery_probe_20_id_888000_retry1.log"
PID_FILE="/root/ask4help_stage2_logs/stackpyramid_recovery_probe_20_id_888000_retry1.pid"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$PROBE"

for STEP in 5000 10000 15000 20000; do
  [[ -f "$TRAIN/ckpt-$STEP/model.safetensors" ]] || {
    printf 'missing checkpoint ckpt-%s\n' "$STEP" >&2
    exit 2
  }
done

for STEP in 5000 10000 15000 20000; do
  OUT="$PROBE/step-$STEP"
  if [[ -f "$OUT/EVAL_COMPLETE" ]]; then
    continue
  fi
  "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$TRAIN/ckpt-$STEP" --xvla-root "$XVLA_ROOT" --output "$OUT" \
    --split id --episodes 20 --start-seed 888000 --max-episode-steps 300 \
    --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
    --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode \
    >> "$LOG" 2>&1
  "$PY" - "$OUT/summary.json" "$OUT" "$STEP" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
step = int(sys.argv[3])
row = {
    "step": step,
    "episodes": int(summary.get("episodes", 0)),
    "strict_success": int(summary.get("strict_success", 0)),
    "ever_grasped": int(summary.get("ever_grasped", 0)),
    "ever_base_completed": int(summary.get("ever_base_completed", 0)),
    "red_grasped": int(summary.get("stage_event_counts", {}).get("red_grasped", 0)),
    "red_placed": int(summary.get("stage_event_counts", {}).get("red_placed", 0)),
    "blue_lifted": int(summary.get("stage_event_counts", {}).get("blue_lifted", 0)),
    "videos": int(summary.get("video_count", 0)),
    "actions": int(summary.get("action_array_count", 0)),
    "states": int(summary.get("state_timeline_count", 0)),
}
if row["episodes"] != 20 or row["videos"] != 20 or row["actions"] != 20 or row["states"] != 20:
    raise SystemExit(f"incomplete probe evidence: {row}")
(out / "probe_metrics.json").write_text(json.dumps(row, indent=2) + "\n")
(out / "PROBE_DIAGNOSTIC_ONLY").write_text("trend probe only; not a formal ID gate\n")
PY
done

"$PY" - "$PROBE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = [json.loads(path.read_text()) for path in sorted(root.glob("step-*/probe_metrics.json"), key=lambda p: int(p.parent.name.split("-")[1]))]
if [row["step"] for row in rows] != [5000, 10000, 15000, 20000]:
    raise SystemExit("missing one or more probe checkpoints")
report = {
    "format": "stackpyramid_recovery_early_probe_v1",
    "checkpoint_steps": [5000, 10000, 15000, 20000],
    "seed_manifest": "888000--888019",
    "episodes_per_checkpoint": 20,
    "geometry": "v4",
    "fresh_env_per_episode": True,
    "max_episode_steps": 300,
    "execute_horizon": 5,
    "rows": rows,
    "diagnostic_only": True,
    "formal_selection_unchanged": True,
    "ood_started": False,
}
(root / "probe_summary.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "PROBE_COMPLETE").write_text("complete; diagnostic only\n")
PY
