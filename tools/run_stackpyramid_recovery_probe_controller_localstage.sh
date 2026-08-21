#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
TRAIN="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/recovery_training_50k_retry1/training"
PROBE_DURABLE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1/early_probe_20_id_888000_retry2"
WORK="/tmp/stackpyramid_recovery_probe_20_id_888000_retry2"
LOG="/root/ask4help_stage2_logs/stackpyramid_recovery_probe_20_id_888000_retry2.log"

export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORK" "$PROBE_DURABLE"

for STEP in 5000 10000 15000 20000; do
  [[ -f "$TRAIN/ckpt-$STEP/model.safetensors" ]] || {
    printf 'missing checkpoint ckpt-%s\n' "$STEP" >&2
    exit 2
  }
done

for STEP in 5000 10000 15000 20000; do
  LOCAL_OUT="$WORK/step-$STEP"
  DUR_OUT="$PROBE_DURABLE/step-$STEP"
  if [[ -f "$DUR_OUT/PROBE_DIAGNOSTIC_ONLY" ]]; then
    continue
  fi
  rm -rf "$LOCAL_OUT" "$DUR_OUT"
  "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$TRAIN/ckpt-$STEP" --xvla-root "$XVLA_ROOT" --output "$LOCAL_OUT" \
    --split id --episodes 20 --start-seed 888000 --max-episode-steps 300 \
    --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
    --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode \
    >> "$LOG" 2>&1
  [[ -f "$LOCAL_OUT/EVAL_COMPLETE" ]] || exit 3
  cp -a "$LOCAL_OUT" "$PROBE_DURABLE/"
  "$PY" - "$DUR_OUT" "$LOCAL_OUT" <<'PY'
import json
import sys
from pathlib import Path

durable = Path(sys.argv[1])
local = sys.argv[2]
episodes = durable / "episodes.jsonl"
rows = []
for line in episodes.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)

    def rewrite(value):
        if isinstance(value, str) and value.startswith(local):
            return value.replace(local, str(durable), 1)
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        return value

    rows.append(rewrite(row))
tmp = Path("/tmp") / (durable.name + "_episodes_rewritten.jsonl")
tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
(durable / "episodes.jsonl").write_text(tmp.read_text())
tmp.unlink()
PY
  VIDEO_COUNT="$(find "$DUR_OUT/videos" -type f -name '*.mp4' | wc -l)"
  ACTION_COUNT="$(find "$DUR_OUT/actions" -type f -name '*.npy' | wc -l)"
  STATE_COUNT="$(find "$DUR_OUT/states" -type f -name '*.json' | wc -l)"
  [[ "$VIDEO_COUNT" -eq 20 && "$ACTION_COUNT" -eq 20 && "$STATE_COUNT" -eq 20 ]] || exit 4
  "$PY" - "$DUR_OUT/summary.json" "$DUR_OUT/probe_metrics.json" "$STEP" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
row = {
    "step": int(sys.argv[3]),
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
Path(sys.argv[2]).write_text(json.dumps(row, indent=2) + "\n")
PY
  printf 'local-staged video/actions/state evidence; diagnostic only\n' > "$DUR_OUT/PROBE_DIAGNOSTIC_ONLY"
  rm -rf "$LOCAL_OUT"
done

"$PY" - "$PROBE_DURABLE" <<'PY'
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
    "storage_mode": "local_staging_then_sync",
    "ood_started": False,
}
(root / "probe_summary.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "PROBE_COMPLETE").write_text("complete; diagnostic only\n")
PY
