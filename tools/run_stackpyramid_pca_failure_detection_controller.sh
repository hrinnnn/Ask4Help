#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4-512}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
CHECKPOINT="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/continuation_50k_from_ckpt10000_lr1e-4_retry1/training/ckpt-40000"
ID_H5="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v3/id_training_collection_512_external_links_retry1/id/accepted_suffixes.h5"
PIPE="/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v3/grasp_recovery_v1"
OUT="$PIPE/failure_detection_pca_v1"
WORK="/tmp/stackpyramid_failure_detection_pca_v1"
ASSET_LOCAL="$WORK/bridge_pca.pt"
ASSET_DURABLE="$OUT/bridge_pca.pt"
CAL_LOCAL="$WORK/calibration.json"
CAL_DURABLE="$OUT/calibration.json"
LOG="/root/ask4help_stage2_logs/stackpyramid_failure_detection_pca_v1.log"
STATE_LOCAL="$WORK/pipeline_state.txt"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export STACKPYRAMID_OOD_GEOMETRY=v4
export PYTHONPATH="$ROOT:$ROOT/RLinf:$XVLA_ROOT:${PYTHONPATH:-}"
mkdir -p "$WORK" "$OUT"

state() {
  printf '%s\n' "$1" > "$STATE_LOCAL"
}

[[ -d "$CHECKPOINT" ]] || exit 2
[[ -f "$ID_H5" ]] || exit 2

if [[ ! -f "$ASSET_DURABLE" ]]; then
  state building_id_only_pca_asset
  rm -f "$ASSET_LOCAL"
  "$PY" "$ROOT/tools/build_stackpyramid_bridge_pca.py" \
    --checkpoint "$CHECKPOINT" --xvla-root "$XVLA_ROOT" \
    --collection-root /root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v3/id_training_collection_512_external_links_retry1 \
    --output "$ASSET_LOCAL" --target-episodes 512 >> "$LOG" 2>&1
  cp "$ASSET_LOCAL" "$ASSET_DURABLE"
else
  [[ -f "$ASSET_LOCAL" ]] || cp "$ASSET_DURABLE" "$ASSET_LOCAL"
fi

if [[ ! -f "$CAL_DURABLE" ]]; then
  state calibrating_successful_id_threshold
  rm -f "$CAL_LOCAL"
  "$PY" "$ROOT/tools/calibrate_stackpyramid_bridge_pca.py" \
    --checkpoint "$CHECKPOINT" --xvla-root "$XVLA_ROOT" \
    --asset "$ASSET_LOCAL" --output "$CAL_LOCAL" \
    --successful-rollouts 25 --max-attempts 60 --start-seed 894000 \
    --flow-steps 5 --max-episode-steps 600 --geometry v4 \
    --fresh-env-per-episode --sim-backend gpu --render-backend gpu >> "$LOG" 2>&1
  cp "$CAL_LOCAL" "$CAL_DURABLE"
else
  [[ -f "$CAL_LOCAL" ]] || cp "$CAL_DURABLE" "$CAL_LOCAL"
fi

THRESHOLD="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' "$CAL_LOCAL")"

run_condition() {
  local name="$1" split="$2" seed="$3"
  local local_out="$WORK/$name" durable_out="$OUT/$name"
  if [[ -f "$durable_out/EVAL_COMPLETE" ]]; then return 0; fi
  rm -rf "$local_out" "$durable_out"
  state "${name}_running"
  "$PY" "$ROOT/tools/evaluate_stackpyramid_pca_failure_detection.py" \
    --checkpoint "$CHECKPOINT" --xvla-root "$XVLA_ROOT" \
    --asset "$ASSET_LOCAL" --threshold "$THRESHOLD" --output "$local_out" \
    --split "$split" --episodes 100 --start-seed "$seed" \
    --max-episode-steps 600 --execute-horizon 5 --flow-steps 5 \
    --device cuda --sim-backend gpu --render-backend gpu >> "$LOG" 2>&1
  [[ -f "$local_out/EVAL_COMPLETE" ]] || exit 3
  cp -a "$local_out" "$OUT/"
  "$PY" - "$durable_out" "$local_out" <<'PY'
import json
import sys
from pathlib import Path
durable, local = Path(sys.argv[1]), sys.argv[2]
rows = []
for line in (durable / "episodes.jsonl").read_text().splitlines():
    if not line.strip(): continue
    row = json.loads(line)
    def rewrite(value):
        if isinstance(value, str) and value.startswith(local): return value.replace(local, str(durable), 1)
        if isinstance(value, dict): return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list): return [rewrite(item) for item in value]
        return value
    rows.append(rewrite(row))
tmp = Path("/tmp") / (durable.name + "_episodes_rewritten.jsonl")
tmp.write_text("".join(json.dumps(row) + "\n" for row in rows))
(durable / "episodes.jsonl").write_text(tmp.read_text())
tmp.unlink()
PY
  [[ "$(find "$durable_out/videos" -type f -printf '%f\n' | wc -l)" -eq 100 ]] || exit 4
  [[ "$(find "$durable_out/actions" -type f -printf '%f\n' | wc -l)" -eq 100 ]] || exit 4
  [[ "$(find "$durable_out/states" -type f -printf '%f\n' | wc -l)" -eq 100 ]] || exit 4
  rm -rf "$local_out"
}

run_condition stage2_id id 892100
run_condition stage2_ood stage2_ood 892100
run_condition stage3_id id 893100
run_condition stage3_ood stage3_ood 893100

state summarizing_passive_failure_detection
"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

root = Path(sys.argv[1])
threshold = float(json.loads((root / "calibration.json").read_text())["threshold"])
conditions = {}
for path in sorted(root.glob("stage*_*/summary.json")):
    conditions[path.parent.name] = json.loads(path.read_text())

def metrics(rows):
    scores = [float(row["max_pca_score"]) for row in rows]
    labels = [int(row["failure_label"]) for row in rows]
    pred = [int(score > threshold) for score in scores]
    result = {"episodes": len(rows), "failures": sum(labels), "alarms": sum(pred), "threshold": threshold}
    if len(set(labels)) > 1:
        result.update({
            "auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        })
    else:
        result.update({"auroc": None, "auprc": None, "balanced_accuracy": None})
    result["successful_id_false_alarm_rate"] = None
    return result

report = {"format": "stackpyramid_passive_pca_failure_detection_report_v1", "checkpoint": str(root), "threshold": threshold, "conditions": {}, "stages": {}}
for name, summary in conditions.items():
    report["conditions"][name] = metrics(summary["rows"])
for stage in ("stage2", "stage3"):
    id_rows = conditions[stage + "_id"]["rows"]
    ood_rows = conditions[stage + "_ood"]["rows"]
    report["stages"][stage] = metrics(id_rows + ood_rows)
(root / "failure_detection_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
(root / "PASSIVE_FAILURE_DETECTION_ONLY").write_text("no expert intervention; no training; no PCA collection\n")
(root / "FAILURE_DETECTION_COMPLETE").write_text("complete\n")
Path("/tmp/stackpyramid_failure_detection_pca_v1_complete_state").write_text("complete\n")
PY
cp /tmp/stackpyramid_failure_detection_pca_v1_complete_state "$OUT/pipeline_state.txt" 2>/dev/null || true
