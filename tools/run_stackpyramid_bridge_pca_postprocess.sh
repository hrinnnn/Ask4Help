#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zhaozhixuan/xvla_stackcube_data
COLLECT=${COLLECT:-$ROOT/stackpyramid_gated_dagger_v1/bridge_pca_collection_v3}
TRAIN=${TRAIN:-$ROOT/stackpyramid_gated_dagger_v1/bridge_pca_training_v1}
OUT=${OUT:-$ROOT/stackpyramid_gated_dagger_v1/bridge_pca_postprocess_v1}
PY=$ROOT/../envs/xvla_official_5090/bin/python
XVLA=/data/zhaozhixuan/X-VLA
ID_H5=$ROOT/stackpyramid_formal_collection_v2/id/Ask4HelpStackPyramidID-v1/motionplanning/oracle_id.h5
EVAL=$ROOT/tools/evaluate_stackpyramid_xvla.py
TIMING=$ROOT/tools/summarize_stackpyramid_timing.py

mkdir -p "$OUT/logs" "$OUT/timing" "$OUT/eval"
exec >>"$OUT/pipeline.log" 2>&1

state() {
  printf '%s\n' "$1" >"$OUT/pipeline_state.txt"
  printf '[%s] %s\n' "$(date -Is)" "$1"
}

state waiting_for_training
while [[ ! -f "$TRAIN/BRIDGE_PCA_TRAINING_COMPLETE" ]]; do
  current=$(cat "$TRAIN/pipeline_state.txt" 2>/dev/null || true)
  if [[ "$current" == *_failed* || "$current" == *_missing* ]]; then
    state "training_failed_${current}"
    exit 1
  fi
  sleep 180
done

for split in stage1_ood stage2_ood stage3_ood; do
  state "${split}_timing"
  if [[ ! -f "$OUT/timing/$split/summary.json" ]]; then
    "$PY" "$TIMING" \
      --nominal-h5 "$ID_H5" \
      --expert-h5 "$COLLECT/$split/accepted_suffixes.h5" \
      --output "$OUT/timing/$split" \
      --method bridge_pca \
      --split "$split"
  fi
done

run_eval() {
  local stage="$1"
  local split="$2"
  local seed="$3"
  local checkpoint="$TRAIN/$stage/ckpt-2000"
  local output="$OUT/eval/${stage}_${split}"
  if [[ -f "$output/EVAL_COMPLETE" ]]; then
    return 0
  fi
  if [[ -e "$output" ]]; then
    state "${stage}_${split}_partial_requires_new_retry"
    return 1
  fi
  state "${stage}_${split}_running"
  set +e
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT:$XVLA" "$PY" "$EVAL" \
    --checkpoint "$checkpoint" \
    --xvla-root "$XVLA" \
    --output "$output" \
    --split "$split" \
    --episodes 100 \
    --start-seed "$seed" \
    --max-episode-steps 250 \
    --execute-horizon 5 \
    --flow-steps 5 \
    --device cuda \
    --sim-backend cpu \
    --render-backend cpu \
    >"$OUT/logs/${stage}_${split}.log" 2>&1
  local rc=$?
  set -e
  "$PY" - "$output" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
assert summary["episodes"] == 100, summary
assert summary["video_count"] == 100, summary
print(json.dumps({k: summary[k] for k in ("split", "episodes", "ever_grasped", "ever_base_completed", "strict_success", "video_count")}))
PY
  state "${stage}_${split}_passed_rc_${rc}"
}

run_eval stage1_ood id 54000
run_eval stage1_ood stage1_ood 55000
run_eval stage2_ood id 56000
run_eval stage2_ood stage2_ood 57000
run_eval stage3_ood id 58000
run_eval stage3_ood stage3_ood 59000

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted((root / "eval").glob("*/summary.json")):
    summary = json.loads(path.read_text())
    rows.append({"condition": path.parent.name, **{key: summary[key] for key in ("episodes", "ever_grasped", "ever_base_completed", "strict_success", "video_count")}})
(root / "comparison.json").write_text(json.dumps({"format": "stackpyramid_bridge_pca_postprocess_v1", "rows": rows}, indent=2) + "\n")
(root / "comparison.md").write_text("# StackPyramid Bridge-PCA postprocess\n\n" + "\n".join(f"- {row['condition']}: grasp={row['ever_grasped']}/100, strict={row['strict_success']}/100" for row in rows) + "\n")
PY
printf 'complete\n' >"$OUT/PIPELINE_COMPLETE"
state complete
