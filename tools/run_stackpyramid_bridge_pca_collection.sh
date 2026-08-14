#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zhaozhixuan/xvla_stackcube_data
OUT=$ROOT/stackpyramid_gated_dagger_v1/bridge_pca_collection_v1
LOG=$OUT/pipeline.log
PY=$ROOT/../envs/xvla_official_5090/bin/python
SCRIPT=$ROOT/tools/collect_stackpyramid_xvla_dagger.py
MODEL=$ROOT/stackpyramid_id_sft_10000_v1/formal_id_sft/ckpt-10000
XVLA=/data/zhaozhixuan/X-VLA
ASSET=$ROOT/stackpyramid_gated_dagger_v1/assets_retry1/bridge_pca.pt
THRESHOLD=0.8620001673698425
MAX_ATTEMPTS=${MAX_ATTEMPTS:-180}

mkdir -p "$OUT/logs"
exec >>"$LOG" 2>&1

state() {
  printf '%s\n' "$1" >"$OUT/pipeline_state.txt"
  printf '[%s] %s\n' "$(date -Is)" "$1"
}

run_stage() {
  local split="$1"
  local id_seed="$2"
  local ood_seed="$3"
  local stage_out="$OUT/$split"
  local stage_log="$OUT/logs/$split.log"
  if [[ -f "$stage_out/COLLECTION_COMPLETE" ]]; then
    state "${split}_already_complete"
    return 0
  fi
  if [[ -e "$stage_out" || -e "$stage_log" ]]; then
    echo "refusing to reuse partial stage: $split" >&2
    return 1
  fi
  state "${split}_running"
  set +e
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT:$XVLA" "$PY" "$SCRIPT" \
    --method bridge_pca \
    --checkpoint "$MODEL" \
    --xvla-root "$XVLA" \
    --asset "$ASSET" \
    --pca-threshold "$THRESHOLD" \
    --output-dir "$stage_out" \
    --split "$split" \
    --target 100 \
    --id-seed "$id_seed" \
    --ood-seed "$ood_seed" \
    --max-attempts "$MAX_ATTEMPTS" \
    --flow-steps 5 \
    --sim-backend cpu \
    --render-backend cpu \
    >"$stage_log" 2>&1
  local rc=$?
  set -e
  if [[ ! -f "$stage_out/COLLECTION_COMPLETE" || ! -f "$stage_out/summary.json" ]]; then
    state "${split}_failed_rc_${rc}"
    return 1
  fi
  "$PY" - "$stage_out" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
assert summary["accepted_total"] >= 100, summary
assert len(list((root / "raw_videos").glob("*.mp4"))) == summary["raw_attempts"], summary
print(json.dumps({k: summary[k] for k in ("split", "accepted_total", "raw_attempts", "raw_successes", "expert_action_steps")}))
PY
  state "${split}_passed_rc_${rc}"
}

state pending
run_stage stage1_ood 47000 48000
run_stage stage2_ood 49000 50000
run_stage stage3_ood 51000 52000
printf 'complete\n' >"$OUT/BRIDGE_PCA_COLLECTION_COMPLETE"
state complete
