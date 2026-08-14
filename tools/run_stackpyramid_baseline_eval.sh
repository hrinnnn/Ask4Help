#!/usr/bin/env bash
set -u

ROOT=/data/zhaozhixuan/xvla_stackcube_data
OUT=$ROOT/stackpyramid_baseline_eval_ckpt10000_v1
PY=/data/zhaozhixuan/envs/xvla_official_5090/bin/python
MODEL=$ROOT/stackpyramid_id_sft_10000_v1/formal_id_sft/ckpt-10000
SCRIPT=$ROOT/evaluate_stackpyramid_xvla.py
XVLA=/data/zhaozhixuan/X-VLA

mkdir -p "$OUT"
exec >>"$OUT/pipeline.log" 2>&1

write_state() {
  printf '%s\n' "$1" >"$OUT/pipeline_state.txt"
  printf '[%s] %s\n' "$(date -Is)" "$1"
}

wait_gpu() {
  while nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    write_state waiting_for_gpu0
    sleep 300
  done
}

run_split() {
  local split="$1"
  local seed="$2"
  local split_out="$OUT/$split"
  if [ -f "$split_out/EVAL_COMPLETE" ]; then
    write_state "${split}_already_complete"
    return 0
  fi
  if [ -e "$split_out" ]; then
    split_out="$OUT/${split}_retry1"
  fi
  wait_gpu
  write_state "${split}_running"
  set +e
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT:$XVLA" "$PY" "$SCRIPT" \
    --checkpoint "$MODEL" --xvla-root "$XVLA" --output "$split_out" \
    --split "$split" --episodes 100 --start-seed "$seed" \
    --max-episode-steps 250 --execute-horizon 5 --flow-steps 5 \
    --device cuda --sim-backend gpu --render-backend gpu \
    >"$OUT/${split}.log" 2>&1
  local rc=$?
  set -e
  if [ ! -f "$split_out/summary.json" ] || [ ! -f "$split_out/EVAL_COMPLETE" ]; then
    write_state "${split}_failed_rc_${rc}"
    return 1
  fi
  "$PY" - "$split_out" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
s=json.loads((p/'summary.json').read_text())
assert s['episodes'] == 100, s
assert s['video_count'] == 100, s
print(json.dumps({k:s[k] for k in ('split','episodes','ever_grasped','ever_base_completed','strict_success','video_count')}))
PY
  write_state "${split}_passed_rc_${rc}"
  return 0
}

write_state pending
run_split id 30000 || exit $?
run_split stage1_ood 31000 || exit $?
run_split stage2_ood 32000 || exit $?
run_split stage3_ood 33000 || exit $?

write_state complete
printf 'complete\n' >"$OUT/EVAL_PIPELINE_COMPLETE"
