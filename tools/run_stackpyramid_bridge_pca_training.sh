#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/zhaozhixuan/xvla_stackcube_data
COLLECT=${COLLECT:-$ROOT/stackpyramid_gated_dagger_v1/bridge_pca_collection_v3}
OUT=${OUT:-$ROOT/stackpyramid_gated_dagger_v1/bridge_pca_training_v1}
PY=$ROOT/../envs/xvla_official_5090/bin/python
SCRIPT=$ROOT/tools/run_stackpyramid_gated_training.py
ID_H5=${ID_H5:-$ROOT/stackpyramid_formal_collection_v2/id/Ask4HelpStackPyramidID-v1/motionplanning/oracle_id.h5}
MODEL=$ROOT/stackpyramid_id_sft_10000_v1/formal_id_sft/ckpt-10000
XVLA=/data/zhaozhixuan/X-VLA
STEPS=${STEPS:-2000}
COLLECTION_MARKER=$COLLECT/BRIDGE_PCA_COLLECTION_COMPLETE

mkdir -p "$OUT/logs"
exec >>"$OUT/pipeline.log" 2>&1

state() {
  printf '%s\n' "$1" >"$OUT/pipeline_state.txt"
  printf '[%s] %s\n' "$(date -Is)" "$1"
}

state waiting_for_collection
while [[ ! -f "$COLLECTION_MARKER" ]]; do
  current=$(cat "$COLLECT/pipeline_state.txt" 2>/dev/null || true)
  if [[ "$current" == *_failed* || "$current" == *_missing* ]]; then
    state "collection_failed_${current}"
    exit 1
  fi
  sleep 120
done

run_stage() {
  local split="$1"
  local stage_out="$OUT/$split"
  if [[ -f "$stage_out/TRAINING_COMPLETE" ]]; then
    state "${split}_already_complete"
    return 0
  fi
  if [[ -e "$stage_out" ]]; then
    state "${split}_partial_requires_new_retry"
    return 1
  fi
  local expert_h5="$COLLECT/$split/accepted_suffixes.h5"
  if [[ ! -f "$expert_h5" ]]; then
    state "${split}_missing_expert_h5"
    return 1
  fi
  state "${split}_running"
  set +e
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT:$XVLA" "$PY" "$SCRIPT" \
    --xvla-root "$XVLA" \
    --model "$MODEL" \
    --id-h5 "$ID_H5" \
    --expert-h5 "$expert_h5" \
    --output "$stage_out" \
    --steps "$STEPS" \
    --save-interval 500 \
    --batch-size 8 \
    --seed "$((8200 + ${#split}))" \
    >"$OUT/logs/$split.log" 2>&1
  local rc=$?
  set -e
  if [[ ! -f "$stage_out/TRAINING_COMPLETE" ]]; then
    state "${split}_failed_rc_${rc}"
    return 1
  fi
  for step in 500 1000 1500 2000; do
    [[ -f "$stage_out/ckpt-$step/model.safetensors" ]] || { state "${split}_missing_ckpt_${step}"; return 1; }
  done
  state "${split}_passed_rc_${rc}"
}

run_smoke() {
  local split="$1"
  local smoke_out="$OUT/smoke/$split"
  if [[ -f "$smoke_out/RELOAD_SMOKE_COMPLETE" ]]; then
    state "${split}_smoke_already_complete"
    return 0
  fi
  if [[ -e "$smoke_out" ]]; then
    state "${split}_smoke_partial_requires_new_retry"
    return 1
  fi
  local expert_h5="$COLLECT/$split/accepted_suffixes.h5"
  [[ -f "$expert_h5" ]] || { state "${split}_smoke_missing_expert_h5"; return 1; }
  state "${split}_smoke_running"
  set +e
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT:$XVLA" "$PY" "$SCRIPT" \
    --xvla-root "$XVLA" --model "$MODEL" --id-h5 "$ID_H5" --expert-h5 "$expert_h5" \
    --output "$smoke_out" --steps 2 --save-interval 2 --batch-size 8 --seed "$((9100 + ${#split}))" --smoke-only \
    >"$OUT/logs/${split}_smoke.log" 2>&1
  local rc=$?
  set -e
  if [[ ! -f "$smoke_out/RELOAD_SMOKE_COMPLETE" ]]; then
    state "${split}_smoke_failed_rc_${rc}"
    return 1
  fi
  state "${split}_smoke_passed_rc_${rc}"
}

state pending
run_smoke stage1_ood
run_smoke stage2_ood
run_smoke stage3_ood
state formal_training_pending
run_stage stage1_ood
run_stage stage2_ood
run_stage stage3_ood
printf 'complete\n' >"$OUT/BRIDGE_PCA_TRAINING_COMPLETE"
state complete
