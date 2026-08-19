#!/usr/bin/env bash
set -euo pipefail

# Restart-tolerant controller for the 512-ID StackPyramid recovery.
# It never starts OOD, PCA, timing, or downstream DAgger work.

ROOT="${ROOT:-/root/Ask4Help-xvla-stackpyramid-v4}"
PY="${PY:-/root/.venvs/xvla-h20/bin/python}"
XVLA_ROOT="${XVLA_ROOT:-/root/X-VLA}"
BASE_MODEL="/mnt/data/ask4help/models/X-VLA-Pt_from5090_v4"
WORK="${WORK:-/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v2}"
PERSIST="${PERSIST:-/mnt/data/ask4help/results/xvla_stackpyramid_oracle_repair_v2}"
ORIGINAL_ROOT="/root/ask4help_stage2_work/xvla_stackpyramid_oracle_repair_v1/id_collection_256_retry1"
MANIFEST="$ROOT/configs/stackpyramid_id_recovery_512_manifest.json"
PYTHONPATH_VALUE="$ROOT:$XVLA_ROOT:${PYTHONPATH:-}"

COLLECTION_SMOKE="$WORK/id_collection_smoke_retry5"
COLLECTION="$WORK/id_collection_additional_256_retry1"
COLLECTION_AUDIT="$WORK/id_collection_additional_256_audit_retry1"
MERGED="$WORK/id_training_collection_512_retry1"
MERGED_AUDIT="$WORK/id_training_collection_512_audit_retry1"
SMOKE="$WORK/fresh_id_train_smoke_retry1"
TRAIN="$WORK/id_sft_20000_retry1"
SELECTION="$WORK/id_checkpoint_selection_retry1"
FORMAL="$WORK/formal_id_gate_100_retry1"
STATE="$WORK/pipeline_state.json"
LOG="$WORK/controller.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PYTHONPATH_VALUE"
export STACKPYRAMID_OOD_GEOMETRY=v4
mkdir -p "$WORK" "$PERSIST"

write_state() {
  local stage="$1" status="$2"
  "$PY" - "$STATE" "$PERSIST/pipeline_state.json" "$stage" "$status" "$MANIFEST" <<'PY'
import json, sys, time
from pathlib import Path
local, durable, stage, status, manifest = map(Path, sys.argv[1:])
payload = {
    "format": "stackpyramid_id_recovery_512_controller_v1",
    "stage": stage.name if False else str(stage),
    "status": str(status),
    "manifest": str(manifest),
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
local.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
durable.parent.mkdir(parents=True, exist_ok=True)
durable.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

run_once() {
  echo "[$(date '+%F %T%z')] $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

copy_once() {
  local source="$1" durable="$2"
  if [[ -e "$durable" ]]; then
    echo "refusing to overwrite existing durable path: $durable" | tee -a "$LOG"
    return 1
  fi
  mkdir -p "$(dirname "$durable")"
  cp -a "$source" "$durable"
}

if [[ -e "$PERSIST/PIPELINE_COMPLETE" || -e "$PERSIST/PIPELINE_FAILED" ]]; then
  echo "pipeline already has a terminal marker: $PERSIST" >&2
  exit 2
fi
if [[ ! -f "$BASE_MODEL/config.json" || ! -f "$BASE_MODEL/model.safetensors" ]]; then
  echo "canonical original X-VLA base is missing: $BASE_MODEL" >&2
  exit 3
fi
if [[ ! -d "$ORIGINAL_ROOT" || ! -f "$ORIGINAL_ROOT/accepted_suffixes.h5" ]]; then
  echo "audited original 256-ID collection is missing: $ORIGINAL_ROOT" >&2
  exit 4
fi
if pgrep -af 'run_stackpyramid.*ood|run_stackpyramid.*timing|run_stackpyramid.*pca|run_stackpyramid.*diff' >/dev/null 2>&1; then
  echo "downstream or duplicate StackPyramid process is alive; refusing to start" >&2
  exit 5
fi

write_state "preflight" "STARTING"

if [[ ! -f "$COLLECTION_SMOKE/COLLECTION_COMPLETE" ]]; then
  write_state "collection_smoke" "RUNNING"
  run_once "$PY" "$ROOT/tools/collect_stackpyramid_xvla_dagger.py" \
    --method offline_oracle --checkpoint "$BASE_MODEL" --xvla-root "$XVLA_ROOT" \
    --output-dir "$COLLECTION_SMOKE" --split id --target 2 \
    --seed-start 886280 --max-attempts 2 --fresh-env-per-episode \
    --sim-backend gpu --render-backend gpu
  test -f "$COLLECTION_SMOKE/COLLECTION_COMPLETE"
  write_state "collection_smoke" "PASSED"
fi

if [[ ! -f "$COLLECTION/COLLECTION_COMPLETE" ]]; then
  write_state "collection_formal" "RUNNING"
  run_once "$PY" "$ROOT/tools/collect_stackpyramid_xvla_dagger.py" \
    --method offline_oracle --checkpoint "$BASE_MODEL" --xvla-root "$XVLA_ROOT" \
    --output-dir "$COLLECTION" --split id --target 256 \
    --seed-start 886300 --max-attempts 320 --fresh-env-per-episode \
    --sim-backend gpu --render-backend gpu
  test -f "$COLLECTION/COLLECTION_COMPLETE"
  copy_once "$COLLECTION" "$PERSIST/$(basename "$COLLECTION")"
  write_state "collection_formal" "PASSED"
fi

if [[ ! -f "$COLLECTION_AUDIT/ID_COLLECTION_AUDIT_PASS" ]]; then
  write_state "collection_audit" "RUNNING"
  run_once "$PY" "$ROOT/tools/audit_stackpyramid_id_collection.py" \
    --collection-root "$COLLECTION" --output "$COLLECTION_AUDIT" \
    --task-spec "$ROOT/configs/stackpyramid_v4_task_spec.json" \
    --expected-episodes 256
  copy_once "$COLLECTION_AUDIT" "$PERSIST/$(basename "$COLLECTION_AUDIT")"
  write_state "collection_audit" "PASSED"
fi

if [[ ! -f "$MERGED/MERGE_COMPLETE" ]]; then
  write_state "merge_512" "RUNNING"
  run_once "$PY" "$ROOT/tools/merge_stackpyramid_id_collections.py" \
    --original-h5 "$ORIGINAL_ROOT/accepted_suffixes.h5" \
    --additional-h5 "$COLLECTION/accepted_suffixes.h5" \
    --output-root "$MERGED" --manifest "$MANIFEST" \
    --original-root "$ORIGINAL_ROOT" --additional-root "$COLLECTION" \
    --expected-total 512 --format-name stackpyramid_id_recovery_512_v1
  copy_once "$MERGED" "$PERSIST/$(basename "$MERGED")"
  write_state "merge_512" "PASSED"
fi

if [[ ! -f "$MERGED_AUDIT/ID_COLLECTION_AUDIT_PASS" ]]; then
  write_state "merged_audit" "RUNNING"
  run_once "$PY" "$ROOT/tools/audit_stackpyramid_id_collection.py" \
    --collection-root "$MERGED" --output "$MERGED_AUDIT" \
    --task-spec "$ROOT/configs/stackpyramid_v4_task_spec.json" \
    --expected-episodes 512 --skip-video-evidence
  copy_once "$MERGED_AUDIT" "$PERSIST/$(basename "$MERGED_AUDIT")"
  write_state "merged_audit" "PASSED"
fi

if [[ ! -f "$SMOKE/RELOAD_SMOKE_COMPLETE" ]]; then
  write_state "train_smoke" "RUNNING"
  run_once "$PY" "$ROOT/tools/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA_ROOT" --model "$BASE_MODEL" \
    --collection-root "$MERGED" --split id --target-episodes 512 \
    --output "$SMOKE" --batch-size 8 --learning-rate 2.5e-5 \
    --learning-coef 0.1 --weight-decay 0.0 --freeze-steps 1000 \
    --warmup-steps 2000 --log-interval 1 --seed 886300 --dtype bf16 --smoke-only
  test -f "$SMOKE/RELOAD_SMOKE_COMPLETE"
  copy_once "$SMOKE" "$PERSIST/$(basename "$SMOKE")"
  write_state "train_smoke" "PASSED"
fi

if [[ ! -f "$TRAIN/TRAINING_COMPLETE" ]]; then
  write_state "training_20000" "STARTING"
  nohup "$PY" "$ROOT/tools/run_stackpyramid_xvla_training.py" \
    --xvla-root "$XVLA_ROOT" --model "$BASE_MODEL" \
    --collection-root "$MERGED" --split id --target-episodes 512 \
    --output "$TRAIN" --steps 20000 --save-interval 5000 \
    --batch-size 8 --learning-rate 2.5e-5 --learning-coef 0.1 \
    --weight-decay 0.0 --freeze-steps 1000 --warmup-steps 2000 \
    --log-interval 20 --seed 886300 --dtype bf16 > "$WORK/training.log" 2>&1 &
  TRAIN_PID=$!
  printf '%s\n' "$TRAIN_PID" > "$WORK/training.pid"
  write_state "training_20000" "RUNNING"
  trap 'kill -CONT "$TRAIN_PID" 2>/dev/null || true' EXIT INT TERM
  for STEP in 5000 10000 15000 20000; do
    while [[ ! -d "$TRAIN/ckpt-$STEP" ]]; do
      kill -0 "$TRAIN_PID" 2>/dev/null || { write_state "training_20000" "FAILED_BEFORE_CKPT_$STEP"; exit 10; }
      sleep 60
    done
    sleep 10
  done
  wait "$TRAIN_PID"
  test -f "$TRAIN/TRAINING_COMPLETE"
  copy_once "$TRAIN" "$PERSIST/$(basename "$TRAIN")"
  write_state "training_20000" "PASSED"
fi

if [[ ! -f "$SELECTION/SELECTION_COMPLETE" ]]; then
  write_state "checkpoint_selection" "RUNNING"
  run_once "$PY" "$ROOT/tools/run_stackpyramid_id_checkpoint_selection.py" \
    --training-root "$TRAIN" --output-root "$SELECTION" \
    --repo-root "$ROOT" --xvla-root "$XVLA_ROOT" --python "$PY" \
    --start-seed 88500 --episodes 20 --gpu 0 --cpu-set 0-19 \
    --checkpoint-steps 5000,10000,15000,20000
  copy_once "$SELECTION" "$PERSIST/$(basename "$SELECTION")"
  write_state "checkpoint_selection" "PASSED"
fi

SELECTED="$($PY - "$SELECTION/SELECTED_CHECKPOINT.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["checkpoint"])
PY
)"

if [[ ! -f "$FORMAL/EVAL_COMPLETE" ]]; then
  write_state "formal_id_gate_100" "RUNNING"
  run_once "$PY" "$ROOT/tools/evaluate_stackpyramid_xvla.py" \
    --checkpoint "$SELECTED" --xvla-root "$XVLA_ROOT" --output "$FORMAL" \
    --split id --episodes 100 --start-seed 84400 --max-episode-steps 300 \
    --execute-horizon 5 --flow-steps 5 --device cuda --sim-backend gpu \
    --render-backend gpu --formal-evidence --geometry v4 --fresh-env-per-episode
  run_once "$PY" "$ROOT/tools/audit_stackpyramid_formal_id_gate.py" \
    --root "$FORMAL" --expected 100 --minimum-successes 80 \
    --output "$FORMAL/formal_audit.json" || true
  copy_once "$FORMAL" "$PERSIST/$(basename "$FORMAL")"
  if "$PY" - "$FORMAL/formal_audit.json" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["audit_pass"] else 1)
PY
  then
    printf 'selected=%s\nstrict_success>=80/100\n' "$SELECTED" > "$PERSIST/ID_BASE_VALIDATED"
    write_state "formal_id_gate_100" "PASSED"
  else
    printf 'selected=%s\nstrict_success<80/100 or evidence failure\n' "$SELECTED" > "$PERSIST/ID_BASE_NOT_ACCEPTED"
    write_state "formal_id_gate_100" "FAILED_DOWNSTREAM_LOCKED"
  fi
fi

printf 'StackPyramid 512-ID recovery complete; OOD/PCA/timing not started.\n' > "$PERSIST/PIPELINE_COMPLETE"
write_state "pipeline" "COMPLETE"
