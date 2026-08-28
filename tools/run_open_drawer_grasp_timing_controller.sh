#!/usr/bin/env bash
set -u

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
PLANNER=${PANDA_PLANNER_PYTHON:-/data/zhaozhixuan/simplerenv_ms3/env/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:-$ROOT/results/open_drawer_timing_grasp_ood_v1}
CHECKPOINT=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
GPU=${OPEN_DRAWER_TIMING_GPU:-0}
CPU_SET=${OPEN_DRAWER_TIMING_CPU_SET:-0-19}

ANCHORS=(0 50 80 120 160 220)
TARGET=${OPEN_DRAWER_TIMING_TARGET:-5}
MAX_ATTEMPTS=${OPEN_DRAWER_TIMING_MAX_ATTEMPTS:-10}
SEED_START=${OPEN_DRAWER_TIMING_SEED_START:-78100}
STATE=$RUN/pipeline_state.json
LOG=$RUN/controller.log

if [[ -e "$RUN/DIAGNOSTIC_COLLECTION_COMPLETE" || -e "$RUN/PIPELINE_COMPLETE" ]]; then
  echo "timing controller already completed: $RUN" >&2
  exit 2
fi
if [[ -e "$RUN" && -n "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "refusing to reuse non-empty timing root: $RUN" >&2
  exit 2
fi
mkdir -p "$RUN/diagnostic"

write_state() {
  local stage=$1 status=$2 detail=${3:-}
  printf '%s\n' "{\"format\":\"open_drawer_grasp_timing_controller_v1\",\"stage\":\"$stage\",\"status\":\"$status\",\"detail\":\"$detail\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

exec > >(tee -a "$LOG") 2>&1
write_state checkpoint_and_asset_audit running "base=$CHECKPOINT"

for index in "${!ANCHORS[@]}"; do
  step=${ANCHORS[$index]}
  out=$RUN/diagnostic/anchor_$step
  if [[ -e "$out/COLLECTION_COMPLETE" ]]; then
    echo "anchor=$step already complete; keeping evidence"
    continue
  fi
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    write_state "diagnostic_anchor_$step" failed "partial output exists: $out"
    exit 1
  fi
  write_state "diagnostic_anchor_$step" running "takeover_step=$step"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="$GPU" ASK4HELP_RLINF_ROOT="$ROOT/RLinf" \
    PANDA_PLANNER_PYTHON="$PLANNER" PANDA_PLANNER_RENDER_BACKEND=cpu \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    taskset -c "$CPU_SET" "$PY" -u "$ROOT/tools/collect_open_drawer_fixed_timing.py" \
      --checkpoint "$CHECKPOINT" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
      --output-root "$out" --takeover-step "$step" --start-seed "$((SEED_START + index * 100))" \
      --target "$TARGET" --max-attempts "$MAX_ATTEMPTS" \
      > "$RUN/diagnostic_anchor_${step}.log" 2>&1
  rc=$?
  if [[ ! -e "$out/COLLECTION_COMPLETE" ]]; then
    write_state "diagnostic_anchor_$step" failed "collector_rc=$rc; completion marker missing"
    exit 1
  fi
  echo "anchor=$step collector_rc=$rc marker=COLLECTION_COMPLETE"
  sleep 5
done

write_state diagnostic_artifact_audit running "anchors=${ANCHORS[*]}"
AUDIT_PYTHON="$PY" "$PY" -u "$ROOT/tools/audit_open_drawer_grasp_timing.py" \
  --root "$RUN/diagnostic" --anchors "${ANCHORS[@]}" --target "$TARGET" \
  > "$RUN/diagnostic_audit.log" 2>&1
rc=$?
if [[ $rc -ne 0 || ! -e "$RUN/diagnostic/AUDIT_PASS" ]]; then
  write_state diagnostic_artifact_audit failed "audit_rc=$rc"
  exit 1
fi
(cd "$RUN" && printf '%s\n' 'diagnostic collection and independent artifact audit passed' > DIAGNOSTIC_COLLECTION_COMPLETE)
write_state diagnostic_fixed_timing_collection complete "artifact audit passed"
echo "OPEN_DRAWER_GRASP_TIMING_DIAGNOSTIC_COMPLETE"
