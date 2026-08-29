#!/usr/bin/env bash
set -u

# Durable post-adaptive evaluator.  It waits for the adaptive single-seed
# training marker, acquires one genuinely idle GPU, evaluates each frozen
# anchor on the fixed 100-ID/100-Grasp-OOD splits, and then runs the
# independent reconciliation.  It never changes the scientific protocol.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_retry6_adaptive}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
FORMAL_ROOT=${OPEN_DRAWER_TIMING_FORMAL_ROOT:-$RUN/formal}
BUDGET_ROOT=${OPEN_DRAWER_TIMING_BUDGET_ROOT:-$RUN/formal_budget}
POLICY_ONLY=${OPEN_DRAWER_TIMING_POLICY_ONLY_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_retry2/policy_only_grasp_ood}
EVALUATOR=${OPEN_DRAWER_TIMING_EVALUATOR:-$ROOT/tools/evaluate_open_drawer_id_pi05.py}
EVAL_WRAPPER=${OPEN_DRAWER_TIMING_EVAL_WRAPPER:-$ROOT/tools/run_open_drawer_timing_eval.py}
RECONCILER=${OPEN_DRAWER_TIMING_RECONCILER:-$ROOT/tools/summarize_open_drawer_adaptive_timing.py}
TRAINING_MARKER=$RUN/ADAPTIVE_TIMING_TRAINING_COMPLETE
STATE=$RUN/adaptive_formal_eval_pipeline_state.json
LOG=$RUN/adaptive_formal_eval_controller.log
LOCK_ROOT=${OPEN_DRAWER_TIMING_GPU_LOCK_ROOT:-$RUN/.gpu_locks}
POLL_SECONDS=${OPEN_DRAWER_TIMING_FORMAL_POLL_SECONDS:-900}
ANCHORS=(0 50 80 120 160 220)
GPU_POOL=(${OPEN_DRAWER_TIMING_GPU_POOL:-"0 1 2 3 4 5 6 7"})
TRAIN_SEED=${OPEN_DRAWER_TIMING_TRAIN_SEED:-9301}

mkdir -p "$RUN" "$RUN/evaluation" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_adaptive_formal_eval_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

fail() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "stage=$1 detail=${2:-}" > "$RUN/ADAPTIVE_FORMAL_EVAL_FAILED"
  exit 1
}

pid_alive() {
  kill -0 "$1" 2>/dev/null
}

cpu_set_for_gpu() {
  case "$1" in
    0) printf '%s\n' '0-19' ;;
    1) printf '%s\n' '20-39' ;;
    2) printf '%s\n' '40-59' ;;
    3) printf '%s\n' '60-79' ;;
    4) printf '%s\n' '80-99' ;;
    5) printf '%s\n' '100-119' ;;
    6) printf '%s\n' '120-139' ;;
    7) printf '%s\n' '140-159' ;;
    *) return 1 ;;
  esac
}

gpu_is_idle() {
  local gpu=$1 used util uuid apps
  read -r used util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$uuid" || true)
  [[ "$used" -le 100 && "$util" -le 5 && -z "$apps" ]]
}

acquire_gpu() {
  local gpu lock owner cpuset
  mkdir -p "$LOCK_ROOT"
  while true; do
    for gpu in "${GPU_POOL[@]}"; do
      gpu_is_idle "$gpu" || continue
      lock="$LOCK_ROOT/gpu_$gpu"
      if ! mkdir "$lock" 2>/dev/null; then
        owner=$(cat "$lock/owner" 2>/dev/null || true)
        if [[ -n "$owner" ]] && ! pid_alive "$owner"; then
          rm -f "$lock/owner" 2>/dev/null || true
          rmdir "$lock" 2>/dev/null || true
        fi
        continue
      fi
      cpuset=$(cpu_set_for_gpu "$gpu") || { rmdir "$lock"; continue; }
      if ! gpu_is_idle "$gpu"; then
        rmdir "$lock" 2>/dev/null || true
        continue
      fi
      printf '%s\n' "$$" > "$lock/owner"
      SELECTED_GPU=$gpu
      SELECTED_CPU_SET=$cpuset
      return 0
    done
    write_state waiting_for_idle_gpu waiting "gpu_pool=${GPU_POOL[*]}"
    sleep "$POLL_SECONDS"
  done
}

release_gpu() {
  local lock="$LOCK_ROOT/gpu_$1"
  rm -f "$lock/owner" 2>/dev/null || true
  rmdir "$lock" 2>/dev/null || true
}

checkpoint_for_anchor() {
  local anchor=$1 frozen=$2 marker="$RUN/training/anchor_${anchor}/seed_${TRAIN_SEED}/steps_${frozen}/SEGMENT_COMPLETE"
  [[ -f "$marker" ]] || return 1
  "$PY" - "$marker" <<'PY'
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(r"(?:^|\s)checkpoint=([^\s]+)", text)
if match is None:
    raise SystemExit(1)
print(match.group(1))
PY
}

mkdir -p "$RUN/runtime/ray" "$RUN/runtime/tmp"
write_state waiting_for_adaptive_training waiting "marker=$TRAINING_MARKER"
while [[ ! -f "$TRAINING_MARKER" ]]; do
  if [[ -f "$RUN/ADAPTIVE_TIMING_FAILED" ]]; then
    fail waiting_for_adaptive_training "adaptive_training_failed"
  fi
  sleep "$POLL_SECONDS"
done

[[ -s "$MODEL/actor/model_state_dict/full_weights.pt" ]] || fail preflight "missing immutable base full_weights"
[[ -f "$FORMAL_ROOT/TIMING_COLLECTION_COMPLETE" ]] || fail preflight "missing formal collection marker"
[[ -f "$FORMAL_ROOT/AUDIT_PASS" ]] || fail preflight "missing formal collection audit"
[[ -f "$BUDGET_ROOT/BUDGET_AUDIT_PASS" ]] || fail preflight "missing exact budget audit"

frozen_steps=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["frozen_steps"])' "$RUN/adaptive_steps.json") || fail preflight "invalid adaptive_steps.json"
[[ "$frozen_steps" -ge 5000 ]] || fail preflight "frozen_steps below 5000: $frozen_steps"
write_state preflight_complete running "frozen_steps=$frozen_steps seed=$TRAIN_SEED"

for index in "${!ANCHORS[@]}"; do
  anchor=${ANCHORS[$index]}
  eval_root="$RUN/evaluation/anchor_${anchor}/seed_${TRAIN_SEED}/steps_${frozen_steps}"
  if [[ -f "$eval_root/EVAL_COMPLETE" ]]; then
    write_state "evaluation_anchor_${anchor}" completed "existing verified marker"
    continue
  fi
  if [[ -e "$eval_root" && -n "$(find "$eval_root" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "evaluation_anchor_${anchor}" "refusing partial output: $eval_root"
  fi
  checkpoint=$(checkpoint_for_anchor "$anchor" "$frozen_steps") || fail "checkpoint_anchor_${anchor}" "segment marker missing"
  [[ -s "$checkpoint/actor/model_state_dict/full_weights.pt" && -d "$checkpoint/actor/dcp_checkpoint" ]] || fail "checkpoint_anchor_${anchor}" "checkpoint incomplete: $checkpoint"
  mkdir -p "$eval_root"
  acquire_gpu
  trap 'release_gpu "$SELECTED_GPU"' EXIT
  write_state "evaluation_anchor_${anchor}" running "gpu=$SELECTED_GPU checkpoint=$checkpoint"
  if ! taskset -c "$SELECTED_CPU_SET" env CUDA_VISIBLE_DEVICES="$SELECTED_GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
      PYTHONPATH="$ROOT:$RL" ASK4HELP_RLINF_ROOT="$RL" \
      OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 TOKENIZERS_PARALLELISM=false \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
      "$PY" -u "$EVAL_WRAPPER" --root "$ROOT" --checkpoint "$checkpoint" \
      --pi05-base "$PI05_BASE" --norm-stats "$NORM" --evaluator "$EVALUATOR" \
      --output-root "$eval_root" --python "$PY" --episodes 100 \
      --id-seed "$((78500 + index * 200))" --ood-seed "$((78600 + index * 200))" \
      --gpu "$SELECTED_GPU" --cpu-set "$SELECTED_CPU_SET" \
      > "$RUN/logs/formal_eval_anchor_${anchor}_steps_${frozen_steps}.log" 2>&1; then
    fail "evaluation_anchor_${anchor}" "evaluator_failed"
  fi
  [[ -f "$eval_root/EVAL_COMPLETE" ]] || fail "evaluation_anchor_${anchor}" "missing EVAL_COMPLETE"
  write_state "evaluation_anchor_${anchor}" completed "100-ID+100-Grasp-OOD audited by wrapper"
  release_gpu "$SELECTED_GPU"
  trap - EXIT
done

write_state independent_reconciliation running "six anchors one model seed=$TRAIN_SEED"
if ! "$PY" -u "$RECONCILER" --root "$RUN" --formal-root "$FORMAL_ROOT" --budget-root "$BUDGET_ROOT" \
    --policy-only-root "$POLICY_ONLY" --output "$RUN/final_report.json" \
    --anchors "${ANCHORS[@]}" --seed "$TRAIN_SEED" --episodes 100 \
    > "$RUN/logs/adaptive_reconciliation.log" 2>&1; then
  fail independent_reconciliation "reconciler_failed"
fi
[[ -f "$RUN/INDEPENDENT_RECONCILIATION_COMPLETE" ]] || fail independent_reconciliation "missing reconciliation marker"
[[ -f "$RUN/final_report.md" && -f "$RUN/final_report.json" ]] || fail independent_reconciliation "missing final report"
printf '%s\n' "OpenDrawer adaptive timing sweep complete after independent reconciliation." > "$RUN/PIPELINE_COMPLETE"
write_state pipeline_complete complete "six anchors, frozen_steps=$frozen_steps, seed=$TRAIN_SEED"
echo OPEN_DRAWER_ADAPTIVE_TIMING_PIPELINE_COMPLETE
