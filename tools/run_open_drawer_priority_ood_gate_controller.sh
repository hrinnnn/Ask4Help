#!/usr/bin/env bash
set -u

# Boundary controller for the 2026-08-29 checkpoint-first decision.  It only
# pauses/replaces the two timing-training parent shells after their currently
# running child has written a complete final checkpoint.  The child training
# process is never stopped by this controller.  The replacement controllers
# contain the persistent per-checkpoint OOD20 wait gate.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:?set OPEN_DRAWER_TIMING_ROOT}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
ID_DATASET=${OPEN_DRAWER_TIMING_ID_DATASET:?set OPEN_DRAWER_TIMING_ID_DATASET}
BUDGET=${OPEN_DRAWER_TIMING_BUDGET_ROOT:-$RUN/formal_budget}
FORMAL_ROOT=${OPEN_DRAWER_TIMING_FORMAL_ROOT:-$RUN/formal}
FORMAL_MARKER=${OPEN_DRAWER_TIMING_FORMAL_MARKER:-$RUN/formal/TIMING_COLLECTION_COMPLETE}

MAIN_TOTAL_PID=${OPEN_DRAWER_PRIORITY_MAIN_TOTAL_PID:?set OPEN_DRAWER_PRIORITY_MAIN_TOTAL_PID}
MAIN_TOTAL_CHILD_PID=${OPEN_DRAWER_PRIORITY_MAIN_TOTAL_CHILD_PID:?set OPEN_DRAWER_PRIORITY_MAIN_TOTAL_CHILD_PID}
MAIN_TRAIN_PARENT_PID=${OPEN_DRAWER_PRIORITY_MAIN_TRAIN_PARENT_PID:?set OPEN_DRAWER_PRIORITY_MAIN_TRAIN_PARENT_PID}
MAIN_TRAIN_CHILD_PID=${OPEN_DRAWER_PRIORITY_MAIN_TRAIN_CHILD_PID:?set OPEN_DRAWER_PRIORITY_MAIN_TRAIN_CHILD_PID}
HELPER_PARENT_PID=${OPEN_DRAWER_PRIORITY_HELPER_PARENT_PID:?set OPEN_DRAWER_PRIORITY_HELPER_PARENT_PID}
HELPER_CHILD_PID=${OPEN_DRAWER_PRIORITY_HELPER_CHILD_PID:?set OPEN_DRAWER_PRIORITY_HELPER_CHILD_PID}

STATE=$RUN/priority_ood20_gate_controller_state.json
LOG=$RUN/priority_ood20_gate_controller.log
exec > >(tee -a "$LOG") 2>&1

mkdir -p "$RUN"

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_priority_ood20_gate_controller_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

die() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "priority gate controller failed at $1: ${2:-}" > "$RUN/PRIORITY_OOD20_GATE_FAILED"
  exit 1
}

pid_stat() {
  ps -p "$1" -o stat= 2>/dev/null | tr -d '[:space:]'
}

pid_live() {
  local stat
  stat=$(pid_stat "$1")
  [[ -n "$stat" && "$stat" != Z* ]]
}

require_cmd() {
  local pid=$1 needle=$2
  local cmd
  cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
  [[ "$cmd" == *"$needle"* ]] || die preflight "pid=$pid does not match expected command substring=$needle cmd=$cmd"
}

wait_for_checkpoint_and_exit() {
  local label=$1 child=$2 checkpoint=$3
  while [[ ! -s "$checkpoint" ]]; do
    pid_live "$child" || die "${label}_checkpoint" "child exited before checkpoint: $checkpoint"
    write_state "waiting_${label}_checkpoint" waiting "child=$child checkpoint_pending"
    sleep 30
  done
  write_state "${label}_checkpoint_written" running "waiting for child cleanup before controller replacement"
  while pid_live "$child"; do
    sleep 15
  done
  printf '%s\n' "$label checkpoint complete: $checkpoint"
}

stop_and_replace_main() {
  local checkpoint="$RUN/training/anchor_0/seed_9302/run/checkpoints/global_step_2500/actor/model_state_dict/full_weights.pt"
  wait_for_checkpoint_and_exit main "$MAIN_TRAIN_CHILD_PID" "$checkpoint"
  require_cmd "$MAIN_TOTAL_PID" "run_open_drawer_grasp_timing_pipeline_controller.sh"
  require_cmd "$MAIN_TOTAL_CHILD_PID" "run_open_drawer_grasp_timing_pipeline_controller.sh"
  write_state main_replacing_total running "old training parent stopped at checkpoint boundary"
  kill -STOP "$MAIN_TOTAL_PID" "$MAIN_TOTAL_CHILD_PID" 2>/dev/null || die main_replacing_total "cannot stop old total controller"
  kill -KILL "$MAIN_TRAIN_PARENT_PID" 2>/dev/null || true
  kill -KILL "$MAIN_TOTAL_CHILD_PID" "$MAIN_TOTAL_PID" 2>/dev/null || true
  sleep 2
  if pid_live "$MAIN_TOTAL_PID" || pid_live "$MAIN_TOTAL_CHILD_PID"; then
    die main_replacing_total "old total controller did not exit"
  fi
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_PYTHON="$PY" \
    OPEN_DRAWER_TIMING_ROOT="$RUN" OPEN_DRAWER_TIMING_CHECKPOINT="$MODEL" \
    OPEN_DRAWER_TIMING_PI05_BASE="$PI05_BASE" OPEN_DRAWER_TIMING_NORM="$NORM" \
    OPEN_DRAWER_TIMING_ID_DATASET="$ID_DATASET" OPEN_DRAWER_TIMING_BUDGET_ROOT="$BUDGET" \
    OPEN_DRAWER_TIMING_FORMAL_ROOT="$FORMAL_ROOT" OPEN_DRAWER_TIMING_FORMAL_MARKER="$FORMAL_MARKER" \
    OPEN_DRAWER_PRIORITY_OOD20_GATE=1 \
    nohup bash "$ROOT/tools/run_open_drawer_grasp_timing_pipeline_controller.sh" \
      >> "$RUN/priority_restarted_total_controller.log" 2>&1 &
  local new_pid=$!
  printf '%s\n' "$new_pid" > "$RUN/priority_restarted_total_controller.pid"
  write_state main_replaced running "new total controller pid=$new_pid; checkpoint-first gate enabled"
}

stop_and_replace_helper() {
  local checkpoint="$RUN/training/anchor_120/seed_9301/run/checkpoints/global_step_2500/actor/model_state_dict/full_weights.pt"
  wait_for_checkpoint_and_exit helper "$HELPER_CHILD_PID" "$checkpoint"
  require_cmd "$HELPER_PARENT_PID" "run_open_drawer_grasp_timing_parallel_helper.sh"
  kill -KILL "$HELPER_PARENT_PID" 2>/dev/null || true
  sleep 2
  if pid_live "$HELPER_PARENT_PID"; then
    die helper_replacing "old helper controller did not exit"
  fi
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_RLINF_ROOT="$ROOT/RLinf" OPEN_DRAWER_PYTHON="$PY" \
    OPEN_DRAWER_TIMING_ROOT="$RUN" OPEN_DRAWER_TIMING_CHECKPOINT="$MODEL" \
    OPEN_DRAWER_TIMING_ID_DATASET="$ID_DATASET" OPEN_DRAWER_TIMING_NORM="$NORM" \
    OPEN_DRAWER_TIMING_BUDGET_ROOT="$BUDGET" OPEN_DRAWER_PRIORITY_OOD20_GATE=1 \
    nohup bash "$ROOT/tools/run_open_drawer_grasp_timing_parallel_helper.sh" \
      >> "$RUN/priority_restarted_parallel_helper.log" 2>&1 &
  local new_pid=$!
  printf '%s\n' "$new_pid" > "$RUN/priority_restarted_parallel_helper.pid"
  write_state helper_replaced running "new helper controller pid=$new_pid; checkpoint-first gate enabled"
}

require_cmd "$MAIN_TOTAL_PID" "run_open_drawer_grasp_timing_pipeline_controller.sh"
require_cmd "$MAIN_TOTAL_CHILD_PID" "run_open_drawer_grasp_timing_pipeline_controller.sh"
require_cmd "$MAIN_TRAIN_PARENT_PID" "run_open_drawer_grasp_timing_training.sh"
require_cmd "$MAIN_TRAIN_CHILD_PID" "train_vla_sft.py"
require_cmd "$HELPER_PARENT_PID" "run_open_drawer_grasp_timing_parallel_helper.sh"
require_cmd "$HELPER_CHILD_PID" "train_vla_sft.py"

write_state preflight running "validated current PIDs; pausing only training parent shells"
kill -STOP "$MAIN_TRAIN_PARENT_PID" "$HELPER_PARENT_PID" 2>/dev/null || die preflight "cannot stop current training parent shells"
write_state parents_paused running "children continue; waiting for their final checkpoints"

stop_and_replace_main &
main_job=$!
stop_and_replace_helper &
helper_job=$!
wait "$main_job" || exit 1
wait "$helper_job" || exit 1
write_state replacements_complete complete "checkpoint-first training controllers active"
printf '%s\n' 'OPEN_DRAWER_PRIORITY_OOD20_GATE_CONTROLLER_COMPLETE'
