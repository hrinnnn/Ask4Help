#!/usr/bin/env bash
set -u

# Shell-level restart supervisor.  The Python recovery controller owns all
# scientific decisions; this wrapper only waits with literal sleep 900/1800
# and relaunches it after an engineering failure when no training PID remains.

RUN=${OPEN_DRAWER_TIMING_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_adaptive_retry1}
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
SCRIPT=${OPEN_DRAWER_RECOVERY_SCRIPT:-$ROOT/tools/run_open_drawer_direct_oracle_adaptive_recovery_controller.py}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:-/sdd/ask4help-open-drawer/results/open_drawer_pi05_v9_recovery_from5000_v4/training/v9_full_prompt/checkpoints/global_step_5000}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:-$ROOT/results/model_cache/pi05_base_pytorch_v1}
ID_DATASET=${OPEN_DRAWER_TIMING_ID_DATASET:-$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1}
NORM=${OPEN_DRAWER_TIMING_NORM:-$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1}
FORMAL_ROOT=${OPEN_DRAWER_TIMING_FORMAL_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1/formal}
BUDGET_ROOT=${OPEN_DRAWER_TIMING_BUDGET_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_formal_retry1/formal_budget}
GPU_POOL=${OPEN_DRAWER_TIMING_GPU_POOL:-"4 7"}
FIXED_OVERRIDE=${OPEN_DRAWER_FIXED_5000:-0}
RAY_TMP_ROOT=${OPEN_DRAWER_TIMING_RAY_TMP_ROOT:-/sdd/r_od1}
TMP_ROOT=${OPEN_DRAWER_TIMING_TMP_ROOT:-/sdd/t_od1}
PIPELINE_MARKER=${OPEN_DRAWER_PIPELINE_MARKER:-$RUN/PIPELINE_COMPLETE}
STATE=${OPEN_DRAWER_RECOVERY_SUPERVISOR_STATE:-$RUN/direct_oracle_adaptive_recovery_supervisor_state.json}
LOG=${OPEN_DRAWER_RECOVERY_SUPERVISOR_LOG:-$RUN/logs/direct_oracle_adaptive_recovery_supervisor.log}
PIDFILE=${OPEN_DRAWER_RECOVERY_SUPERVISOR_PIDFILE:-$RUN/recovery_supervisor.pid}
LOCK=${OPEN_DRAWER_RECOVERY_SUPERVISOR_LOCK:-$RUN/.recovery_supervisor_lock}
CHECK_SECONDS=${OPEN_DRAWER_RECOVERY_SUPERVISOR_CHECK_SECONDS:-900}
STABLE_SECONDS=${OPEN_DRAWER_RECOVERY_SUPERVISOR_STABLE_SECONDS:-1800}

mkdir -p "$(dirname "$STATE")" "$(dirname "$LOG")"
if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/owner" 2>/dev/null || true)
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then
    exit 0
  fi
  rmdir "$LOCK" 2>/dev/null || exit 0
  mkdir "$LOCK" 2>/dev/null || exit 0
fi
printf '%s\n' "$$" > "$LOCK/owner"
printf '%s\n' "$$" > "$PIDFILE"
cleanup() {
  rm -f "$PIDFILE" "$LOCK/owner"
  rmdir "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" >> "$LOG"
}

write_state() {
  local stage=$1 status=$2 detail=${3:-} attempt=${4:-0}
  printf '%s\n' "{\"format\":\"open_drawer_direct_oracle_adaptive_recovery_supervisor_v1\",\"stage\":\"$stage\",\"status\":\"$status\",\"detail\":\"$detail\",\"attempt\":$attempt,\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

pid_live() {
  local pid=$1
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

recorded_training_live() {
  local file="$RUN/recovery_active_jobs.json" p
  [[ -f "$file" ]] || return 1
  while read -r p; do
    [[ -n "$p" ]] || continue
    if pid_live "$p"; then
      return 0
    fi
  done < <(grep -o '"pid"[[:space:]]*:[[:space:]]*[0-9][0-9]*' "$file" 2>/dev/null | grep -o '[0-9][0-9]*' || true)
  return 1
}

launch_recovery() {
  local attempt=$1 new_pid
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_RLINF_ROOT="$RL" OPEN_DRAWER_PYTHON="$PY" \
    OPEN_DRAWER_TIMING_ROOT="$RUN" OPEN_DRAWER_TIMING_CHECKPOINT="$MODEL" \
    OPEN_DRAWER_TIMING_PI05_BASE="$PI05_BASE" OPEN_DRAWER_TIMING_ID_DATASET="$ID_DATASET" \
    OPEN_DRAWER_TIMING_NORM="$NORM" OPEN_DRAWER_TIMING_FORMAL_ROOT="$FORMAL_ROOT" \
    OPEN_DRAWER_TIMING_BUDGET_ROOT="$BUDGET_ROOT" OPEN_DRAWER_TIMING_GPU_POOL="$GPU_POOL" \
    OPEN_DRAWER_TIMING_RAY_TMP_ROOT="$RAY_TMP_ROOT" OPEN_DRAWER_TIMING_TMP_ROOT="$TMP_ROOT" \
    OPEN_DRAWER_RECOVERY_CHECK_SECONDS=900 OPEN_DRAWER_RECOVERY_STABLE_SECONDS=1800 \
    OPEN_DRAWER_RECOVERY_GPU_WAIT_SECONDS=900 OPEN_DRAWER_OOD20_SEED_START=79000 \
    OPEN_DRAWER_FIXED_5000="$FIXED_OVERRIDE" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
    nohup "$PY" -u "$SCRIPT" >> "$RUN/logs/direct_oracle_adaptive_recovery_supervised_retry.log" 2>&1 &
  new_pid=$!
  printf '%s\n' "$new_pid" > "$RUN/recovery_controller.pid"
  log "launched recovery controller pid=$new_pid attempt=$attempt"
  write_state recovery_controller_launched running "pid=$new_pid" "$attempt"
}

attempt=0
log "shell recovery supervisor started pid=$$"
while [[ ! -f "$PIPELINE_MARKER" ]]; do
  controller_pid=$(cat "$RUN/recovery_controller.pid" 2>/dev/null || true)
  if pid_live "$controller_pid"; then
    write_state waiting_controller running "pid=$controller_pid; next check in ${STABLE_SECONDS}s" "$attempt"
    sleep 1800
    continue
  fi
  if recorded_training_live; then
    write_state waiting_training_pids waiting "controller absent but recorded training PID remains; next check in ${CHECK_SECONDS}s" "$attempt"
    sleep 900
    continue
  fi
  attempt=$((attempt + 1))
  launch_recovery "$attempt"
  sleep 5
  if ! pid_live "$(cat "$RUN/recovery_controller.pid" 2>/dev/null || true)"; then
    log "recovery controller exited during startup; retrying after ${CHECK_SECONDS}s"
    write_state recovery_start_failed waiting "retry in ${CHECK_SECONDS}s" "$attempt"
    sleep 900
  fi
done
write_state pipeline_complete complete "pipeline marker present; supervisor exiting" "$attempt"
log "pipeline marker present; shell recovery supervisor exiting"
