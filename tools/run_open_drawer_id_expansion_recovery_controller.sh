#!/usr/bin/env bash
set -u

# Restart-tolerant ID expansion controller. It never starts OOD/PCA/DAgger.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=$ROOT/RLinf
PY=${OPEN_DRAWER_PYTHON:-$RL/.venv/bin/python}
RUN=${OPEN_DRAWER_EXPANSION_ROOT:-/sdd/ask4help-open-drawer/results/open_drawer_id_expansion_recovery_v1}
OLD_DATA=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1
OLD_COLLECTION=$ROOT/results/id_oracle_collection_v1/formal_128_retry1/episodes.jsonl
OLD_VIDEOS=$ROOT/results/id_oracle_collection_v1/formal_128_retry1/videos
BASE=$ROOT/results/open_drawer_failure_detection_v1/id_base_protocol_v3_batch128/sft_10000/checkpoints/global_step_4000
MODEL=$ROOT/results/model_cache/pi05_base_pytorch_v1
NEW_DATA=${OPEN_DRAWER_EXPANSION_NEW_DATA:-$RUN/datasets/id_extra128}
MERGED=${OPEN_DRAWER_EXPANSION_MERGED_DATA:-$RUN/datasets/id_merged256}
NORM=${OPEN_DRAWER_EXPANSION_NORM_STATS:-$RUN/norm_stats/id_merged256}
SMOKE=$RUN/oracle_smoke
COLLECTION=${OPEN_DRAWER_EXPANSION_COLLECTION:-$RUN/collection_extra128}
AUDIT=${OPEN_DRAWER_EXPANSION_AUDIT:-$RUN/audit/data_audit.json}
TRAIN=$RUN/training/sft_from4000_to10000
SMOKE_TRAIN=$RUN/smoke_training
LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
STATE=$RUN/pipeline_state.json
CPU_TRAIN=${OPEN_DRAWER_EXPANSION_CPU_TRAIN:-40-59}
CPU_EVAL=${OPEN_DRAWER_EXPANSION_CPU_EVAL:-60-79}
TRAIN_GPU=${OPEN_DRAWER_EXPANSION_TRAIN_GPU:-2}
EVAL_GPU=${OPEN_DRAWER_EXPANSION_EVAL_GPU:-0}
PLANNER_PYTHONPATH=${OPEN_DRAWER_PLANNER_PYTHONPATH:-}
PLANNER_CUDA_VISIBLE_DEVICES=${OPEN_DRAWER_PLANNER_CUDA_VISIBLE_DEVICES:-0}
NEW_SEED_START=76000
GATE_SEED_START=52000
RAY_SMOKE_DIR=${OPEN_DRAWER_EXPANSION_RAY_SMOKE_DIR:-/sdd/od_open_drawer_expansion_smoke}
RAY_TRAIN_DIR=${OPEN_DRAWER_EXPANSION_RAY_TRAIN_DIR:-/sdd/od_open_drawer_expansion_train}

mkdir -p "$RUN" "$LOG_DIR" "$PID_DIR" "$RUN/provenance"

log() { printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_DIR/controller.log"; }
state() { printf '{"task":"OpenDrawer ID expansion recovery","stage":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' "$1" "$2" "$3" "$(date -Is)" > "$STATE"; }
alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
fail() { log "FAILED $*"; state failed failed "$*"; printf '%s\n' "$*" > "$RUN/PIPELINE_FAILED"; exit 1; }

write_manifest() {
  cat > "$RUN/provenance/run_manifest.txt" <<EOF
task=OpenDrawerRetrievePlace ID expansion recovery
old_id_dataset=$OLD_DATA
new_id_dataset=$NEW_DATA
merged_id_dataset=$MERGED
immutable_step4000=$BASE
pi05_base=$MODEL
old_id_seed_range=75000-75131
new_id_seed_range=76000-76127
gate_id_seed_range=52000-52099
max_episode_steps=400
execute_horizon=5
action_horizon=10
global_batch_size=128
micro_batch_size=32
noise_method=flow_sde
train_expert_only=true
awbc=false
norm_scope=merged ID-only frozen norm
planner_runtime=isolated numpy 1.26.4 with environment mplib 0.1.1
OOD_started=false
EOF
  printf '%s\n' '{"old_id_seed_start":75000,"old_id_episodes":128,"new_id_seed_start":76000,"new_id_episodes":128,"gate_id_seed_start":52000,"gate_id_episodes":100,"split":"id"}' > "$RUN/provenance/seed_manifest.json"
}

checkpoint_complete() {
  [ -f "$TRAIN/checkpoints/global_step_$1/actor/model_state_dict/full_weights.pt" ] &&
    [ -d "$TRAIN/checkpoints/global_step_$1/actor/dcp_checkpoint" ]
}

start_stage() {
  local pid_file=$1 log_file=$2
  shift 2
  if alive "$pid_file"; then return 0; fi
  nohup "$@" > "$log_file" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  sleep 30
  alive "$pid_file"
}

wait_stage() {
  local pid_file=$1 stage=$2
  while alive "$pid_file"; do
    state "$stage" running "pid=$(cat "$pid_file")"
    sleep 300
  done
}

check_oracle_smoke() {
  [ -f "$SMOKE/summary.json" ] || return 1
  "$PY" - "$SMOKE/summary.json" "$SMOKE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[2])
summary = json.loads(Path(sys.argv[1]).read_text())
row = summary.get("id", summary)
videos = list((root / "id" / "videos").glob("*.mp4"))
if row.get("attempts") != 20 or row.get("successes", 0) < 19 or len(videos) != 20:
    raise SystemExit(f"oracle smoke failed: {row}, videos={len(videos)}")
PY
}

run_oracle_smoke() {
  [ -f "$RUN/ORACLE_SMOKE_PASSED" ] && return 0
  if [ -e "$SMOKE" ] && [ ! -f "$SMOKE/summary.json" ]; then fail oracle_smoke_partial_output; fi
  if [ ! -e "$SMOKE" ]; then
    state oracle_smoke starting starting
    start_stage "$PID_DIR/oracle_smoke.pid" "$LOG_DIR/oracle_smoke.log" \
      env CUDA_VISIBLE_DEVICES="$PLANNER_CUDA_VISIBLE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="$PLANNER_PYTHONPATH:$RL:$ROOT" PANDA_PLANNER_PYTHON="$PY" \
      taskset -c "$CPU_TRAIN" "$PY" "$RL/toolkits/lerobot/validate_open_drawer_retrieve_place_oracle.py" \
      --split id --start-seed "$NEW_SEED_START" --num-seeds 20 --output-dir "$SMOKE" --save-video \
      || true
  fi
  wait_stage "$PID_DIR/oracle_smoke.pid" oracle_smoke
  check_oracle_smoke || fail oracle_smoke_gate_failed
  "$PY" "$ROOT/tools/audit_open_drawer_oracle_smoke.py" \
    --root "$SMOKE" --expected 20 --output "$RUN/audit/oracle_smoke.json" \
    || fail oracle_smoke_audit_failed
  printf '%s\n' '20 ID oracle attempts; strict success 20/20; videos and trajectory state/action boundaries passed.' > "$RUN/ORACLE_SMOKE_PASSED"
}

run_collection() {
  [ -f "$RUN/COLLECTION_AUDIT_PASSED" ] && return 0
  if [ -e "$COLLECTION" ] && [ ! -f "$COLLECTION/summary.json" ]; then fail collection_partial_output; fi
  if [ -e "$NEW_DATA" ] && [ ! -f "$NEW_DATA/meta/info.json" ]; then fail new_dataset_partial_output; fi
  if [ ! -e "$COLLECTION" ]; then
    state id_collection_extra128 starting starting
    start_stage "$PID_DIR/collection_extra128.pid" "$LOG_DIR/collection_extra128.log" \
      env CUDA_VISIBLE_DEVICES="$PLANNER_CUDA_VISIBLE_DEVICES" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH="$PLANNER_PYTHONPATH:$RL:$ROOT" PANDA_PLANNER_PYTHON="$PY" HF_LEROBOT_HOME=/sdd/ask4help-open-drawer/lerobot_cache \
      taskset -c "$CPU_TRAIN" "$PY" "$RL/toolkits/lerobot/collect_open_drawer_retrieve_place_lerobot.py" \
      --repo-id "$NEW_DATA" --output-dir "$COLLECTION" --video-dir "$COLLECTION/videos" \
      --num-episodes 128 --seed "$NEW_SEED_START" --max-attempts 160 --image-size 384 \
      --control-freq 10 --max-episode-steps 400 --save-videos \
      || true
  fi
  wait_stage "$PID_DIR/collection_extra128.pid" id_collection_extra128
  [ -f "$COLLECTION/summary.json" ] || fail collection_summary_missing
  state id_data_audit running audit
  "$PY" "$ROOT/tools/audit_open_drawer_id_expansion.py" \
    --old-dataset "$OLD_DATA" --new-dataset "$NEW_DATA" \
    --old-collection "$OLD_COLLECTION" --new-collection "$COLLECTION/episodes.jsonl" \
    --old-videos "$OLD_VIDEOS" --new-videos "$COLLECTION/videos" --output "$AUDIT" \
    || fail id_data_audit_failed
  printf '%s\n' 'Old/new ID datasets, action/state/video alignment, lengths, and temporal-mask boundaries passed.' > "$RUN/COLLECTION_AUDIT_PASSED"
}

run_merge_and_norm() {
  [ -f "$RUN/NORM_FROZEN" ] && return 0
  if [ -e "$MERGED" ] && [ ! -f "$MERGED/meta/info.json" ]; then fail merged_dataset_partial_output; fi
  if [ ! -e "$MERGED" ]; then
    state merge_id_dataset running merging
    mkdir -p "$(dirname "$MERGED")"
    "$PY" "$RL/toolkits/lerobot/merge_lerobot_datasets.py" \
      --source-dir "$OLD_DATA" "$NEW_DATA" --output-dir "$MERGED" \
      > "$LOG_DIR/merge_id_dataset.log" 2>&1 || fail merge_failed
  fi
  "$PY" "$ROOT/tools/audit_open_drawer_id_expansion.py" \
    --old-dataset "$OLD_DATA" --new-dataset "$NEW_DATA" \
    --old-collection "$OLD_COLLECTION" --new-collection "$COLLECTION/episodes.jsonl" \
    --old-videos "$OLD_VIDEOS" --new-videos "$COLLECTION/videos" \
    --merged-dataset "$MERGED" --output "$RUN/audit/merged_audit.json" \
    || fail merged_audit_failed
  if [ -e "$NORM" ] && [ ! -f "$NORM/norm_stats.json" ]; then fail norm_partial_output; fi
  if [ ! -e "$NORM" ]; then
    state norm_stats running computing
    "$PY" "$ROOT/tools/compute_open_drawer_norm_stats.py" --dataset "$MERGED" --output "$NORM" \
      > "$LOG_DIR/norm_stats.log" 2>&1 || fail norm_stats_failed
  fi
  printf '%s\n' 'Merged 256 ID episodes; state/action norm recomputed from ID-only data and frozen.' > "$RUN/NORM_FROZEN"
}

run_training_smoke() {
  [ -f "$RUN/SMOKE_COMPLETE" ] && return 0
  if [ -e "$SMOKE_TRAIN" ] && [ ! -f "$SMOKE_TRAIN/RELOAD_FORWARD_COMPLETE" ]; then
    if [ ! -f "$SMOKE_TRAIN/checkpoints/global_step_4002/actor/model_state_dict/full_weights.pt" ]; then fail training_smoke_partial; fi
  fi
  if [ ! -e "$SMOKE_TRAIN" ]; then
    state training_smoke starting starting
    mkdir -p "$SMOKE_TRAIN/tmp" "$RAY_SMOKE_DIR"
    start_stage "$PID_DIR/training_smoke.pid" "$LOG_DIR/training_smoke.log" \
      env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" ASK4HELP_RLINF_PLACEMENT="$TRAIN_GPU-$TRAIN_GPU" \
      OPEN_DRAWER_ID_DATASET="$MERGED" OPEN_DRAWER_ID_NORM_STATS="$NORM" \
      OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" OPEN_DRAWER_RESUME_DIR="$BASE" \
      OPEN_DRAWER_RUN_ROOT="$SMOKE_TRAIN" OPEN_DRAWER_EXPERIMENT_NAME=expansion_smoke \
      RAY_TMPDIR="$RAY_SMOKE_DIR" TMPDIR="$SMOKE_TRAIN/tmp" HF_HOME="$ROOT/runtime_cache/hf_home" \
      PYTHONUNBUFFERED=1 PYTHONPATH="$RL:$ROOT" EMBODIED_PATH="$RL/examples/sft" REPO_PATH="$RL" \
      taskset -c "$CPU_TRAIN" "$PY" "$RL/examples/sft/train_vla_sft.py" \
      --config-path "$RL/examples/sft/config" --config-name open_drawer_retrieve_place_id_continuation_openpi_pi05 \
      runner.max_steps=4002 runner.save_interval=2 \
      || true
  fi
  wait_stage "$PID_DIR/training_smoke.pid" training_smoke
  local smoke_ckpt="$SMOKE_TRAIN/checkpoints/global_step_4002"
  [ -f "$smoke_ckpt/actor/model_state_dict/full_weights.pt" ] || fail training_smoke_checkpoint_missing
  printf '%s\n' '2-step resume training completed from immutable step4000.' > "$SMOKE_TRAIN/SMOKE_COMPLETE"
  if [ ! -f "$SMOKE_TRAIN/RELOAD_FORWARD_COMPLETE" ]; then
    mkdir -p "$SMOKE_TRAIN/reload_forward_eval"
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$RL:$ROOT" \
      taskset -c "$CPU_EVAL" "$PY" "$ROOT/tools/evaluate_open_drawer_id_pi05.py" \
      --checkpoint "$smoke_ckpt" --pi05-base "$MODEL" --norm-stats "$NORM" \
      --output-dir "$SMOKE_TRAIN/reload_forward_eval" --episodes 1 --seed 52000 --split id \
      --execute-horizon 5 --max-episode-steps 20 > "$LOG_DIR/reload_forward_smoke.log" 2>&1 || true
    [ -f "$SMOKE_TRAIN/reload_forward_eval/summary.json" ] || fail reload_forward_summary_missing
    [ "$(find "$SMOKE_TRAIN/reload_forward_eval/videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')" = 1 ] || fail reload_forward_video_missing
    printf '%s\n' 'Reload/forward smoke produced one complete summary and video.' > "$SMOKE_TRAIN/RELOAD_FORWARD_COMPLETE"
  fi
  printf '%s\n' 'Merged-data training smoke and reload/forward smoke passed.' > "$RUN/SMOKE_COMPLETE"
}

run_gate() {
  local step=$1
  local ckpt="$TRAIN/checkpoints/global_step_$step"
  local eval_dir="$RUN/eval_id_100_by_step/step_$step"
  local pid_file="$PID_DIR/eval_id_100_step_$step.pid"
  local log_file="$LOG_DIR/eval_id_100_step_$step.log"
  if [ -e "$eval_dir" ] && [ ! -f "$eval_dir/summary.json" ] && ! alive "$pid_file" \
    && [ -n "$(find "$eval_dir" -mindepth 1 -maxdepth 1 -type f -print -quit 2>/dev/null)" ]; then
    fail "gate_partial_output_step=$step"
  fi
  mkdir -p "$eval_dir"
  if [ ! -f "$eval_dir/summary.json" ] && ! alive "$pid_file"; then
    env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$RL:$ROOT" \
      taskset -c "$CPU_EVAL" nohup "$PY" "$ROOT/tools/evaluate_open_drawer_id_pi05.py" \
      --checkpoint "$ckpt" --pi05-base "$MODEL" --norm-stats "$NORM" \
      --output-dir "$eval_dir" --episodes 100 --seed "$GATE_SEED_START" --split id \
      --execute-horizon 5 --max-episode-steps 400 > "$log_file" 2>&1 < /dev/null &
    echo $! > "$pid_file"
    log "started id_gate step=$step pid=$(cat "$pid_file")"
  fi
  while alive "$pid_file"; do
    state "id_gate_step_$step" running "pid=$(cat "$pid_file")"
    sleep 300
  done
  [ -f "$eval_dir/summary.json" ] || fail "gate_summary_missing_step=$step"
  local videos
  videos=$(find "$eval_dir/videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  "$PY" - "$eval_dir/summary.json" "$RUN" "$step" "$videos" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
run = Path(sys.argv[2])
step = int(sys.argv[3])
videos = int(sys.argv[4])
payload = {
    "step": step,
    "episodes": summary.get("episodes"),
    "successes": summary.get("successes"),
    "video_count": videos,
    "summary": str(Path(sys.argv[1])),
    "videos": str(Path(sys.argv[1]).parent / "videos"),
    "pass": summary.get("episodes") == 100 and videos == 100 and summary.get("successes", 0) >= 80,
}
(run / f"ID_GATE_STEP_{step}.json").write_text(json.dumps(payload, indent=2) + "\n")
if payload["pass"] and not (run / "ID_BASE_CANDIDATE_STEP.json").exists():
    (run / "ID_BASE_CANDIDATE_STEP.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

run_formal_training() {
  [ -f "$RUN/ID_RECOVERY_COMPLETE" ] && return 0
  if [ -e "$TRAIN" ] && [ ! -s "$PID_DIR/train_to_10000.pid" ] && ! checkpoint_complete 10000; then
    fail formal_training_partial_without_pid
  fi
  if ! checkpoint_complete 10000 && ! alive "$PID_DIR/train_to_10000.pid"; then
    state training_to_10000 starting starting
    mkdir -p "$TRAIN/tmp" "$RAY_TRAIN_DIR"
    start_stage "$PID_DIR/train_to_10000.pid" "$LOG_DIR/train_to_10000.log" \
      env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" ASK4HELP_RLINF_PLACEMENT="$TRAIN_GPU-$TRAIN_GPU" \
      OPEN_DRAWER_ID_DATASET="$MERGED" OPEN_DRAWER_ID_NORM_STATS="$NORM" \
      OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" OPEN_DRAWER_RESUME_DIR="$BASE" \
      OPEN_DRAWER_RUN_ROOT="$TRAIN" OPEN_DRAWER_EXPERIMENT_NAME=expansion_recovery_from4000 \
      RAY_TMPDIR="$RAY_TRAIN_DIR" TMPDIR="$TRAIN/tmp" HF_HOME="$ROOT/runtime_cache/hf_home" \
      PYTHONUNBUFFERED=1 PYTHONPATH="$RL:$ROOT" EMBODIED_PATH="$RL/examples/sft" REPO_PATH="$RL" \
      taskset -c "$CPU_TRAIN" "$PY" "$RL/examples/sft/train_vla_sft.py" \
      --config-path "$RL/examples/sft/config" --config-name open_drawer_retrieve_place_id_continuation_openpi_pi05 \
      runner.max_steps=10000 runner.save_interval=500 \
      || fail training_start_failed
  fi
  for step in 6000 8000 10000; do
    while ! checkpoint_complete "$step"; do
      if ! alive "$PID_DIR/train_to_10000.pid"; then fail "training_stopped_before_checkpoint=$step"; fi
      state "training_to_$step" running "pid=$(cat "$PID_DIR/train_to_10000.pid")"
      sleep 600
    done
    [ -f "$RUN/ID_GATE_STEP_$step.json" ] || run_gate "$step"
  done
  wait_stage "$PID_DIR/train_to_10000.pid" training_to_10000
  if [ -f "$RUN/ID_BASE_CANDIDATE_STEP.json" ]; then
    cp -n "$RUN/ID_BASE_CANDIDATE_STEP.json" "$RUN/ID_BASE_FROZEN_PENDING_TOTAL_CONTROL_REVIEW.json"
    printf '%s\n' 'ID recovery passed; earliest passing checkpoint is pending total-control review. OOD remains locked.' > "$RUN/ID_RECOVERY_COMPLETE"
    state complete passed awaiting_total_control_review
  else
    printf '%s\n' 'No checkpoint at 6000, 8000, or 10000 reached 80/100 independent ID success. OOD remains locked.' > "$RUN/ID_BASE_PROTOCOL_FAILED"
    printf '%s\n' 'ID recovery completed with no passing checkpoint; diagnostic only.' > "$RUN/ID_RECOVERY_COMPLETE"
    state complete failed id_gate_failed_all_checkpoints
  fi
}

main() {
  state preflight running checking_inputs
  for path in "$OLD_DATA" "$OLD_COLLECTION" "$OLD_VIDEOS" "$BASE/actor/model_state_dict/full_weights.pt" "$BASE/actor/dcp_checkpoint" "$MODEL"; do
    [ -e "$path" ] || fail "missing_input=$path"
  done
  if [ ! -f "$RUN/CONTROLLER_READY" ]; then
    write_manifest
    printf '%s\n' 'Controller preflight passed; canonical recovery root is on /sdd.' > "$RUN/CONTROLLER_READY"
  fi
  run_oracle_smoke
  run_collection
  run_merge_and_norm
  run_training_smoke
  run_formal_training
}

main "$@"
