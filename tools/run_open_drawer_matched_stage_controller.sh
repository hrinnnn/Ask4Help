#!/usr/bin/env bash
set -u

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=$ROOT/RLinf
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=$ROOT/results/open_drawer_failure_detection_v1/matched_updates_v1

OFF_NAME=${OPEN_DRAWER_OFFLINE_METHOD:-offline_oracle_sft_10000_from_id2000_retry3}
FAIL_NAME=${OPEN_DRAWER_FAILURE_METHOD:-failure_recovery_sft_10000_from_id2000_retry2}
GATED_NAME=${OPEN_DRAWER_GATED_METHOD:-robot_gated_sft_10000_from_id2000_retry2}
OFF_PIDFILE=$RUN/pids/$OFF_NAME.pid
FAIL_PIDFILE=$RUN/pids/$FAIL_NAME.pid
GATED_PIDFILE=$RUN/pids/$GATED_NAME.pid
OFF_LOG=$RUN/logs/$OFF_NAME.log
FAIL_LOG=$RUN/logs/$FAIL_NAME.log
GATED_LOG=$RUN/logs/$GATED_NAME.log
GATED_OUT=$RUN/$GATED_NAME

checkpoint_file() {
  printf '%s/%s/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt' "$RUN" "$1"
}

pid_alive() {
  local pidfile=$1
  [ -f "$pidfile" ] || return 1
  kill -0 "$(cat "$pidfile")" 2>/dev/null
}

mkdir -p "$RUN/pids" "$RUN/logs"
while true; do
  off_alive=0
  fail_alive=0
  pid_alive "$OFF_PIDFILE" && off_alive=1 || true
  pid_alive "$FAIL_PIDFILE" && fail_alive=1 || true
  echo "$(date -Is) offline_alive=$off_alive failure_alive=$fail_alive" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  if [ "$off_alive" -eq 0 ] && [ "$fail_alive" -eq 0 ]; then
    if [ -f "$(checkpoint_file "$OFF_NAME")" ] && [ -f "$(checkpoint_file "$FAIL_NAME")" ]; then
      break
    fi
    echo "one matched training ended before step10000 checkpoint" >> "$RUN/logs/matched_update_stage_controller_retry.log"
    exit 1
  fi
  sleep 1800
done

if [ -e "$GATED_OUT" ]; then
  echo "refusing to overwrite existing gated output: $GATED_OUT" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  exit 1
fi

mkdir -p /data/odray_gated_retry2 "$ROOT/runtime_cache/tmp_gated_retry2"
gpu=
for candidate in 1 6; do
  used=$(nvidia-smi -i "$candidate" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  if [ "$used" -lt 1000 ]; then
    gpu=$candidate
    break
  fi
done
while [ -z "$gpu" ]; do
  echo "$(date -Is) waiting for GPU1/GPU6" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  sleep 300
  for candidate in 1 6; do
    used=$(nvidia-smi -i "$candidate" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [ "$used" -lt 1000 ]; then gpu=$candidate; break; fi
  done
done

if [ "$gpu" -eq 1 ]; then cpuset=0-19; else cpuset=20-39; fi
export PYTHONPATH=$RL EMBODIED_PATH=$RL/examples/sft
export ASK4HELP_RLINF_PLACEMENT=$gpu-$gpu
export OPEN_DRAWER_ID_DATASET=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1
export OPEN_DRAWER_EXPERT_DATASET=$ROOT/results/open_drawer_failure_detection_v1/collections/robot_gated_v4/lerobot_dataset
export OPEN_DRAWER_ID_NORM_STATS=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
export OPEN_DRAWER_PI05_MODEL_PATH=$ROOT/results/id_policy_training_v1/sft_2000/checkpoints/global_step_2000
export OPEN_DRAWER_EXPERIMENT_NAME=$GATED_NAME OPEN_DRAWER_RUN_ROOT=$RUN OPEN_DRAWER_TRAIN_SEED=9206
export HF_DATASETS_CACHE=$ROOT/runtime_cache/hf_datasets HF_HOME=$ROOT/runtime_cache/hf_home
export RAY_TMPDIR=/data/odray_gated_retry2 TMPDIR=$ROOT/runtime_cache/tmp_gated_retry2 PYTHONUNBUFFERED=1

cd "$RL/examples/sft"
nohup taskset -c "$cpuset" env PYTHONPATH=$PYTHONPATH EMBODIED_PATH=$EMBODIED_PATH \
  ASK4HELP_RLINF_PLACEMENT=$ASK4HELP_RLINF_PLACEMENT OPEN_DRAWER_ID_DATASET=$OPEN_DRAWER_ID_DATASET \
  OPEN_DRAWER_EXPERT_DATASET=$OPEN_DRAWER_EXPERT_DATASET OPEN_DRAWER_ID_NORM_STATS=$OPEN_DRAWER_ID_NORM_STATS \
  OPEN_DRAWER_PI05_MODEL_PATH=$OPEN_DRAWER_PI05_MODEL_PATH OPEN_DRAWER_EXPERIMENT_NAME=$OPEN_DRAWER_EXPERIMENT_NAME \
  OPEN_DRAWER_RUN_ROOT=$OPEN_DRAWER_RUN_ROOT OPEN_DRAWER_TRAIN_SEED=$OPEN_DRAWER_TRAIN_SEED \
  HF_DATASETS_CACHE=$HF_DATASETS_CACHE HF_HOME=$HF_HOME RAY_TMPDIR=$RAY_TMPDIR TMPDIR=$TMPDIR \
  PYTHONUNBUFFERED=1 "$PY" train_vla_sft.py --config-path "$RL/examples/sft/config" \
  --config-name open_drawer_retrieve_place_dagger_sft_openpi_pi05 \
  runner.max_steps=10000 runner.save_interval=500 runner.resume_dir=null \
  > "$GATED_LOG" 2>&1 < /dev/null &
echo $! > "$GATED_PIDFILE"
sleep 30
if ! pid_alive "$GATED_PIDFILE"; then
  echo "gated process exited during startup" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  exit 1
fi
mask_ok=0
for _ in $(seq 1 60); do
  if grep -q actions_is_pad "$GATED_LOG" 2>/dev/null; then
    mask_ok=1
    break
  fi
  if ! pid_alive "$GATED_PIDFILE"; then
    break
  fi
  sleep 10
done
if [ "$mask_ok" -ne 1 ]; then
  echo "gated startup missing temporal mask marker" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  exit 1
fi
echo "$(date -Is) gated_started gpu=$gpu pid=$(cat "$GATED_PIDFILE")" >> "$RUN/logs/matched_update_stage_controller_retry.log"

while pid_alive "$GATED_PIDFILE"; do
  echo "$(date -Is) gated_alive=1" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  sleep 1800
done

if [ -f "$(checkpoint_file "$GATED_NAME")" ]; then
  printf 'Matched OpenDrawer training completed through step10000.\n' > "$GATED_OUT/TRAINING_COMPLETE"
  echo "$(date -Is) gated_training_complete" >> "$RUN/logs/matched_update_stage_controller_retry.log"
else
  echo "gated training ended before step10000 checkpoint" >> "$RUN/logs/matched_update_stage_controller_retry.log"
  exit 1
fi
