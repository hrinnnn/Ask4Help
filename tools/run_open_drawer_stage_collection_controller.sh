#!/usr/bin/env bash
set -u

# Stage-aware OpenDrawer collector. Existing aggregate runs are diagnostic; this
# controller creates independent Handle/Grasp/Goal datasets for each method.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
TOOLS=${OPEN_DRAWER_TOOLS:-/data/zhaozhixuan/Ask4Help/tools}
RLINF=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
RUN=$ROOT/results/open_drawer_failure_detection_v1/stage_specific_v1
SIDE=$ROOT/results/open_drawer_failure_detection_v1/matched_updates_v1/parallel_pca_diff_v1
TARGET_PER_SOURCE=50
MAX_ATTEMPTS=1600
LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
mkdir -p "$LOG_DIR" "$PID_DIR"
LOG=$LOG_DIR/controller.log

DET=$ROOT/results/open_drawer_failure_detection_v1/assets_id_v1/detector_assets.pt
THRESH=$ROOT/results/open_drawer_failure_detection_v1/calibration_id20_shard2_v1/calibration.json
DIFF_CAL=$SIDE/diff_calibration_successful_id_v1/calibration.json
CKPT=$ROOT/results/id_policy_training_v1/sft_2000/checkpoints/global_step_2000
NORM=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
PI05=$ROOT/results/model_cache/pi05_base_pytorch_v1
ID_DATA=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1

PLANNER_PY=${PANDA_PLANNER_PYTHON:-/data/zhaozhixuan/simplerenv_ms3/env/bin/python}
PLANNER_SITE=${PANDA_PLANNER_SITE_PACKAGES:-/data/zhaozhixuan/simplerenv_ms3/env/lib/python3.10/site-packages}

alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

next_out() {
  local base=$1 n=1 out
  out=${base}_retry${n}
  while [ -e "$out" ]; do n=$((n + 1)); out=${base}_retry${n}; done
  printf '%s' "$out"
}

run_collector() {
  local name=$1 gpu=$2 cpuset=$3 method=$4 split=$5 id_seed=$6 ood_seed=$7 target=$8 out=$9
  local log=$LOG_DIR/${name}.log pidfile=$PID_DIR/${name}.pid
  local cuda=$gpu
  if [ "$gpu" = "cpu" ]; then cuda=""; fi
  local -a cmd=("$PY" "$TOOLS/collect_open_drawer_dagger.py"
    --method "$method" --checkpoint "$CKPT" --pi05-base "$PI05" --norm-stats "$NORM")
  if [ "$method" = "diffdagger" ]; then
    cmd+=(--detector-assets "$DET" --diff-calibration "$DIFF_CAL"
      --diff-alpha 0.95 --diff-patience 2 --diff-timesteps 16 --diff-noise-samples 1)
  elif [ "$method" = "pca_only" ] || [ "$method" = "failure_recovery" ]; then
    cmd+=(--detector-assets "$DET" --thresholds "$THRESH")
  fi
  cmd+=(--output-root "$out" --target-per-source "$target" --max-attempts "$MAX_ATTEMPTS"
    --id-seed-start "$id_seed" --ood-seed-start "$ood_seed" --ood-split "$split"
    --execute-horizon 5 --max-policy-steps 240)
  env ASK4HELP_RLINF_ROOT="$RLINF" \
    PANDA_PLANNER_PYTHON="$PLANNER_PY" \
    PANDA_PLANNER_SITE_PACKAGES="$PLANNER_SITE" \
    CUDA_VISIBLE_DEVICES="$cuda" \
    OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
    taskset -c "$cpuset" nohup "${cmd[@]}" >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$(date -Is) started name=$name gpu=$gpu pid=$(cat "$pidfile") output=$out" >> "$LOG"
}

wait_job() {
  local name=$1 out=$2 pidfile=$PID_DIR/${name}.pid
  while alive "$pidfile"; do sleep 300; done
  if [ -f "$out/COLLECTION_COMPLETE" ]; then
    echo "$(date -Is) complete name=$name" >> "$LOG"
    return 0
  fi
  echo "$(date -Is) failed_or_incomplete name=$name" >> "$LOG"
  return 1
}

echo "$(date -Is) controller_started" >> "$LOG"
OLD_PCA="$SIDE/pids/pca_only_collection_v1.pid"
OLD_DIFF_PIDS=(
  "$SIDE/pids/diffdagger_collection_successful_id_v1.pid"
  "$SIDE/pids/diffdagger_collection_retry1.pid"
)

# The aggregate DiffDAgger run can be long.  Start the stage PCA as soon as
# the aggregate PCA releases GPU4, then let it overlap with the tail of the
# aggregate DiffDAgger run on GPU5.  This keeps the four-GPU budget useful
# without duplicating a stage or touching the main training GPUs.
while alive "$OLD_PCA"; do sleep 300; done
echo "$(date -Is) aggregate_pca_exited" >> "$LOG"

wait_old_diff() {
  local old
  for old in "${OLD_DIFF_PIDS[@]}"; do
    while alive "$old"; do sleep 300; done
  done
}

active_stage_output() {
  local base=$1 out name pidfile
  for out in "${base}"_retry*; do
    [ -d "$out" ] || continue
    name=$(basename "$out")
    pidfile="$PID_DIR/${name}.pid"
    if alive "$pidfile"; then
      printf '%s' "$out"
      return 0
    fi
  done
  return 1
}

completed_stage_output() {
  local base=$1 out
  for out in "${base}"_retry*; do
    [ -d "$out" ] || continue
    if [ -f "$out/COLLECTION_COMPLETE" ]; then
      printf '%s' "$out"
      return 0
    fi
  done
  return 1
}

record_part() {
  local stage=$1 out=$2 list=$stage/collection_parts.txt
  mkdir -p "$stage"
  touch "$list"
  if ! grep -Fqx "$out" "$list"; then
    printf '%s\n' "$out" >> "$list"
  fi
}

read_total_counts() {
  local stage=$1
  "$PY" - "$stage/collection_parts.txt" <<'PY'
import json
import sys
from pathlib import Path

parts = Path(sys.argv[1])
total_id = 0
total_ood = 0
for line in parts.read_text().splitlines() if parts.exists() else []:
    summary = Path(line.strip()) / "summary.json"
    if not summary.exists():
        continue
    payload = json.loads(summary.read_text())
    total_id += int(payload.get("accepted_id", 0))
    total_ood += int(payload.get("accepted_ood", 0))
print(total_id, total_ood)
PY
}

wait_collection_with_retries() {
  local stage=$1 base=$2 current_out=$3 current_name=$4 gpu=$5 cpuset=$6 method=$7 split=$8 id_seed=$9 ood_seed=${10}
  local out="$current_out" name="$current_name" target=$TARGET_PER_SOURCE
  local total_id total_ood remaining_id remaining_ood
  while true; do
    if [ -z "$out" ]; then
      if out=$(completed_stage_output "$base"); then
        name=$(basename "$out")
      elif out=$(active_stage_output "$base"); then
        name=$(basename "$out")
      else
        out=$(next_out "$base")
        name=$(basename "$out")
        run_collector "$name" "$gpu" "$cpuset" "$method" "$split" "$id_seed" "$ood_seed" "$target" "$out"
      fi
    fi
    record_part "$stage" "$out"
    if wait_job "$name" "$out"; then
      return 0
    fi
    read -r total_id total_ood < <(read_total_counts "$stage" || printf '0 0\n')
    total_id=${total_id:-0}
    total_ood=${total_ood:-0}
    remaining_id=$((TARGET_PER_SOURCE - total_id))
    remaining_ood=$((TARGET_PER_SOURCE - total_ood))
    if [ "$remaining_id" -le 0 ] && [ "$remaining_ood" -le 0 ]; then
      echo "$(date -Is) incomplete_without_remaining_quota method=$method split=$split" >> "$LOG"
      return 1
    fi
    target=$remaining_id
    [ "$remaining_ood" -gt "$target" ] && target=$remaining_ood
    [ "$target" -gt 0 ] || target=1
    id_seed=$((id_seed + 10000))
    ood_seed=$((ood_seed + 10000))
    echo "$(date -Is) continuing method=$method split=$split total_id=$total_id total_ood=$total_ood next_target=$target" >> "$LOG"
    out=""
    name=""
  done
}

stage_index=0
for split in handle_ood grasp_ood goal_ood; do
  stage_index=$((stage_index + 1))
  stage=$RUN/$split
  mkdir -p "$stage"
  id_seed=$((100000 + stage_index * 4000))
  ood_seed=$((110000 + stage_index * 4000))

  if pca_out=$(active_stage_output "$stage/pca_only"); then
    pca_name=$(basename "$pca_out")
    echo "$(date -Is) resuming name=$pca_name output=$pca_out" >> "$LOG"
  elif pca_out=$(completed_stage_output "$stage/pca_only"); then
    pca_name=$(basename "$pca_out")
    echo "$(date -Is) reusing_completed name=$pca_name output=$pca_out" >> "$LOG"
  else
    pca_out=$(next_out "$stage/pca_only")
    pca_name=$(basename "$pca_out")
    run_collector "$pca_name" 4 40-59 pca_only "$split" "$id_seed" "$ood_seed" "$TARGET_PER_SOURCE" "$pca_out"
  fi
  record_part "$stage" "$pca_out"
  echo "$(date -Is) stage_pca_started_before_aggregate_diff_exit split=$split" >> "$LOG"

  # GPU5 remains occupied by all aggregate DiffDAgger runs until they exit.
  # The stage PCA above continues independently while this wait is in progress.
  wait_old_diff
  echo "$(date -Is) aggregate_diff_exited split=$split" >> "$LOG"
  if diff_out=$(active_stage_output "$stage/diffdagger"); then
    diff_name=$(basename "$diff_out")
    echo "$(date -Is) resuming name=$diff_name output=$diff_out" >> "$LOG"
  elif diff_out=$(completed_stage_output "$stage/diffdagger"); then
    diff_name=$(basename "$diff_out")
    echo "$(date -Is) reusing_completed name=$diff_name output=$diff_out" >> "$LOG"
  else
    diff_out=$(next_out "$stage/diffdagger")
    diff_name=$(basename "$diff_out")
    run_collector "$diff_name" 5 60-79 diffdagger "$split" "$((id_seed + 1000))" "$((ood_seed + 1000))" "$TARGET_PER_SOURCE" "$diff_out"
  fi
  record_part "$stage" "$diff_out"
  wait_collection_with_retries "$stage" "$stage/pca_only" "$pca_out" "$pca_name" 4 40-59 pca_only "$split" "$id_seed" "$ood_seed" || exit 1
  wait_collection_with_retries "$stage" "$stage/diffdagger" "$diff_out" "$diff_name" 5 60-79 diffdagger "$split" "$((id_seed + 1000))" "$((ood_seed + 1000))" || exit 1

  if fail_out=$(active_stage_output "$stage/failure_recovery"); then
    fail_name=$(basename "$fail_out")
  elif fail_out=$(completed_stage_output "$stage/failure_recovery"); then
    fail_name=$(basename "$fail_out")
  else
    fail_out=$(next_out "$stage/failure_recovery")
    fail_name=$(basename "$fail_out")
    run_collector "$fail_name" 4 40-59 failure_recovery "$split" "$((id_seed + 2000))" "$((ood_seed + 2000))" "$TARGET_PER_SOURCE" "$fail_out"
  fi
  if off_out=$(active_stage_output "$stage/offline_oracle"); then
    off_name=$(basename "$off_out")
  elif off_out=$(completed_stage_output "$stage/offline_oracle"); then
    off_name=$(basename "$off_out")
  else
    off_out=$(next_out "$stage/offline_oracle")
    off_name=$(basename "$off_out")
    run_collector "$off_name" cpu 80-99 offline_oracle "$split" "$((id_seed + 3000))" "$((ood_seed + 3000))" "$TARGET_PER_SOURCE" "$off_out"
  fi
  wait_collection_with_retries "$stage" "$stage/failure_recovery" "$fail_out" "$fail_name" 4 40-59 failure_recovery "$split" "$((id_seed + 2000))" "$((ood_seed + 2000))" || exit 1
  wait_collection_with_retries "$stage" "$stage/offline_oracle" "$off_out" "$off_name" cpu 80-99 offline_oracle "$split" "$((id_seed + 3000))" "$((ood_seed + 3000))" || exit 1
  echo "$(date -Is) stage_complete split=$split" >> "$LOG"
done

printf 'Three OOD stages collected for PCA, DiffDAgger, Failure-Recovery, and Offline BC.\n' > "$RUN/STAGE_COLLECTIONS_COMPLETE"
echo "$(date -Is) all_stage_collections_complete" >> "$LOG"
