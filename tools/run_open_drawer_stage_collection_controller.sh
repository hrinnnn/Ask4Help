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
  local name=$1 gpu=$2 cpuset=$3 method=$4 split=$5 id_seed=$6 ood_seed=$7 out=$8
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
  cmd+=(--output-root "$out" --target-per-source 50 --max-attempts 800
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
OLD_DIFF="$SIDE/pids/diffdagger_collection_successful_id_v1.pid"

# The aggregate DiffDAgger run can be long.  Start the stage PCA as soon as
# the aggregate PCA releases GPU4, then let it overlap with the tail of the
# aggregate DiffDAgger run on GPU5.  This keeps the four-GPU budget useful
# without duplicating a stage or touching the main training GPUs.
while alive "$OLD_PCA"; do sleep 300; done
echo "$(date -Is) aggregate_pca_exited" >> "$LOG"

stage_index=0
for split in handle_ood grasp_ood goal_ood; do
  stage_index=$((stage_index + 1))
  stage=$RUN/$split
  mkdir -p "$stage"
  id_seed=$((100000 + stage_index * 4000))
  ood_seed=$((110000 + stage_index * 4000))

  pca_out=$(next_out "$stage/pca_only")
  diff_out=$(next_out "$stage/diffdagger")
  pca_name=$(basename "$pca_out")
  diff_name=$(basename "$diff_out")
  run_collector "$pca_name" 4 40-59 pca_only "$split" "$id_seed" "$ood_seed" "$pca_out"
  echo "$(date -Is) stage_pca_started_before_aggregate_diff_exit split=$split" >> "$LOG"

  # GPU5 remains occupied by the aggregate DiffDAgger until it exits.  The
  # stage PCA above continues independently while this wait is in progress.
  while alive "$OLD_DIFF"; do sleep 300; done
  echo "$(date -Is) aggregate_diff_exited split=$split" >> "$LOG"
  run_collector "$diff_name" 5 60-79 diffdagger "$split" "$((id_seed + 1000))" "$((ood_seed + 1000))" "$diff_out"
  wait_job "$pca_name" "$pca_out" || exit 1
  wait_job "$diff_name" "$diff_out" || exit 1

  fail_out=$(next_out "$stage/failure_recovery")
  off_out=$(next_out "$stage/offline_oracle")
  fail_name=$(basename "$fail_out")
  off_name=$(basename "$off_out")
  run_collector "$fail_name" 4 40-59 failure_recovery "$split" "$((id_seed + 2000))" "$((ood_seed + 2000))" "$fail_out"
  run_collector "$off_name" cpu 80-99 offline_oracle "$split" "$((id_seed + 3000))" "$((ood_seed + 3000))" "$off_out"
  wait_job "$fail_name" "$fail_out" || exit 1
  wait_job "$off_name" "$off_out" || exit 1
  echo "$(date -Is) stage_complete split=$split" >> "$LOG"
done

printf 'Three OOD stages collected for PCA, DiffDAgger, Failure-Recovery, and Offline BC.\n' > "$RUN/STAGE_COLLECTIONS_COMPLETE"
echo "$(date -Is) all_stage_collections_complete" >> "$LOG"
