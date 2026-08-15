#!/usr/bin/env bash
set -u

# Fresh ID-base gate. This stage is independent of all old collection and
# matched-update outputs. It evaluates the first usable base checkpoint at
# 2k, then at later checkpoints only when the previous gate was not met.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=$ROOT/RLinf
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_ID_BASE_RUN:-$ROOT/results/open_drawer_failure_detection_v1/id_base_protocol_v2}
TRAIN=$RUN/sft_10000
EVAL_ROOT=$RUN/eval_id_100_by_step
LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
LOG=$LOG_DIR/id_base_controller.log
DATA=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1
NORM=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
PI05=$ROOT/results/model_cache/pi05_base_pytorch_v1
EVALUATOR=${OPEN_DRAWER_EVALUATOR:-/data/zhaozhixuan/Ask4Help/tools/evaluate_open_drawer_id_pi05.py}
PLACEMENT=${OPEN_DRAWER_RLINF_PLACEMENT:-7-7}

mkdir -p "$LOG_DIR" "$PID_DIR"

alive() {
  [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

ckpt() {
  local step=$1
  printf '%s/checkpoints/global_step_%s/actor/model_state_dict/full_weights.pt' "$TRAIN" "$step"
}

checkpoint_complete() {
  [ -f "$(ckpt "$1")" ]
}

stop_training() {
  local pidfile=$PID_DIR/sft_10000.pid
  if ! alive "$pidfile"; then
    return 0
  fi
  local pid
  pid=$(cat "$pidfile")
  echo "$(date -Is) stopping_base_training pid=$pid" >> "$LOG"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 120); do
    alive "$pidfile" || return 0
    sleep 1
  done
  echo "$(date -Is) force_stopping_base_training pid=$pid" >> "$LOG"
  kill -KILL "$pid" 2>/dev/null || true
}

echo "$(date -Is) id_base_protocol_v2_started" >> "$LOG"
for path in "$DATA" "$NORM" "$PI05"; do
  [ -e "$path" ] || { echo "$(date -Is) missing_input=$path" >> "$LOG"; exit 2; }
done

if [ -f "$RUN/ID_BASE_VALIDATED" ]; then
  echo "$(date -Is) already_validated" >> "$LOG"
  exit 0
fi

# Reattach to an already-running training process after a controller restart.
if ! alive "$PID_DIR/sft_10000.pid" && ! checkpoint_complete 10000; then
  mkdir -p "$TRAIN"
  export PYTHONPATH=$RL EMBODIED_PATH=$RL/examples/sft
  export ASK4HELP_RLINF_PLACEMENT=$PLACEMENT
  export OPEN_DRAWER_ID_DATASET="$DATA"
  export OPEN_DRAWER_ID_NORM_STATS="$NORM"
  export OPEN_DRAWER_PI05_MODEL_PATH="$PI05"
  export OPEN_DRAWER_RUN_ROOT="$RUN"
  export OPEN_DRAWER_EXPERIMENT_NAME=sft_10000
  export HF_DATASETS_CACHE=$ROOT/runtime_cache/hf_datasets
  export HF_HOME=$ROOT/runtime_cache/hf_home
  export RAY_TMPDIR=${OPEN_DRAWER_RAY_TMPDIR:-$ROOT/runtime_cache/ray} \
    TMPDIR=${OPEN_DRAWER_TMPDIR:-$RUN/tmp} PYTHONUNBUFFERED=1
  mkdir -p "$RAY_TMPDIR" "$TMPDIR"
  taskset -c 140-159 env PYTHONPATH=$PYTHONPATH EMBODIED_PATH=$EMBODIED_PATH \
    ASK4HELP_RLINF_PLACEMENT=$ASK4HELP_RLINF_PLACEMENT \
    OPEN_DRAWER_ID_DATASET=$OPEN_DRAWER_ID_DATASET \
    OPEN_DRAWER_ID_NORM_STATS=$OPEN_DRAWER_ID_NORM_STATS \
    OPEN_DRAWER_PI05_MODEL_PATH=$OPEN_DRAWER_PI05_MODEL_PATH \
    OPEN_DRAWER_RUN_ROOT=$OPEN_DRAWER_RUN_ROOT \
    OPEN_DRAWER_EXPERIMENT_NAME=$OPEN_DRAWER_EXPERIMENT_NAME \
    HF_DATASETS_CACHE=$HF_DATASETS_CACHE HF_HOME=$HF_HOME \
    RAY_TMPDIR=$RAY_TMPDIR TMPDIR=$TMPDIR PYTHONUNBUFFERED=1 \
    nohup "$PY" "$RL/examples/sft/train_vla_sft.py" \
      --config-path "$RL/examples/sft/config" \
      --config-name open_drawer_retrieve_place_id_sft_openpi_pi05 \
      runner.max_steps=10000 runner.save_interval=500 \
      actor.micro_batch_size=32 actor.global_batch_size=128 \
      >"$LOG_DIR/sft_10000.log" 2>&1 < /dev/null &
  echo $! > "$PID_DIR/sft_10000.pid"
  sleep 30
  if ! alive "$PID_DIR/sft_10000.pid"; then
    echo "$(date -Is) training_failed_during_startup" >> "$LOG"
    exit 1
  fi
fi

# Evaluate only at these checkpoints. Loss is never used as the base-policy
# gate; each decision is made from an independent 100-ID pure-policy rollout.
CHECKPOINT_STEPS=(2000 4000 6000 8000 10000)
for step in "${CHECKPOINT_STEPS[@]}"; do
  while ! checkpoint_complete "$step"; do
    if ! alive "$PID_DIR/sft_10000.pid"; then
      echo "$(date -Is) training_stopped_before_checkpoint=$step" >> "$LOG"
      exit 1
    fi
    echo "$(date -Is) waiting_for_checkpoint=$step training_alive=1" >> "$LOG"
    sleep 300
  done

  eval_dir=$EVAL_ROOT/step_$step
  eval_pidfile=$PID_DIR/eval_id_100_step_${step}.pid
  eval_log=$LOG_DIR/eval_id_100_step_${step}.log
  mkdir -p "$eval_dir"
  if [ ! -f "$eval_dir/summary.json" ] && ! alive "$eval_pidfile"; then
    export PYTHONPATH=$RL
    export ASK4HELP_RLINF_ROOT=$RL
    taskset -c 140-159 env CUDA_VISIBLE_DEVICES=${OPEN_DRAWER_EVAL_GPU:-7} \
      PYTHONPATH=$PYTHONPATH ASK4HELP_RLINF_ROOT=$ASK4HELP_RLINF_ROOT \
      nohup "$PY" "$EVALUATOR" \
        --checkpoint "$(ckpt "$step")" --pi05-base "$PI05" --norm-stats "$NORM" \
        --output-dir "$eval_dir" --episodes 100 --seed 90000 --split id \
        --execute-horizon 5 --max-episode-steps 400 \
        >"$eval_log" 2>&1 < /dev/null &
    echo $! > "$eval_pidfile"
    echo "$(date -Is) id_eval_started step=$step" >> "$LOG"
  fi
  while alive "$eval_pidfile"; do
    echo "$(date -Is) id_eval_alive=1 step=$step" >> "$LOG"
    sleep 600
  done
  if [ ! -f "$eval_dir/summary.json" ]; then
    echo "$(date -Is) id_eval_missing_summary step=$step" >> "$LOG"
    exit 1
  fi

  "$PY" - "$eval_dir/summary.json" "$RUN" "$step" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
run = Path(sys.argv[2])
step = int(sys.argv[3])
payload = {
    "checkpoint_step": step,
    "summary": summary,
    "requirement": "episodes == 100 and successes >= 80",
}
(run / f"ID_BASE_STEP_{step}_EVAL.json").write_text(json.dumps(payload, indent=2) + "\n")
if summary.get("episodes") != 100:
    raise SystemExit(f"ID evaluation has {summary.get('episodes')} episodes, expected 100")
if summary.get("successes", 0) < 80:
    (run / f"ID_BASE_STEP_{step}_REJECTED").write_text(json.dumps(payload, indent=2) + "\n")
    raise SystemExit(10)
(run / "ID_BASE_VALIDATED").write_text(json.dumps({
    "base_step": step,
    "checkpoint": str(run / "sft_10000" / "checkpoints" / f"global_step_{step}"),
    "summary": summary,
}, indent=2) + "\n")
PY
  result=$?
  if [ "$result" -eq 0 ]; then
    stop_training
    echo "$(date -Is) id_base_validated step=$step" >> "$LOG"
    exit 0
  fi
  if [ "$result" -eq 10 ]; then
    if [ "$step" -ge 4000 ]; then
      stop_training
      printf '%s\n' "ID base failed the required >=80% independent ID gate at step $step; no OOD or gated update may start." \
        > "$RUN/ID_BASE_PROTOCOL_FAILED"
      echo "$(date -Is) id_base_step_rejected_final step=$step; stopping_diagnostic" >> "$LOG"
      exit 10
    fi
    echo "$(date -Is) id_base_step_rejected step=$step; continuing" >> "$LOG"
    continue
  fi
  echo "$(date -Is) id_eval_invalid step=$step" >> "$LOG"
  exit 1
done

echo "$(date -Is) id_base_rejected_all_checkpoints" >> "$LOG"
printf 'no checkpoint met the independent ID success gate.\n' > "$RUN/ID_BASE_REJECTED"
exit 1
