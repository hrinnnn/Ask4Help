#!/usr/bin/env bash
set -u

# Fresh ID-base gate. This stage is intentionally independent of all old
# stage-specific collection and matched-update outputs.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=$ROOT/RLinf
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_ID_BASE_RUN:-$ROOT/results/open_drawer_failure_detection_v1/id_base_protocol_v2}
TRAIN=$RUN/sft_10000
EVAL=$RUN/eval_id_100
LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
LOG=$LOG_DIR/id_base_controller.log
DATA=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1
NORM=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
PI05=$ROOT/results/model_cache/pi05_base_pytorch_v1

mkdir -p "$LOG_DIR" "$PID_DIR"

alive() {
  [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

ckpt() {
  printf '%s/checkpoints/global_step_10000/actor/model_state_dict/full_weights.pt' "$TRAIN"
}

echo "$(date -Is) id_base_protocol_v2_started" >> "$LOG"
for path in "$DATA" "$NORM" "$PI05"; do
  [ -e "$path" ] || { echo "$(date -Is) missing_input=$path" >> "$LOG"; exit 2; }
done

if [ -f "$RUN/ID_BASE_VALIDATED" ]; then
  echo "$(date -Is) already_validated" >> "$LOG"
  exit 0
fi

if [ ! -f "$(ckpt)" ]; then
  mkdir -p "$TRAIN"
  export PYTHONPATH=$RL EMBODIED_PATH=$RL/examples/sft
  export ASK4HELP_RLINF_PLACEMENT=${ASK4HELP_RLINF_PLACEMENT:-7-7}
  export OPEN_DRAWER_ID_DATASET="$DATA"
  export OPEN_DRAWER_ID_NORM_STATS="$NORM"
  export OPEN_DRAWER_PI05_MODEL_PATH="$PI05"
  export OPEN_DRAWER_RUN_ROOT="$RUN"
  export OPEN_DRAWER_EXPERIMENT_NAME=sft_10000
  export HF_DATASETS_CACHE=$ROOT/runtime_cache/hf_datasets
  export HF_HOME=$ROOT/runtime_cache/hf_home
  export RAY_TMPDIR=${OPEN_DRAWER_RAY_TMPDIR:-/tmp/odray_open_drawer_id_base_v2} \
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
      >"$LOG_DIR/sft_10000.log" 2>&1 < /dev/null &
  echo $! > "$PID_DIR/sft_10000.pid"
  sleep 30
  if ! alive "$PID_DIR/sft_10000.pid"; then
    echo "$(date -Is) training_failed_during_startup" >> "$LOG"
    exit 1
  fi
fi

while alive "$PID_DIR/sft_10000.pid"; do
  echo "$(date -Is) training_alive=1" >> "$LOG"
  sleep 1800
done

if [ ! -f "$(ckpt)" ]; then
  echo "$(date -Is) training_missing_final_checkpoint" >> "$LOG"
  exit 1
fi
for step in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000; do
  [ -f "$TRAIN/checkpoints/global_step_$step/actor/model_state_dict/full_weights.pt" ] || {
    echo "$(date -Is) missing_checkpoint=$step" >> "$LOG"
    exit 1
  }
done

if [ ! -f "$EVAL/summary.json" ]; then
  mkdir -p "$EVAL"
  export PYTHONPATH=$RL
  export ASK4HELP_RLINF_ROOT=$RL
  taskset -c 140-159 env CUDA_VISIBLE_DEVICES=${OPEN_DRAWER_EVAL_GPU:-7} \
    PYTHONPATH=$PYTHONPATH ASK4HELP_RLINF_ROOT=$ASK4HELP_RLINF_ROOT \
    nohup "$PY" "$ROOT/tools/evaluate_open_drawer_id_pi05.py" \
      --checkpoint "$(ckpt)" --pi05-base "$PI05" --norm-stats "$NORM" \
      --output-dir "$EVAL" --episodes 100 --seed 90000 --split id \
      --execute-horizon 5 --max-episode-steps 400 \
      >"$LOG_DIR/eval_id_100.log" 2>&1 < /dev/null &
  echo $! > "$PID_DIR/eval_id_100.pid"
fi

while alive "$PID_DIR/eval_id_100.pid"; do
  echo "$(date -Is) id_eval_alive=1" >> "$LOG"
  sleep 600
done

"$PY" - "$EVAL/summary.json" "$RUN" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
run = Path(sys.argv[2])
if summary.get("episodes") != 100 or summary.get("successes", 0) < 80:
    (run / "ID_BASE_REJECTED").write_text(
        json.dumps({"summary": summary, "requirement": "successes >= 80/100"}, indent=2) + "\n"
    )
    raise SystemExit("ID base validation failed")
(run / "ID_BASE_VALIDATED").write_text(
    json.dumps({"checkpoint": str(run / "sft_10000"), "summary": summary}, indent=2) + "\n"
)
PY
printf 'fresh ID base passed independent 100-ID validation.\n' > "$RUN/ID_BASE_VALIDATED"
echo "$(date -Is) id_base_validated" >> "$LOG"
