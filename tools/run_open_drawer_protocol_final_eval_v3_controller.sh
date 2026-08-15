#!/usr/bin/env bash
set -u

# Final evaluation controller for the three independent OpenDrawer OOD stages.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=$ROOT/RLinf
TOOLS=${OPEN_DRAWER_TOOLS:-/data/zhaozhixuan/Ask4Help/tools}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_PROTOCOL_RUN:-$ROOT/results/open_drawer_failure_detection_v1/stage_specific_protocol_v3}
TRAIN=$RUN/matched_budget_training_v3
EVAL=$RUN/final_evaluation_v3
LOG_DIR=$EVAL/logs
PID_DIR=$EVAL/pids
NORM=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
PI05=$ROOT/results/model_cache/pi05_base_pytorch_v1
STEPS=${OPEN_DRAWER_PROTOCOL_TRAINING_STEPS:-10000}

mkdir -p "$LOG_DIR" "$PID_DIR"
LOG=$LOG_DIR/controller.log
alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

echo "$(date -Is) final_eval_controller_started" >> "$LOG"
while [ ! -f "$TRAIN/TRAINING_COMPLETE" ]; do
  echo "$(date -Is) waiting_for_training" >> "$LOG"
  sleep 1800
done

split_seed() {
  case "$1" in
    id) printf '90000' ;;
    handle_ood) printf '91000' ;;
    grasp_ood) printf '92000' ;;
    goal_ood) printf '93000' ;;
  esac
}

launch_eval() {
  local split=$1 method=$2 gpu=$3 cpuset=$4
  local checkpoint="$TRAIN/$split/$method/checkpoints/global_step_${STEPS}"
  local out="$EVAL/$split/$method"
  local pidfile="$PID_DIR/${split}_${method}.pid"
  local log="$LOG_DIR/${split}_${method}.log"
  [ -f "$checkpoint/actor/model_state_dict/full_weights.pt" ] || { echo "$(date -Is) missing_checkpoint=$checkpoint" >> "$LOG"; return 1; }
  if [ -f "$out/EVAL_COMPLETE" ]; then return 0; fi
  if [ -e "$out" ] && [ -n "$(find "$out" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "$(date -Is) refusing_nonempty_eval_output=$out" >> "$LOG"
    return 1
  fi
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$gpu" ASK4HELP_RLINF_ROOT="$RL" \
    PYTHONPATH="$ROOT:$RL" OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
    PYTHONUNBUFFERED=1 taskset -c "$cpuset" nohup "$PY" "$TOOLS/run_open_drawer_split_eval.py" \
      --python "$PY" --evaluator "$TOOLS/evaluate_open_drawer_id_pi05.py" \
      --checkpoint "$checkpoint" --pi05-base "$PI05" --norm-stats "$NORM" \
      --output-dir "$out" --split "$split" --seed "$(split_seed "$split")" \
      --episodes 100 --execute-horizon 5 --max-episode-steps 400 \
      >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$(date -Is) eval_started split=$split method=$method gpu=$gpu pid=$(cat "$pidfile")" >> "$LOG"
}

for split in id handle_ood grasp_ood goal_ood; do
  for method in pca_only diffdagger failure_recovery offline_oracle; do
    wave=${wave:-0}
    wave=$((wave + 1))
    launch_eval "$split" "$method" 1 0-19 || exit 1
    first_pid="$PID_DIR/${split}_${method}.pid"
    while alive "$first_pid"; do
      echo "$(date -Is) eval_alive split=$split method=$method" >> "$LOG"
      sleep 600
    done
    [ -f "$EVAL/$split/$method/EVAL_COMPLETE" ] || { echo "$(date -Is) eval_incomplete split=$split method=$method" >> "$LOG"; exit 1; }
  done
done

"$PY" - "$EVAL" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = ("id", "handle_ood", "grasp_ood", "goal_ood")
methods = ("pca_only", "diffdagger", "failure_recovery", "offline_oracle")
comparison = {"format": "open_drawer_protocol_final_evaluation_v3", "methods": {}}
for method in methods:
    rows = {}
    for split in splits:
        path = root / split / method / "summary.json"
        payload = json.loads(path.read_text())
        if payload.get("episodes") != 100 or len(list((path.parent / "videos").glob("*.mp4"))) != 100:
            raise SystemExit(f"incomplete evaluation: {path}")
        rows[split] = {
            key: payload.get(key)
            for key in ("episodes", "successes", "success_rate", "drawer_opened_rate",
                        "grasp_rate", "lift_rate", "in_target_rate", "execute_horizon",
                        "max_episode_steps", "seed_start")
        }
    comparison["methods"][method] = rows
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
lines = [
    "# OpenDrawer corrected protocol final evaluation",
    "",
    "| Method | ID | Handle OOD | Grasp OOD | Goal OOD |",
    "|---|---:|---:|---:|---:|",
]
for method in methods:
    rows = comparison["methods"][method]
    rates = [f"{100.0 * float(rows[split]['success_rate']):.1f}%" for split in splits]
    lines.append(f"| {method} | " + " | ".join(rates) + " |")
(root / "comparison.md").write_text("\n".join(lines) + "\n")
(root / "FINAL_EVAL_COMPLETE").write_text("all 16 split/method evaluations verified with 100 episodes and 100 videos each\n")
PY
echo "$(date -Is) final_eval_complete" >> "$LOG"
