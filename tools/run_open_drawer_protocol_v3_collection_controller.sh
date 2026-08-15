#!/usr/bin/env bash
set -u

# Protocol-v3 collection controller. It uses one independent alternating stream
# per OOD split and stops each gated method at total accepted=100.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
TOOLS=${OPEN_DRAWER_TOOLS:-/data/zhaozhixuan/Ask4Help/tools}
RLINF=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
RUN=${OPEN_DRAWER_PROTOCOL_RUN:-$ROOT/results/open_drawer_failure_detection_v1/stage_specific_protocol_v3}
GATE_DIR=$RUN/protocol_gates
LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
LOG=$LOG_DIR/controller.log

TARGET_TOTAL_ACCEPTED=${OPEN_DRAWER_TARGET_TOTAL_ACCEPTED:-100}
MAX_ATTEMPTS=${OPEN_DRAWER_MAX_ATTEMPTS:-1600}
CKPT=${OPEN_DRAWER_PROTOCOL_CKPT:-$RUN/id_base/sft_10000/checkpoints/global_step_10000}
NORM=${OPEN_DRAWER_PROTOCOL_NORM:-$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1}
DET=${OPEN_DRAWER_PROTOCOL_DETECTOR_ASSETS:-$RUN/id_only_detector_assets/detector_assets.pt}
PCA_CAL=${OPEN_DRAWER_PROTOCOL_PCA_CALIBRATION:-$RUN/id_only_calibration/pca_calibration.json}
DIFF_CAL=${OPEN_DRAWER_PROTOCOL_DIFF_CALIBRATION:-$RUN/id_only_calibration/diff_calibration.json}
PI05=$ROOT/results/model_cache/pi05_base_pytorch_v1
PLANNER_PY=${PANDA_PLANNER_PYTHON:-/data/zhaozhixuan/simplerenv_ms3/env/bin/python}
PLANNER_SITE=${PANDA_PLANNER_SITE_PACKAGES:-/data/zhaozhixuan/simplerenv_ms3/env/lib/python3.10/site-packages}

mkdir -p "$LOG_DIR" "$PID_DIR" "$GATE_DIR"

alive() {
  [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

require_gate() {
  local path=$1 label=$2
  if [ ! -f "$path" ]; then
    echo "$(date -Is) waiting_for_gate label=$label path=$path" >> "$LOG"
    return 1
  fi
  return 0
}

run_collector() {
  local name=$1 gpu=$2 cpuset=$3 method=$4 split=$5 id_seed=$6 ood_seed=$7 target=$8 offset=$9 out=${10}
  local log=$LOG_DIR/${name}.log pidfile=$PID_DIR/${name}.pid
  local cuda=$gpu
  if [ "$gpu" = "cpu" ]; then cuda=""; fi
  local -a cmd=("$PY" "$TOOLS/collect_open_drawer_dagger.py"
    --method "$method" --checkpoint "$CKPT" --pi05-base "$PI05" --norm-stats "$NORM"
    --output-root "$out" --target-total-accepted "$target" --max-attempts "$MAX_ATTEMPTS"
    --attempt-offset "$offset" --id-seed-start "$id_seed" --ood-seed-start "$ood_seed"
    --ood-split "$split" --execute-horizon 5 --max-policy-steps 240)
  if [ "$method" = "offline_oracle" ]; then
    cmd+=(--single-source-split "$split")
  fi
  if [ "$method" = "diffdagger" ]; then
    cmd+=(--detector-assets "$DET" --diff-calibration "$DIFF_CAL"
      --diff-alpha 0.95 --diff-patience 2 --diff-timesteps 16 --diff-noise-samples 1)
  elif [ "$method" = "pca_only" ]; then
    cmd+=(--detector-assets "$DET" --thresholds "$PCA_CAL" --pca-layer vlm_bridge_final_mean)
  elif [ "$method" = "failure_recovery" ]; then
    cmd+=(--detector-assets "$DET" --thresholds "$PCA_CAL")
  fi
  env ASK4HELP_RLINF_ROOT="$RLINF" \
    PANDA_PLANNER_PYTHON="$PLANNER_PY" \
    PANDA_PLANNER_SITE_PACKAGES="$PLANNER_SITE" \
    CUDA_VISIBLE_DEVICES="$cuda" \
    OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 PYTHONUNBUFFERED=1 \
    taskset -c "$cpuset" nohup "${cmd[@]}" >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "$(date -Is) started name=$name method=$method split=$split gpu=$gpu offset=$offset target=$target output=$out" >> "$LOG"
}

summary_value() {
  local path=$1 key=$2
  "$PY" - "$path" "$key" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(int(payload.get(sys.argv[2], 0)))
PY
}

parts_total() {
  local parts=$1 key=$2 total=0 value
  while IFS= read -r out; do
    [ -n "$out" ] || continue
    [ -f "$out/summary.json" ] || continue
    value=$(summary_value "$out/summary.json" "$key") || return 1
    total=$((total + value))
  done < "$parts"
  printf '%s\n' "$total"
}

parts_attempts() {
  local parts=$1 total=0 value
  while IFS= read -r out; do
    [ -n "$out" ] || continue
    [ -f "$out/summary.json" ] || continue
    value=$(summary_value "$out/summary.json" attempts) || return 1
    total=$((total + value))
  done < "$parts"
  printf '%s\n' "$total"
}

wait_part() {
  local name=$1 out=$2 pidfile=$PID_DIR/${name}.pid
  while alive "$pidfile"; do sleep 300; done
  [ -f "$out/summary.json" ]
}

run_method() {
  local split=$1 method=$2 gpu=$3 cpuset=$4 id_seed=$5 ood_seed=$6
  local stage=$RUN/$split base=$stage/$method parts=$base/collection_parts.txt
  local total=0 offset=0 part=0 out name
  mkdir -p "$base"
  touch "$parts"
  total=$(parts_total "$parts" accepted 2>/dev/null || printf '0')
  offset=$(parts_attempts "$parts" 2>/dev/null || printf '0')
  if [ "$total" -ge "$TARGET_TOTAL_ACCEPTED" ]; then
    touch "$base/COLLECTION_COMPLETE"
    return 0
  fi
  while [ "$total" -lt "$TARGET_TOTAL_ACCEPTED" ]; do
    part=$((part + 1))
    out=$base/part_$(printf '%03d' "$part")
    while [ -e "$out" ]; do
      part=$((part + 1))
      out=$base/part_$(printf '%03d' "$part")
    done
    name="${split}_${method}_part_$(printf '%03d' "$part")"
    run_collector "$name" "$gpu" "$cpuset" "$method" "$split" "$id_seed" "$ood_seed" "$((TARGET_TOTAL_ACCEPTED - total))" "$offset" "$out"
    printf '%s\n' "$out" >> "$parts"
    if ! wait_part "$name" "$out"; then
      echo "$(date -Is) part_failed split=$split method=$method output=$out" >> "$LOG"
      return 1
    fi
    next_total=$(parts_total "$parts" accepted 2>/dev/null || printf '0')
    next_offset=$(parts_attempts "$parts" 2>/dev/null || printf '0')
    if [ "$next_total" -le "$total" ] && [ "$next_offset" -le "$offset" ]; then
      echo "$(date -Is) no_progress split=$split method=$method" >> "$LOG"
      return 1
    fi
    total=$next_total
    offset=$next_offset
    echo "$(date -Is) part_complete split=$split method=$method accepted=$total raw_attempts=$offset" >> "$LOG"
  done
  printf 'total accepted target reached; accepted ID/OOD counts remain observational\n' > "$base/COLLECTION_COMPLETE"
}

echo "$(date -Is) protocol_v3_collection_controller_started" >> "$LOG"
require_gate "$GATE_DIR/PROTOCOL_GATES_COMPLETE" protocol_gates || exit 2
for path in "$CKPT" "$NORM" "$DET" "$PCA_CAL" "$DIFF_CAL"; do
  [ -e "$path" ] || { echo "$(date -Is) missing_input path=$path" >> "$LOG"; exit 2; }
done

stage_index=0
for split in handle_ood grasp_ood goal_ood; do
  stage_index=$((stage_index + 1))
  id_seed=$((200000 + stage_index * 10000))
  ood_seed=$((300000 + stage_index * 10000))
  run_method "$split" pca_only 4 40-59 "$id_seed" "$ood_seed" &
  pca_pid=$!
  run_method "$split" diffdagger 5 60-79 "$id_seed" "$ood_seed" &
  diff_pid=$!
  wait "$pca_pid" || exit 1
  wait "$diff_pid" || exit 1
  run_method "$split" failure_recovery 4 40-59 "$id_seed" "$ood_seed" &
  fail_pid=$!
  run_method "$split" offline_oracle cpu 80-99 "$id_seed" "$ood_seed" &
  offline_pid=$!
  wait "$fail_pid" || exit 1
  wait "$offline_pid" || exit 1
  echo "$(date -Is) stage_collection_complete split=$split" >> "$LOG"
done

if ! "$PY" - "$RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
methods = ("pca_only", "diffdagger", "failure_recovery", "offline_oracle")
report = {
    "format": "open_drawer_protocol_v3_collection_gate",
    "rules": {
        "pca_only": {"accepted": 100, "pilot_accepted": 20, "pilot_ood_ratio_min": 0.8, "formal_ood_ratio_min": 0.8},
        "diffdagger": {"accepted": 100, "ood_ratio": "natural"},
        "failure_recovery": {"accepted": 100, "ood_ratio": "natural"},
        "offline_oracle": {"accepted": 100, "accepted_ood": 100, "accepted_id": 0},
    },
    "splits": {},
}
violations = []
for split in ("handle_ood", "grasp_ood", "goal_ood"):
    report["splits"][split] = {}
    for method in methods:
        base = run / split / method
        parts_file = base / "collection_parts.txt"
        parts = [Path(line.strip()) for line in parts_file.read_text().splitlines() if line.strip()]
        summaries = [json.loads((part / "summary.json").read_text()) for part in parts]
        accepted_rows = []
        raw_rows = []
        for part in parts:
            accepted_rows.extend(json.loads(line) for line in (part / "accepted_experts.jsonl").read_text().splitlines() if line)
            raw_rows.extend(json.loads(line) for line in (part / "raw_attempts.jsonl").read_text().splitlines() if line)
        accepted = sum(int(item.get("accepted", 0)) for item in summaries)
        accepted_id = sum(int(item.get("accepted_id", 0)) for item in summaries)
        accepted_ood = sum(int(item.get("accepted_ood", 0)) for item in summaries)
        pilot = accepted_rows[:20]
        pilot_ood_ratio = sum(row.get("source") == "ood" for row in pilot) / len(pilot) if pilot else 0.0
        formal_ood_ratio = accepted_ood / accepted if accepted else 0.0
        passed = accepted == 100
        if method == "offline_oracle":
            passed = passed and accepted_id == 0 and accepted_ood == 100
        elif method == "pca_only":
            passed = passed and len(pilot) >= 20 and pilot_ood_ratio >= 0.8 and formal_ood_ratio >= 0.8
        if method != "offline_oracle":
            sources = [row.get("source") for row in raw_rows]
            passed = passed and all(a != b for a, b in zip(sources, sources[1:]))
        if not passed:
            violations.append({"split": split, "method": method, "accepted": accepted, "accepted_id": accepted_id, "accepted_ood": accepted_ood, "pilot_ood_ratio": pilot_ood_ratio, "formal_ood_ratio": formal_ood_ratio})
        report["splits"][split][method] = {
            "accepted": accepted, "accepted_id": accepted_id, "accepted_ood": accepted_ood,
            "pilot_accepted": len(pilot), "pilot_ood_ratio": pilot_ood_ratio,
            "formal_ood_ratio": formal_ood_ratio, "raw_attempts": len(raw_rows), "pass": passed,
        }
(run / "collection_protocol_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if violations:
    report["violations"] = violations
    (run / "collection_protocol_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (run / "STAGE_COLLECTIONS_REJECTED").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    raise SystemExit("collection gate failed")
PY
then
  echo "$(date -Is) collection_protocol_rejected" >> "$LOG"
  exit 1
fi

printf 'Protocol-v3 stage collections complete.\n' > "$RUN/STAGE_COLLECTIONS_COMPLETE"
echo "$(date -Is) all_stage_collections_complete" >> "$LOG"

TRAIN_CONTROLLER=${OPEN_DRAWER_TRAINING_CONTROLLER:-$TOOLS/run_open_drawer_protocol_training_v3_controller.sh}
TRAIN_PID=$PID_DIR/protocol_training_v3_controller.pid
if ! alive "$TRAIN_PID" && [ ! -f "$RUN/matched_budget_training_v3/TRAINING_COMPLETE" ]; then
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_PYTHON="$PY" OPEN_DRAWER_TOOLS="$TOOLS" \
    OPEN_DRAWER_PROTOCOL_CKPT="$CKPT" OPEN_DRAWER_PROTOCOL_NORM="$NORM" OPEN_DRAWER_PROTOCOL_RUN="$RUN" \
    OPEN_DRAWER_PROTOCOL_TRAINING_STEPS=10000 \
    nohup bash "$TRAIN_CONTROLLER" >"$LOG_DIR/protocol_training_v3_controller_stdout.log" 2>&1 < /dev/null &
  echo $! > "$TRAIN_PID"
  echo "$(date -Is) training_controller_started pid=$(cat "$TRAIN_PID")" >> "$LOG"
fi
