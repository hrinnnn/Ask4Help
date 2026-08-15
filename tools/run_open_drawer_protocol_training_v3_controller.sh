#!/usr/bin/env bash
set -u

# Restart-tolerant formal training controller for the corrected OpenDrawer v3 protocol.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=$ROOT/RLinf
TOOLS=${OPEN_DRAWER_TOOLS:-/data/zhaozhixuan/Ask4Help/tools}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_PROTOCOL_RUN:-$ROOT/results/open_drawer_failure_detection_v1/stage_specific_protocol_v3}
COLLECTIONS=$RUN
TRAIN=$RUN/matched_budget_training_v3
LOG_DIR=$TRAIN/logs
PID_DIR=$TRAIN/pids
BASE=${OPEN_DRAWER_PROTOCOL_CKPT:-$ROOT/results/open_drawer_failure_detection_v1/id_base_protocol_v3_batch128/sft_10000/checkpoints/global_step_10000}
ID_DATA=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1
NORM=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
STEPS=${OPEN_DRAWER_PROTOCOL_TRAINING_STEPS:-10000}

mkdir -p "$LOG_DIR" "$PID_DIR"
LOG=$LOG_DIR/controller.log

alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }

echo "$(date -Is) protocol_training_controller_started" >> "$LOG"
while [ ! -f "$RUN/STAGE_COLLECTIONS_COMPLETE" ]; do
  if [ -f "$RUN/STAGE_COLLECTIONS_REJECTED" ]; then
    echo "$(date -Is) stage_collections_rejected" >> "$LOG"
    exit 1
  fi
  echo "$(date -Is) waiting_for_stage_collections" >> "$LOG"
  sleep 1800
done

for split in handle_ood grasp_ood goal_ood; do
  for method in pca_only diffdagger failure_recovery offline_oracle; do
    base="$COLLECTIONS/$split/$method"
    [ -f "$base/COLLECTION_COMPLETE" ] || { echo "$(date -Is) missing_collection=$base" >> "$LOG"; exit 2; }
    part=$(find "$base" -type f -path '*/lerobot_dataset/meta/info.json' -print -quit)
    [ -n "$part" ] || { echo "$(date -Is) missing_dataset=$base" >> "$LOG"; exit 2; }
  done
done

if ! "$PY" - "$RUN" "$TRAIN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
train = Path(sys.argv[2])
bad = []
report = {
    "format": "open_drawer_method_specific_collection_gate_v1",
    "rules": {
        "pca_only": {"pilot_accepted": 20, "pilot_ood_ratio_min": 0.8, "formal_accepted": 100, "formal_ood_ratio_min": 0.8},
        "diffdagger": {"formal_accepted": 100, "ood_ratio_min": None, "rule": "natural_ratio_only"},
        "failure_recovery": {"formal_accepted": 100, "ood_ratio_min": None, "rule": "natural_ratio_only"},
        "offline_oracle": {"formal_accepted": 100, "ood_ratio": 1.0},
    },
    "splits": {},
}
for split in ("handle_ood", "grasp_ood", "goal_ood"):
    report["splits"][split] = {}
    for method in ("pca_only", "diffdagger", "failure_recovery", "offline_oracle"):
        base = run / split / method
        parts = [Path(line.strip()) for line in (base / "collection_parts.txt").read_text().splitlines() if line.strip()]
        summaries = [json.loads((part / "summary.json").read_text()) for part in parts]
        accepted = sum(int(item.get("accepted", 0)) for item in summaries)
        accepted_id = sum(int(item.get("accepted_id", 0)) for item in summaries)
        accepted_ood = sum(int(item.get("accepted_ood", 0)) for item in summaries)
        accepted_rows = []
        for part in parts:
            accepted_rows.extend(
                json.loads(line)
                for line in (part / "accepted_experts.jsonl").read_text().splitlines()
                if line
            )
        formal_ood_ratio = accepted_ood / accepted if accepted else 0.0
        pilot_rows = accepted_rows[:20]
        pilot_ood = sum(row.get("source") == "ood" for row in pilot_rows)
        pilot_ratio = pilot_ood / len(pilot_rows) if pilot_rows else 0.0
        passed = accepted == 100
        if accepted != 100:
            bad.append({"split": split, "method": method, "reason": "accepted_not_100", "accepted": accepted})
        if method == "offline_oracle":
            if accepted_ood != 100 or accepted_id != 0:
                passed = False
                bad.append({"split": split, "method": method, "reason": "offline_not_ood_only", "accepted_id": accepted_id, "accepted_ood": accepted_ood})
        elif method == "pca_only":
            if len(pilot_rows) < 20 or pilot_ratio < 0.8 or formal_ood_ratio < 0.8:
                passed = False
                bad.append({"split": split, "method": method, "reason": "pca_ood_ratio_below_80_percent", "pilot_ood_ratio": pilot_ratio, "formal_ood_ratio": formal_ood_ratio})
        for part in parts:
            rows = [json.loads(line) for line in (part / "raw_attempts.jsonl").read_text().splitlines() if line]
            sources = [row.get("source") for row in rows]
            if method != "offline_oracle" and any(sources[i] == sources[i + 1] for i in range(len(sources) - 1)):
                passed = False
                bad.append({"split": split, "method": method, "reason": "raw_stream_not_alternating", "part": str(part)})
        report["splits"][split][method] = {
            "accepted": accepted,
            "accepted_id": accepted_id,
            "accepted_ood": accepted_ood,
            "formal_ood_ratio": formal_ood_ratio,
            "pilot_accepted": len(pilot_rows),
            "pilot_ood": pilot_ood,
            "pilot_ood_ratio": pilot_ratio,
            "pass": passed,
        }
(run / "collection_protocol_summary.json").write_text(json.dumps(report, indent=2) + "\n")
if bad:
    report["violations"] = bad
    (run / "collection_protocol_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (train / "COLLECTION_GATE_REJECTED").write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(json.dumps(bad))
(train / "COLLECTION_PROTOCOL_VALIDATED").write_text("all collection counts, split roles, and alternating streams validated\n")
PY
then
  echo "$(date -Is) collection_protocol_rejected" >> "$LOG"
  exit 1
fi

for split in handle_ood grasp_ood goal_ood; do
  output="$TRAIN/budget_prepared/$split"
  if [ ! -f "$output/budget_manifest.json" ]; then
    [ ! -e "$output.tmp" ] || { echo "$(date -Is) refusing_nonempty_budget_tmp=$output.tmp" >> "$LOG"; exit 1; }
    pca=$(find "$COLLECTIONS/$split/pca_only" -type f -path '*/lerobot_dataset/meta/info.json' -print -quit | sed 's#/meta/info.json##')
    diff=$(find "$COLLECTIONS/$split/diffdagger" -type f -path '*/lerobot_dataset/meta/info.json' -print -quit | sed 's#/meta/info.json##')
    fail=$(find "$COLLECTIONS/$split/failure_recovery" -type f -path '*/lerobot_dataset/meta/info.json' -print -quit | sed 's#/meta/info.json##')
    off=$(find "$COLLECTIONS/$split/offline_oracle" -type f -path '*/lerobot_dataset/meta/info.json' -print -quit | sed 's#/meta/info.json##')
    [ -n "$pca" ] && [ -n "$diff" ] && [ -n "$fail" ] && [ -n "$off" ] || { echo "$(date -Is) missing_budget_source split=$split" >> "$LOG"; exit 2; }
    if ! "$PY" "$TOOLS/prepare_open_drawer_matched_budget.py" \
      --output-root "$output.tmp" --pca-only "$pca" --diffdagger "$diff" \
      --failure-recovery "$fail" --offline-oracle "$off" \
      >"$LOG_DIR/budget_$split.log" 2>&1; then
      echo "$(date -Is) budget_prepare_failed split=$split" >> "$LOG"
      exit 1
    fi
    mv "$output.tmp" "$output"
  fi
done

launch_one() {
  local split=$1 method=$2 gpu=$3 cpuset=$4 wave_id=$5
  local dataset="$TRAIN/budget_prepared/$split/$method"
  local out="$TRAIN/$split/$method"
  local pidfile="$PID_DIR/${split}_${method}.pid"
  local log="$LOG_DIR/${split}_${method}.log"
  if [ -f "$out/TRAINING_COMPLETE" ]; then return 0; fi
  if [ -e "$out" ] && [ -n "$(find "$out" -mindepth 1 -print -quit 2>/dev/null)" ]; then
    echo "$(date -Is) refusing_nonempty_training_output=$out" >> "$LOG"
    return 1
  fi
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$gpu" ASK4HELP_RLINF_PLACEMENT=0-0 \
    OPEN_DRAWER_ID_DATASET="$ID_DATA" OPEN_DRAWER_EXPERT_DATASET="$dataset" \
    OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$BASE" \
    OPEN_DRAWER_EXPERIMENT_NAME="${split}_${method}_sft_${STEPS}" \
    OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_TRAIN_SEED=$((9400 + wave_id)) \
    HF_DATASETS_CACHE="$ROOT/runtime_cache/hf_datasets" HF_HOME="$ROOT/runtime_cache/hf_home" \
    RAY_TMPDIR="/data/odray_open_drawer_${split}_${method}" TMPDIR="$out/tmp" \
    PYTHONUNBUFFERED=1 taskset -c "$cpuset" nohup "$PY" "$RL/examples/sft/train_vla_sft.py" \
      --config-path "$RL/examples/sft/config" \
      --config-name open_drawer_retrieve_place_dagger_sft_openpi_pi05 \
      runner.max_steps="$STEPS" runner.save_interval=500 runner.resume_dir=null \
      >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  sleep 30
  alive "$pidfile" || { echo "$(date -Is) startup_failed split=$split method=$method" >> "$LOG"; return 1; }
  echo "$(date -Is) started split=$split method=$method gpu=$gpu pid=$(cat "$pidfile")" >> "$LOG"
}

wave=0
for split in handle_ood grasp_ood goal_ood; do
  for first_second in "pca_only diffdagger" "failure_recovery offline_oracle"; do
    set -- $first_second
    first=$1
    second=$2
    wave=$((wave + 1))
    launch_one "$split" "$first" 1 0-19 "$wave" || exit 1
    launch_one "$split" "$second" 2 20-39 "$((wave + 20))" || exit 1
    first_pid="$PID_DIR/${split}_${first}.pid"
    second_pid="$PID_DIR/${split}_${second}.pid"
    while alive "$first_pid" || alive "$second_pid"; do
      echo "$(date -Is) wave=$wave split=$split first=$first second=$second" >> "$LOG"
      sleep 1800
    done
    for method in "$first" "$second"; do
      out="$TRAIN/$split/$method"
      checkpoint="$out/checkpoints/global_step_${STEPS}/actor/model_state_dict/full_weights.pt"
      [ -f "$checkpoint" ] || { echo "$(date -Is) missing_final_checkpoint=$checkpoint" >> "$LOG"; exit 1; }
      printf 'OpenDrawer protocol training complete through step %s.\n' "$STEPS" > "$out/TRAINING_COMPLETE"
    done
  done
done

printf 'All three OpenDrawer OOD splits and four matched-budget methods trained.\n' > "$TRAIN/TRAINING_COMPLETE"
echo "$(date -Is) protocol_training_complete" >> "$LOG"

EVAL_CONTROLLER=${OPEN_DRAWER_FINAL_EVAL_CONTROLLER:-$TOOLS/run_open_drawer_protocol_final_eval_v3_controller.sh}
EVAL_PID=$PID_DIR/final_eval_v3_controller.pid
if ! alive "$EVAL_PID" && [ ! -f "$RUN/final_evaluation_v3/FINAL_EVAL_COMPLETE" ]; then
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_PYTHON="$PY" OPEN_DRAWER_TOOLS="$TOOLS" \
    OPEN_DRAWER_PROTOCOL_TRAINING_STEPS="$STEPS" OPEN_DRAWER_PROTOCOL_RUN="$RUN" \
    nohup bash "$EVAL_CONTROLLER" >"$LOG_DIR/final_eval_controller_stdout.log" 2>&1 < /dev/null &
  echo $! > "$EVAL_PID"
  echo "$(date -Is) final_eval_controller_started pid=$(cat "$EVAL_PID")" >> "$LOG"
fi
