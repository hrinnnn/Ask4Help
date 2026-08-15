#!/usr/bin/env bash
set -u

# Gate controller for the corrected OpenDrawer protocol v3. It waits for the
# new ID base, validates Oracle coverage, rebuilds ID-only detector assets, and
# only then starts the independent stage collector.
ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
TOOLS=${OPEN_DRAWER_TOOLS:-/data/zhaozhixuan/Ask4Help/tools}
RLINF=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
RUN=${OPEN_DRAWER_PROTOCOL_RUN:-$ROOT/results/open_drawer_failure_detection_v1/stage_specific_protocol_v3}
ID_BASE_RUN=${OPEN_DRAWER_ID_BASE_RUN:-$ROOT/results/open_drawer_failure_detection_v1/id_base_protocol_v3_batch128}
CKPT=${OPEN_DRAWER_PROTOCOL_CKPT:-$ID_BASE_RUN/sft_10000}
NORM=$ROOT/results/id_policy_training_v1/norm_stats_open_drawer_id_raw_v1
PI05=$ROOT/results/model_cache/pi05_base_pytorch_v1
ID_DATA=$ROOT/results/id_oracle_collection_v1/lerobot_datasets/open_drawer_retrieve_place/id_oracle_128_retry1_v1
ORACLE=$RUN/oracle_validation_v2
ASSET=$RUN/id_only_detector_assets
PCA_CAL=$RUN/id_only_calibration/pca
DIFF_RAW=$RUN/id_only_calibration/diff_raw
DIFF_CAL=$RUN/id_only_calibration/diff_calibration.json
LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
LOG=$LOG_DIR/gate_controller.log

mkdir -p "$LOG_DIR" "$PID_DIR" "$ORACLE" "$ASSET" "$PCA_CAL" "$RUN/protocol_gates"

alive() {
  [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null
}

echo "$(date -Is) gate_controller_started" >> "$LOG"
while [ ! -f "$ID_BASE_RUN/ID_BASE_VALIDATED" ]; do
  if [ -f "$ID_BASE_RUN/ID_BASE_REJECTED" ] || [ -f "$ID_BASE_RUN/ID_BASE_PROTOCOL_FAILED" ]; then
    echo "$(date -Is) id_base_protocol_failed; stopping_before_ood" >> "$LOG"
    exit 10
  fi
  echo "$(date -Is) waiting_for_id_base path=$ID_BASE_RUN/ID_BASE_VALIDATED" >> "$LOG"
  sleep 1800
done

BASE_STEP="$($PY - "$ID_BASE_RUN/ID_BASE_VALIDATED" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(int(payload.get("base_step", 10000)))
PY
)" || exit 1
BASE_CKPT="$($PY - "$ID_BASE_RUN/ID_BASE_VALIDATED" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("checkpoint", ""))
PY
)" || exit 1
[ -n "$BASE_CKPT" ] || BASE_CKPT=$CKPT/checkpoints/global_step_${BASE_STEP}
echo "$(date -Is) id_base_validated_step=$BASE_STEP checkpoint=$BASE_CKPT" >> "$LOG"
for path in "$BASE_CKPT" "$NORM" "$PI05" "$ID_DATA"; do
  [ -e "$path" ] || { echo "$(date -Is) missing_input=$path" >> "$LOG"; exit 2; }
done

run_oracle() {
  local split=$1 index=$2 cpuset=$3
  local out=$ORACLE/$split log=$LOG_DIR/oracle_$split.log pidfile=$PID_DIR/oracle_$split.pid
  [ -f "$out/COLLECTION_COMPLETE" ] && return 0
  if [ -e "$out" ] && [ -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
    echo "$(date -Is) refusing_nonempty_oracle_output=$out" >> "$LOG"
    return 1
  fi
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="" ASK4HELP_RLINF_ROOT="$RLINF" OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 \
    taskset -c "$cpuset" nohup "$PY" "$TOOLS/collect_open_drawer_dagger.py" \
      --method offline_oracle --checkpoint "$BASE_CKPT" --pi05-base "$PI05" --norm-stats "$NORM" \
      --output-root "$out" --raw-attempt-budget 200 --max-attempts 200 \
      --id-seed-start $((400000 + index * 10000)) --ood-seed-start $((500000 + index * 10000)) \
      --ood-split "$split" --execute-horizon 5 --max-policy-steps 240 \
      >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
}

for spec in "handle_ood 1 0-19" "grasp_ood 2 20-39" "goal_ood 3 40-59"; do
  set -- $spec
  run_oracle "$1" "$2" "$3" &
done
wait

for split in handle_ood grasp_ood goal_ood; do
  while alive "$PID_DIR/oracle_$split.pid"; do sleep 300; done
  [ -f "$ORACLE/$split/summary.json" ] || {
    echo "$(date -Is) oracle_missing_summary split=$split" >> "$LOG"
    exit 1
  }
done

"$PY" - "$ORACLE" "$RUN" <<'PY'
import json
import sys
from pathlib import Path

oracle = Path(sys.argv[1])
run = Path(sys.argv[2])
splits = ["handle_ood", "grasp_ood", "goal_ood"]
reports = {}
for split in splits:
    rows = [
        json.loads(line)
        for line in (oracle / split / "raw_attempts.jsonl").read_text().splitlines()
        if line
    ]
    id_rows = [row for row in rows if row.get("source") == "id"]
    ood_rows = [row for row in rows if row.get("source") == "ood"]
    reports[split] = {
        "raw_attempts": len(rows),
        "id_attempts": len(id_rows),
        "ood_attempts": len(ood_rows),
        "id_success_rate": sum(bool(row.get("success")) for row in id_rows) / max(1, len(id_rows)),
        "ood_success_rate": sum(bool(row.get("success")) for row in ood_rows) / max(1, len(ood_rows)),
    }
id_rows = [
    json.loads(line)
    for line in (oracle / "handle_ood" / "raw_attempts.jsonl").read_text().splitlines()
    if line and json.loads(line).get("source") == "id"
]
reports["id_reference"] = {
    "attempts": len(id_rows),
    "success_rate": sum(bool(row.get("success")) for row in id_rows) / max(1, len(id_rows)),
}
passed = (
    len(id_rows) >= 100
    and reports["id_reference"]["success_rate"] >= 0.90
    and all(
        reports[split]["ood_attempts"] >= 100
        and reports[split]["ood_success_rate"] >= 0.90
        for split in splits
    )
)
(run / "oracle_gate_summary.json").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
if not passed:
    (run / "ORACLE_GATE_REJECTED").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    raise SystemExit("Oracle gate failed")
(run / "ORACLE_GATE_COMPLETE").write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
PY

if [ ! -f "$ASSET/manifest.json" ]; then
  CUDA_VISIBLE_DEVICES=${OPEN_DRAWER_ASSET_GPU:-1} ASK4HELP_RLINF_ROOT="$RLINF" \
    taskset -c 140-159 "$PY" "$TOOLS/build_open_drawer_internal_detector_assets.py" \
      --checkpoint "$BASE_CKPT" --pi05-base "$PI05" --norm-stats "$NORM" \
      --dataset-root "$ID_DATA" --output-dir "$ASSET"
fi

if [ ! -f "$PCA_CAL/calibration.json" ]; then
  CUDA_VISIBLE_DEVICES=${OPEN_DRAWER_CAL_GPU:-1} ASK4HELP_RLINF_ROOT="$RLINF" \
    taskset -c 140-159 "$PY" "$TOOLS/evaluate_open_drawer_failure_detectors.py" \
      --mode calibrate --checkpoint "$BASE_CKPT" --pi05-base "$PI05" --norm-stats "$NORM" \
      --detector-assets "$ASSET/detector_assets.pt" --output-dir "$PCA_CAL" \
      --split id --seed 600000 --episodes 20 --target-successes 20 \
      --execute-horizon 5 --max-episode-steps 400
fi

if [ ! -f "$DIFF_CAL" ]; then
  CUDA_VISIBLE_DEVICES=${OPEN_DRAWER_CAL_GPU:-1} ASK4HELP_RLINF_ROOT="$RLINF" \
    taskset -c 140-159 "$PY" "$TOOLS/calibrate_open_drawer_diffdagger.py" \
      --checkpoint "$BASE_CKPT" --pi05-base "$PI05" --norm-stats "$NORM" \
      --output "$DIFF_RAW" --num-episodes 80 --seed-start 610000 \
      --execute-horizon 5 --max-policy-steps 240 --timesteps 16 --noise-samples 1
  "$PY" - "$DIFF_RAW" "$DIFF_CAL" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text())
successful = [episode for episode in source["episodes"] if episode.get("success")]
if len(successful) < 20:
    raise SystemExit(f"Diff calibration has only {len(successful)}/20 successful ID rollouts")
selected = successful[:20]
scores = [score for episode in selected for score in episode["scores"]]
payload = {
    "format": "open_drawer_diffdagger_successful_id_calibration_v2",
    "checkpoint": source["checkpoint"],
    "norm_stats": source["norm_stats"],
    "split": "id",
    "target_successes": 20,
    "successful_seeds": [episode["seed"] for episode in selected],
    "timesteps": source["timesteps"],
    "noise_samples": source["noise_samples"],
    "scores": scores,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
fi

printf 'ID base, Oracle, PCA and Diff gates passed.\n' > "$RUN/protocol_gates/PROTOCOL_GATES_COMPLETE"
echo "$(date -Is) protocol_gates_complete" >> "$LOG"

STAGE_PID=$PID_DIR/protocol_v3_collection_controller.pid
if ! alive "$STAGE_PID"; then
  STAGE_CONTROLLER=${OPEN_DRAWER_STAGE_CONTROLLER:-$TOOLS/run_open_drawer_protocol_v3_collection_controller.sh}
  env OPEN_DRAWER_ROOT="$ROOT" OPEN_DRAWER_PYTHON="$PY" OPEN_DRAWER_TOOLS="$TOOLS" \
    OPEN_DRAWER_RLINF_ROOT="$RLINF" OPEN_DRAWER_PROTOCOL_CKPT="$BASE_CKPT" \
    OPEN_DRAWER_PROTOCOL_NORM="$NORM" OPEN_DRAWER_PROTOCOL_DETECTOR_ASSETS="$ASSET/detector_assets.pt" \
    OPEN_DRAWER_PROTOCOL_PCA_CALIBRATION="$PCA_CAL/calibration.json" \
    OPEN_DRAWER_PROTOCOL_DIFF_CALIBRATION="$DIFF_CAL" OPEN_DRAWER_PROTOCOL_RUN="$RUN" \
    OPEN_DRAWER_TARGET_TOTAL_ACCEPTED=100 OPEN_DRAWER_MAX_ATTEMPTS=1600 \
    nohup bash "$STAGE_CONTROLLER" >"$LOG_DIR/protocol_v3_collection_controller_stdout.log" 2>&1 < /dev/null &
  echo $! > "$STAGE_PID"
fi
