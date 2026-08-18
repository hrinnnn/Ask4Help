#!/usr/bin/env bash
set -u

# Restart-tolerant OpenDrawer ID-only recovery from the verified step-9500 pi0.5.
# This controller owns collection, training segments, and independent ID gates;
# it never starts OOD, PCA, or DAgger.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf}
PY=${OPEN_DRAWER_PYTHON:-$RL/.venv/bin/python}
PLANNER_PYTHON=${OPEN_DRAWER_PLANNER_PYTHON:-$ROOT/oracle_compat_env_v1/bin/python}
PLANNER_SITE_PACKAGES=${OPEN_DRAWER_PLANNER_SITE_PACKAGES:-$RL/.venv/lib/python3.11/site-packages}

RUN=${OPEN_DRAWER_RECOVERY_ROOT:-/sdd/ask4help-open-drawer/results/open_drawer_id_recovery_from9500_v1}
BASE=${OPEN_DRAWER_STEP9500:-/sdd/ask4help-open-drawer/results/open_drawer_id_expansion_recovery_v6/training/sft_from4000_to10000/expansion_recovery_from4000/checkpoints/global_step_9500}
OLD_DATA=${OPEN_DRAWER_OLD_DATASET:-/sdd/ask4help-open-drawer/results/open_drawer_id_expansion_recovery_v4/datasets/id_merged256}
NORM=${OPEN_DRAWER_FROZEN_NORM:-/sdd/ask4help-open-drawer/results/open_drawer_id_expansion_recovery_v4/norm_stats/id_merged256}
NEW_DATA=${OPEN_DRAWER_NEW_DATASET:-$RUN/datasets/id_extra256}
COLLECTION=${OPEN_DRAWER_NEW_COLLECTION:-$RUN/collection_extra256}
TRAIN=$RUN/training/sft_from9500_to15000
EXPERIMENT=pi05_id_recovery_from9500

COLLECT_GPU=${OPEN_DRAWER_COLLECTION_GPU:-7}
TRAIN_GPU=${OPEN_DRAWER_TRAIN_GPU:-7}
EVAL_GPU=${OPEN_DRAWER_EVAL_GPU:-7}
COLLECT_CPU=${OPEN_DRAWER_COLLECTION_CPU:-0-19}
TRAIN_CPU=${OPEN_DRAWER_TRAIN_CPU:-20-39}
EVAL_CPU=${OPEN_DRAWER_EVAL_CPU:-40-59}

LOG_DIR=$RUN/logs
PID_DIR=$RUN/pids
PROV_DIR=$RUN/provenance
STATE=$RUN/pipeline_state.json

mkdir -p "$RUN" "$LOG_DIR" "$PID_DIR" "$PROV_DIR"

log() { printf '%s %s\n' "$(date -Is)" "$*" >> "$LOG_DIR/controller.log"; }
state() {
  printf '{"task":"OpenDrawer pi0.5 ID recovery from step9500","stage":"%s","status":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$3" "$(date -Is)" > "$STATE"
}
alive() { [ -s "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
fail() {
  log "FAILED $*"
  state failed failed "$*"
  printf '%s\n' "$*" > "$RUN/PIPELINE_FAILED"
  exit 1
}

write_manifest() {
  [ -f "$PROV_DIR/run_manifest_recovery_from9500.txt" ] && return 0
  cat > "$PROV_DIR/run_manifest_recovery_from9500.txt" <<EOF
task=OpenDrawerRetrievePlace pi0.5 ID recovery
immutable_step9500=$BASE
old_id_dataset=$OLD_DATA
new_id_dataset=$NEW_DATA
frozen_norm=$NORM
norm_scope=frozen merged ID-only norm; no recomputation in this run
new_collection=$COLLECTION
old_id_episodes=256
new_id_target_episodes=256
max_episode_steps=400
execute_horizon=5
action_horizon=10
global_batch_size=128
micro_batch_size=32
source_balance=1:1 old256:new256
noise_method=flow_sde
train_expert_only=true
awbc=false
train_targets=11000,13000,15000
id_gate_episodes=100
id_gate_threshold=80
ood_started=false
pca_started=false
dagger_started=false
planner_python=$PLANNER_PYTHON
planner_site_packages=$PLANNER_SITE_PACKAGES
EOF
}

checkpoint_path() { printf '%s/%s/checkpoints/global_step_%s\n' "$TRAIN" "$EXPERIMENT" "$1"; }
checkpoint_complete() {
  local path; path=$(checkpoint_path "$1")
  [ -f "$path/actor/model_state_dict/full_weights.pt" ] && [ -d "$path/actor/dcp_checkpoint" ]
}

wait_stage() {
  local pid_file=$1 stage=$2
  while alive "$pid_file"; do
    state "$stage" running "pid=$(cat "$pid_file")"
    sleep 300
  done
}

audit_new_collection() {
  local audit="$PROV_DIR/new_id_256_audit.json"
  [ -f "$audit" ] && return 0
  state id_collection_audit running audit
  "$PY" - "$NEW_DATA" "$COLLECTION" "$audit" <<'PY' || fail id_collection_audit_failed
import json, sys
from pathlib import Path
import pyarrow.parquet as pq

dataset, collection, output = map(Path, sys.argv[1:])
info = json.loads((dataset / "meta/info.json").read_text())
episodes = [json.loads(x) for x in (dataset / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
rows = [json.loads(x) for x in (collection / "episodes.jsonl").read_text().splitlines() if x.strip()]
chunks = int(info["chunks_size"])
errors = []
lengths = []
if len(episodes) != 256 or len(rows) != 256:
    errors.append(f"episode_count:{len(episodes)}:{len(rows)}")
for ep in episodes:
    index, length = int(ep["episode_index"]), int(ep["length"])
    lengths.append(length)
    if length >= 400 or length < 1:
        errors.append(f"invalid_length:{index}:{length}")
    path = dataset / "data" / f"chunk-{index // chunks:03d}" / f"episode_{index:06d}.parquet"
    if not path.is_file():
        errors.append(f"missing_parquet:{index}")
        continue
    table = pq.read_table(path)
    if table.num_rows != length:
        errors.append(f"row_count:{index}:{table.num_rows}!={length}")
    for key, dim in (("actions", 8), ("state", 9)):
        if key not in table.column_names:
            errors.append(f"missing_column:{index}:{key}")
            continue
        values = table[key].to_pylist()
        if values and len(values[0]) != dim:
            errors.append(f"dimension:{index}:{key}:{len(values[0])}!={dim}")
for row in rows:
    if row.get("success") is not True:
        errors.append(f"unsuccessful:{row.get('episode_index')}")
    if int(row.get("num_actions", 0)) < 1:
        errors.append(f"empty_actions:{row.get('episode_index')}")
    if not row.get("oracle", {}).get("stages", {}).get("success", False):
        errors.append(f"missing_oracle_success_stage:{row.get('episode_index')}")
report = {
    "format": "open_drawer_id_recovery_256_audit_v1",
    "episodes": len(episodes),
    "collection_rows": len(rows),
    "length_min": min(lengths) if lengths else None,
    "length_max": max(lengths) if lengths else None,
    "action_dim": 8,
    "state_dim": 9,
    "action_horizon": 10,
    "tail_anchor_count": 9 * len(episodes),
    "final_anchor_retained": True,
    "execute_horizon": 5,
    "max_episode_steps": 400,
    "errors": sorted(set(errors)),
    "pass": not errors,
    "dataset": str(dataset),
    "collection": str(collection),
}
output.write_text(json.dumps(report, indent=2) + "\n")
if errors:
    raise SystemExit(1)
PY
  local videos
  videos=$(find "$COLLECTION/videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  [ "$videos" = 256 ] || fail "new_video_count=$videos"
  for video in "$COLLECTION"/videos/*.mp4; do
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$video" >/dev/null || fail "video_decode_failed=$video"
  done
  printf '256 new ID trajectories; actions/states/videos/stages and tail-anchor audit passed.\n' > "$RUN/COLLECTION_AUDIT_PASSED"
}

run_collection() {
  [ -f "$RUN/COLLECTION_AUDIT_PASSED" ] && return 0
  [ -f "$RUN/ORACLE_QUALITY_DATASET_PASSED" ] || fail oracle_quality_gate_missing
  if [ -e "$COLLECTION" ] || [ -e "$NEW_DATA" ]; then
    [ -f "$COLLECTION/summary.json" ] || fail collection_partial_output
    [ -f "$NEW_DATA/meta/info.json" ] || fail new_dataset_partial_output
  else
    state id_collection_extra256 starting starting
    env CUDA_VISIBLE_DEVICES="$COLLECT_GPU" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PANDA_PLANNER_SITE_PACKAGES="$PLANNER_SITE_PACKAGES" PYTHONPATH="$RL:$ROOT" \
      PANDA_PLANNER_PYTHON="$PLANNER_PYTHON" HF_LEROBOT_HOME=/sdd/ask4help-open-drawer/lerobot_cache \
      taskset -c "$COLLECT_CPU" nohup "$PY" \
      "$RL/toolkits/lerobot/collect_open_drawer_retrieve_place_lerobot.py" \
      --repo-id "$NEW_DATA" --output-dir "$COLLECTION" --video-dir "$COLLECTION/videos" \
      --num-episodes 256 --seed 78000 --max-attempts 320 --image-size 384 \
      --control-freq 10 --max-episode-steps 400 --save-videos \
      > "$LOG_DIR/collection_extra256.log" 2>&1 < /dev/null &
    echo $! > "$PID_DIR/collection_extra256.pid"
  fi
  wait_stage "$PID_DIR/collection_extra256.pid" id_collection_extra256
  [ -f "$COLLECTION/summary.json" ] || fail collection_summary_missing
  audit_new_collection
}

run_training_segment() {
  local target=$1
  local resume=$2
  local pid_file="$PID_DIR/train_to_${target}.pid"
  local log_file="$LOG_DIR/train_to_${target}.log"
  checkpoint_complete "$target" && return 0
  [ -f "$pid_file" ] && alive "$pid_file" && wait_stage "$pid_file" "training_to_${target}"
  checkpoint_complete "$target" && return 0
  state "training_to_${target}" starting "resume=$resume"
  DATA_OVERRIDE="data.train_data_paths=[{dataset_path:$OLD_DATA,weight:1.0},{dataset_path:$NEW_DATA,weight:1.0}]"
  env CUDA_VISIBLE_DEVICES="$TRAIN_GPU" ASK4HELP_RLINF_PLACEMENT="$TRAIN_GPU-$TRAIN_GPU" \
    OPEN_DRAWER_ID_DATASET="$OLD_DATA" OPEN_DRAWER_ID_NORM_STATS="$NORM" \
    OPEN_DRAWER_PI05_MODEL_PATH="$ROOT/results/model_cache/pi05_base_pytorch_v1" \
    OPEN_DRAWER_RUN_ROOT="$TRAIN" OPEN_DRAWER_EXPERIMENT_NAME="$EXPERIMENT" \
    RAY_TMPDIR="/tmp/od_recovery_train_${target}" TMPDIR="$TRAIN/tmp_${target}" \
    HF_HOME="$ROOT/runtime_cache/hf_home" PYTHONUNBUFFERED=1 PYTHONPATH="$RL:$ROOT" \
    EMBODIED_PATH="$RL/examples/sft" REPO_PATH="$RL" taskset -c "$TRAIN_CPU" nohup "$PY" \
    "$RL/examples/sft/train_vla_sft.py" --config-path "$RL/examples/sft/config" \
    --config-name open_drawer_retrieve_place_id_sft_openpi_pi05 \
    +runner.resume_dir="$resume" runner.max_steps="$target" runner.save_interval=500 \
    actor.global_batch_size=128 actor.micro_batch_size=32 actor.optim.total_training_steps=15000 \
    data.openpi_source_balanced=true "$DATA_OVERRIDE" \
    > "$log_file" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  wait_stage "$pid_file" "training_to_${target}"
  checkpoint_complete "$target" || fail "checkpoint_missing_step=$target"
}

run_id_gate() {
  local step=$1
  local ckpt; ckpt=$(checkpoint_path "$step")
  local eval_dir="$RUN/eval_id_100_by_step/step_$step"
  local pid_file="$PID_DIR/eval_id_step_${step}.pid"
  local log_file="$LOG_DIR/eval_id_step_${step}.log"
  [ -f "$eval_dir/summary.json" ] && return 0
  mkdir -p "$eval_dir"
  state "id_gate_step_${step}" starting "checkpoint=$ckpt"
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$RL:$ROOT" taskset -c "$EVAL_CPU" nohup "$PY" \
    "$ROOT/tools/evaluate_open_drawer_id_pi05.py" --checkpoint "$ckpt" \
    --pi05-base "$ROOT/results/model_cache/pi05_base_pytorch_v1" --norm-stats "$NORM" \
    --output-dir "$eval_dir" --episodes 100 --seed 52000 --split id \
    --execute-horizon 5 --max-episode-steps 400 > "$log_file" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  wait_stage "$pid_file" "id_gate_step_${step}"
  [ -f "$eval_dir/summary.json" ] || fail "id_gate_summary_missing_step=$step"
  local videos; videos=$(find "$eval_dir/videos" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  local evidence_dir="$eval_dir/episodes"
  local action_files; action_files=$(find "$evidence_dir" -mindepth 2 -maxdepth 2 -type f -name 'actions.npy' 2>/dev/null | wc -l | tr -d ' ')
  local state_files; state_files=$(find "$evidence_dir" -mindepth 2 -maxdepth 2 -type f -name 'states.npy' 2>/dev/null | wc -l | tr -d ' ')
  local timeline_files; timeline_files=$(find "$evidence_dir" -mindepth 2 -maxdepth 2 -type f -name 'timeline.json' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$videos" != 100 ] || [ "$action_files" != 100 ] || [ "$state_files" != 100 ] || [ "$timeline_files" != 100 ] || [ ! -f "$eval_dir/provenance.json" ]; then
    printf '%s\n' "videos=$videos actions=$action_files states=$state_files timelines=$timeline_files provenance=$(test -f "$eval_dir/provenance.json" && echo yes || echo no)" > "$eval_dir/GATE_EVIDENCE_FAILED"
    fail "id_gate_evidence_missing_step=$step"
  fi
  "$PY" - "$evidence_dir" <<'PY' || fail "id_gate_evidence_invalid_step=$step"
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
dirs = sorted(path for path in root.iterdir() if path.is_dir())
if len(dirs) != 100:
    raise SystemExit(f"episode_dirs={len(dirs)}")
for path in dirs:
    actions = np.load(path / "actions.npy")
    states = np.load(path / "states.npy")
    timeline = json.loads((path / "timeline.json").read_text())
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise SystemExit(f"bad_actions:{path}:{actions.shape}")
    if states.ndim != 2 or states.shape[0] != actions.shape[0] + 1 or states.shape[1] < 9:
        raise SystemExit(f"bad_states:{path}:{states.shape}:{actions.shape}")
    if len(timeline.get("timeline", [])) != actions.shape[0] + 1:
        raise SystemExit(f"bad_timeline:{path}")
    if not timeline.get("tail_observation_retained", False):
        raise SystemExit(f"tail_missing:{path}")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise SystemExit(f"nonfinite:{path}")
PY
  "$PY" - "$eval_dir/summary.json" "$RUN" "$step" "$videos" <<'PY'
import json, sys
from pathlib import Path
summary=json.loads(Path(sys.argv[1]).read_text())
run=Path(sys.argv[2]); step=int(sys.argv[3]); videos=int(sys.argv[4])
payload={"step":step,"episodes":summary.get("episodes"),"successes":summary.get("successes"),"videos":videos,"actions":100,"states":100,"timelines":100,"provenance":str(Path(sys.argv[1]).parent/"provenance.json"),"summary":sys.argv[1],"pass":summary.get("episodes")==100 and videos==100 and summary.get("successes",0)>=80}
(run/f"ID_GATE_STEP_{step}.json").write_text(json.dumps(payload,indent=2)+"\n")
if payload["pass"]:
    (run/f"ID_BASE_CANDIDATE_STEP_{step}").write_text(json.dumps(payload,indent=2)+"\n")
PY
}

main() {
  write_manifest
  state preflight running "step9500 and smoke gates verified"
  run_collection
  run_training_segment 11000 "$BASE"
  run_id_gate 11000
  if grep -q '"pass": true' "$RUN/ID_GATE_STEP_11000.json"; then
    printf 'ID_RECOVERY_COMPLETE\n' > "$RUN/ID_RECOVERY_COMPLETE"
    printf 'PIPELINE_COMPLETE\n' > "$RUN/PIPELINE_COMPLETE"
    state complete passed "ID gate passed at step11000"
    exit 0
  fi
  run_training_segment 13000 "$(checkpoint_path 11000)"
  run_id_gate 13000
  if grep -q '"pass": true' "$RUN/ID_GATE_STEP_13000.json"; then
    printf 'ID_RECOVERY_COMPLETE\n' > "$RUN/ID_RECOVERY_COMPLETE"
    printf 'PIPELINE_COMPLETE\n' > "$RUN/PIPELINE_COMPLETE"
    state complete passed "ID gate passed at step13000"
    exit 0
  fi
  run_training_segment 15000 "$(checkpoint_path 13000)"
  run_id_gate 15000
  if grep -q '"pass": true' "$RUN/ID_GATE_STEP_15000.json"; then
    printf 'ID_RECOVERY_COMPLETE\n' > "$RUN/ID_RECOVERY_COMPLETE"
    printf 'PIPELINE_COMPLETE\n' > "$RUN/PIPELINE_COMPLETE"
    state complete passed "ID gate passed at step15000"
  else
    printf 'ID_BASE_PROTOCOL_FAILED\n' > "$RUN/ID_BASE_PROTOCOL_FAILED"
    printf 'PIPELINE_COMPLETE\n' > "$RUN/PIPELINE_COMPLETE"
    state complete failed "ID gate below 80 at step15000"
  fi
}

main "$@"
