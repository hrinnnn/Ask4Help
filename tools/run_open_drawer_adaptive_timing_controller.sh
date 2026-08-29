#!/usr/bin/env bash
set -u

# Adaptive, one-model-per-anchor timing pipeline.  The pilot is trained in
# cumulative segments (5000, then +2500 as needed) and evaluated after every
# segment.  Once the first strict Grasp-OOD probe exceeds 40%, that cumulative
# step count is frozen for the remaining anchors.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_retry6_adaptive}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
ID_DATASET=${OPEN_DRAWER_TIMING_ID_DATASET:?set OPEN_DRAWER_TIMING_ID_DATASET}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
FORMAL_ROOT=${OPEN_DRAWER_TIMING_FORMAL_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_formal/formal}
BUDGET_ROOT=${OPEN_DRAWER_TIMING_BUDGET_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_retry3/formal_budget}
TRAIN_SEED=${OPEN_DRAWER_TIMING_TRAIN_SEED:-9301}
MIN_STEPS=${OPEN_DRAWER_TIMING_MIN_STEPS:-5000}
INCREMENT=${OPEN_DRAWER_TIMING_STEP_INCREMENT:-2500}
OOD_THRESHOLD=${OPEN_DRAWER_TIMING_OOD_THRESHOLD:-0.4}
OOD_SEED_START=${OPEN_DRAWER_TIMING_OOD_SEED_START:-79000}
GPU_POOL=(${OPEN_DRAWER_TIMING_GPU_POOL:-"0 1 2 3 4 5 6 7"})
LOCK_ROOT=${OPEN_DRAWER_TIMING_GPU_LOCK_ROOT:-$RUN/.gpu_locks}
STATE=$RUN/adaptive_timing_pipeline_state.json
LOG=$RUN/adaptive_timing_controller.log
FROZEN_STEPS=$RUN/adaptive_steps.json
ANCHORS=(0 50 80 120 160 220)

mkdir -p "$RUN" "$RUN/training" "$RUN/ood20_probe" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_adaptive_timing_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

fail() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "stage=$1 detail=${2:-}" > "$RUN/ADAPTIVE_TIMING_FAILED"
  exit 1
}

cpu_set_for_gpu() {
  case "$1" in
    0) printf '%s\n' '0-19' ;;
    1) printf '%s\n' '20-39' ;;
    2) printf '%s\n' '40-59' ;;
    3) printf '%s\n' '60-79' ;;
    4) printf '%s\n' '80-99' ;;
    5) printf '%s\n' '100-119' ;;
    6) printf '%s\n' '120-139' ;;
    7) printf '%s\n' '140-159' ;;
    *) return 1 ;;
  esac
}

gpu_is_idle() {
  local gpu=$1 used util uuid apps
  read -r used util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$uuid" || true)
  [[ "$used" -le 100 && "$util" -le 5 && -z "$apps" ]]
}

acquire_gpu() {
  local gpu lock owner cpuset
  mkdir -p "$LOCK_ROOT"
  while true; do
    for gpu in "${GPU_POOL[@]}"; do
      gpu_is_idle "$gpu" || continue
      lock="$LOCK_ROOT/gpu_$gpu"
      if ! mkdir "$lock" 2>/dev/null; then
        owner=$(cat "$lock/owner" 2>/dev/null || true)
        if [[ -n "$owner" ]] && ! kill -0 "$owner" 2>/dev/null; then
          rm -f "$lock/owner" 2>/dev/null || true
          rmdir "$lock" 2>/dev/null || true
        fi
        continue
      fi
      cpuset=$(cpu_set_for_gpu "$gpu") || { rmdir "$lock"; continue; }
      if ! gpu_is_idle "$gpu"; then
        rmdir "$lock" 2>/dev/null || true
        continue
      fi
      printf '%s\n' "$$" > "$lock/owner"
      SELECTED_GPU=$gpu
      SELECTED_CPU_SET=$cpuset
      return 0
    done
    write_state waiting_for_idle_gpu waiting "gpu_pool=${GPU_POOL[*]}"
    sleep 300
  done
}

release_gpu() {
  local lock="$LOCK_ROOT/gpu_$1"
  rm -f "$lock/owner" 2>/dev/null || true
  rmdir "$lock" 2>/dev/null || true
}

segment_root() {
  printf '%s\n' "$RUN/training/anchor_${1}/seed_${TRAIN_SEED}/steps_${2}"
}

segment_checkpoint() {
  local root=$1 segment_steps=$2
  printf '%s\n' "$root/run/checkpoints/global_step_${segment_steps}"
}

train_segment() {
  local anchor=$1 source_model=$2 segment_steps=$3 cumulative_steps=$4
  local expert="$BUDGET_ROOT/anchor_${anchor}"
  local out
  out=$(segment_root "$anchor" "$cumulative_steps")
  local checkpoint
  checkpoint=$(segment_checkpoint "$out" "$segment_steps")
  local weights="$checkpoint/actor/model_state_dict/full_weights.pt"
  local dcp="$checkpoint/actor/dcp_checkpoint"
  if [[ -s "$weights" && -d "$dcp" ]]; then
    printf '%s\n' "segment already complete anchor=$anchor cumulative=$cumulative_steps"
    printf '%s\n' "checkpoint=$checkpoint source=$source_model" > "$out/SEGMENT_COMPLETE"
    return 0
  fi
  [[ -d "$expert" ]] || return 1
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf '%s\n' "refusing partial adaptive segment output: $out" >&2
    return 1
  fi
  mkdir -p "$out"
  write_state "training_anchor_${anchor}_to_${cumulative_steps}" running "segment_steps=$segment_steps source=$source_model gpu=$SELECTED_GPU"
  local config_path="$RL/examples/sft/config/open_drawer_retrieve_place_dagger_sft_openpi_pi05.yaml"
  grep -Fq 'default_prompt: open the drawer, retrieve the blue object, and place it in the green tray' "$config_path" || return 1
  env CUDA_VISIBLE_DEVICES="$SELECTED_GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ASK4HELP_RLINF_PLACEMENT="$SELECTED_GPU-$SELECTED_GPU" RLINF_RAY_ADDRESS=local \
    EMBODIED_PATH="$RL/examples/sft" PYTHONPATH="$ROOT:$RL" \
    OPEN_DRAWER_ID_DATASET="$ID_DATASET" OPEN_DRAWER_EXPERT_DATASET="$expert" \
    OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$source_model" \
    OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=run \
    OPEN_DRAWER_TRAIN_SEED="$TRAIN_SEED" OPEN_DRAWER_RESUME_DIR="" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_CACHE="$RUN/runtime/hf_datasets" HF_HOME="$RUN/runtime/hf_home" \
    RAY_TMPDIR="$RUN/runtime/ray/anchor_${anchor}_steps_${cumulative_steps}" \
    TMPDIR="$RUN/runtime/tmp/anchor_${anchor}_steps_${cumulative_steps}" PYTHONUNBUFFERED=1 \
    taskset -c "$SELECTED_CPU_SET" "$PY" "$RL/examples/sft/train_vla_sft.py" \
      --config-path "$RL/examples/sft/config" \
      --config-name open_drawer_retrieve_place_dagger_sft_openpi_pi05 \
      runner.max_steps="$segment_steps" runner.save_interval=500 \
      actor.optim.total_training_steps="$segment_steps" runner.resume_dir=null \
      > "$RUN/logs/train_anchor_${anchor}_steps_${cumulative_steps}.log" 2>&1
  [[ -s "$weights" && -d "$dcp" ]] || return 1
  printf '%s\n' "checkpoint=$checkpoint source=$source_model segment_steps=$segment_steps cumulative_steps=$cumulative_steps" > "$out/SEGMENT_COMPLETE"
  return 0
}

audit_probe() {
  local out=$1
  "$PY" - "$out" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1])
summary_path = root / "summary.json"
if not summary_path.is_file():
    raise SystemExit("missing summary")
payload = json.loads(summary_path.read_text(encoding="utf-8"))
if payload.get("split") != "grasp_ood" or payload.get("episodes") != 20:
    raise SystemExit("split/denominator mismatch")
rows = payload.get("rows", [])
if len(rows) != 20 or len(list((root / "videos").glob("*.mp4"))) != 20:
    raise SystemExit("rows/video mismatch")
for row in rows:
    for key in ("actions", "states", "timeline", "reset_metadata", "video"):
        path = Path(str(row.get(key, "")))
        if not path.is_file():
            raise SystemExit(f"missing {key}")
    actions = np.load(Path(row["actions"]))
    states = np.load(Path(row["states"]))
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise SystemExit("bad action shape")
    if states.shape != (len(actions) + 1, 9):
        raise SystemExit("bad state shape")
    timeline = json.loads(Path(row["timeline"]).read_text(encoding="utf-8"))
    if len(timeline.get("timeline", [])) != len(actions) + 1:
        raise SystemExit("timeline mismatch")
    json.loads(Path(row["reset_metadata"]).read_text(encoding="utf-8"))
PY
}

probe_rate() {
  "$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
d = json.loads((Path(sys.argv[1]) / "summary.json").read_text(encoding="utf-8"))
print(float(d["successes"]) / float(d["episodes"]))
PY
}

run_probe() {
  local anchor=$1 cumulative_steps=$2 checkpoint=$3 probe_seed=$4
  local label="anchor_${anchor}/seed_${TRAIN_SEED}/steps_${cumulative_steps}"
  local out="$RUN/ood20_probe/anchor_${anchor}/seed_${TRAIN_SEED}/steps_${cumulative_steps}/grasp_ood"
  local marker="$RUN/ood20_probe/anchor_${anchor}/seed_${TRAIN_SEED}/steps_${cumulative_steps}/OOD20_COMPLETE"
  if [[ -f "$marker" ]]; then
    audit_probe "$out" || return 1
    probe_rate "$out"
    return 0
  fi
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    return 1
  fi
  mkdir -p "$out"
  write_state "ood20_${label//\//_}" running "gpu=$SELECTED_GPU probe_seed=$probe_seed"
  sleep 5
  env CUDA_VISIBLE_DEVICES="$SELECTED_GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ASK4HELP_RLINF_ROOT="$ROOT/RLinf" PYTHONPATH="$ROOT:$ROOT/RLinf" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 PYTHONUNBUFFERED=1 \
    taskset -c "$SELECTED_CPU_SET" "$PY" -u "$ROOT/tools/evaluate_open_drawer_id_pi05.py" \
      --checkpoint "$checkpoint" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
      --output-dir "$out" --episodes 20 --seed "$probe_seed" --split grasp_ood \
      --execute-horizon 5 --max-episode-steps 400 \
      > "$RUN/logs/ood20_anchor_${anchor}_steps_${cumulative_steps}.log" 2>&1 || true
  audit_probe "$out" || return 1
  printf '%s\n' "20 Grasp-OOD episodes audited for $label" > "$marker"
  probe_rate "$out"
}

freeze_steps() {
  local steps=$1 rate=$2
  printf '{"frozen_steps":%s,"pilot_anchor":0,"training_seed":%s,"ood_success_rate":%s,"threshold":%s,"rule":"first strict OOD20 rate above threshold"}\n' "$steps" "$TRAIN_SEED" "$rate" "$OOD_THRESHOLD" > "$FROZEN_STEPS"
}

mkdir -p "$RUN/runtime/ray" "$RUN/runtime/tmp"
[[ -e "$RUN/formal" ]] || ln -s "$FORMAL_ROOT" "$RUN/formal"
[[ -e "$RUN/formal_budget" ]] || ln -s "$BUDGET_ROOT" "$RUN/formal_budget"
[[ -f "$RUN/ADAPTIVE_TIMING_TRAINING_COMPLETE" ]] && { write_state all_adaptive_training complete; exit 0; }

write_state preflight running "one_model_per_anchor seed=$TRAIN_SEED min_steps=$MIN_STEPS increment=$INCREMENT threshold=$OOD_THRESHOLD gpu_pool=${GPU_POOL[*]}"
[[ -f "$BUDGET_ROOT/BUDGET_AUDIT_PASS" ]] || fail preflight "missing exact budget audit: $BUDGET_ROOT"
[[ -f "$FORMAL_ROOT/AUDIT_PASS" ]] || fail preflight "missing formal collection audit: $FORMAL_ROOT"
[[ -s "$MODEL/actor/model_state_dict/full_weights.pt" ]] || fail preflight "missing base full_weights: $MODEL"

if [[ ! -f "$FROZEN_STEPS" ]]; then
  anchor=0
  cumulative=$MIN_STEPS
  source_model=$MODEL
  probe_index=0
  while true; do
    acquire_gpu
    if ! train_segment "$anchor" "$source_model" "$([[ "$cumulative" -eq "$MIN_STEPS" ]] && echo "$MIN_STEPS" || echo "$INCREMENT")" "$cumulative"; then
      release_gpu "$SELECTED_GPU"
      fail "training_anchor_${anchor}_steps_${cumulative}" "checkpoint_missing_or_training_failed"
    fi
    checkpoint=$(segment_checkpoint "$(segment_root "$anchor" "$cumulative")" "$([[ "$cumulative" -eq "$MIN_STEPS" ]] && echo "$MIN_STEPS" || echo "$INCREMENT")")
    rate=$(run_probe "$anchor" "$cumulative" "$checkpoint" "$((OOD_SEED_START + probe_index * 100))") || {
      release_gpu "$SELECTED_GPU"
      fail "ood20_anchor_${anchor}_steps_${cumulative}" "probe_audit_failed"
    }
    release_gpu "$SELECTED_GPU"
    write_state pilot_probe_complete running "cumulative_steps=$cumulative strict_ood20_rate=$rate"
    if awk "BEGIN {exit !($rate > $OOD_THRESHOLD)}"; then
      freeze_steps "$cumulative" "$rate"
      break
    fi
    source_model=$(segment_checkpoint "$(segment_root "$anchor" "$cumulative")" "$([[ "$cumulative" -eq "$MIN_STEPS" ]] && echo "$MIN_STEPS" || echo "$INCREMENT")")
    cumulative=$((cumulative + INCREMENT))
    probe_index=$((probe_index + 1))
  done
else
  cumulative=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["frozen_steps"])' "$FROZEN_STEPS")
fi

frozen_steps=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["frozen_steps"])' "$FROZEN_STEPS")
probe_index=10
for anchor in "${ANCHORS[@]}"; do
  [[ "$anchor" -eq 0 ]] && continue
  segment_steps=$frozen_steps
  source_model=$MODEL
  acquire_gpu
  if ! train_segment "$anchor" "$source_model" "$segment_steps" "$frozen_steps"; then
    release_gpu "$SELECTED_GPU"
    fail "training_anchor_${anchor}_steps_${frozen_steps}" "checkpoint_missing_or_training_failed"
  fi
  checkpoint=$(segment_checkpoint "$(segment_root "$anchor" "$frozen_steps")" "$segment_steps")
  rate=$(run_probe "$anchor" "$frozen_steps" "$checkpoint" "$((OOD_SEED_START + probe_index * 100))") || {
    release_gpu "$SELECTED_GPU"
    fail "ood20_anchor_${anchor}_steps_${frozen_steps}" "probe_audit_failed"
  }
  release_gpu "$SELECTED_GPU"
  write_state "anchor_${anchor}_complete" running "frozen_steps=$frozen_steps strict_ood20_rate=$rate"
  probe_index=$((probe_index + 1))
done

printf '%s\n' "all six anchors trained with seed=$TRAIN_SEED and frozen_steps=$frozen_steps" > "$RUN/ADAPTIVE_TIMING_TRAINING_COMPLETE"
write_state all_adaptive_training complete "frozen_steps=$frozen_steps one_model_per_anchor"
echo OPEN_DRAWER_ADAPTIVE_TIMING_COMPLETE
