#!/usr/bin/env bash
set -u

# Diagnostic-only video sweep for the repaired current-state Oracle.  It never
# touches the formal timing root or the adaptive training root.

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
PLANNER_PY=${PANDA_PLANNER_PYTHON:-/data/zhaozhixuan/simplerenv_ms3/env/bin/python}
CHECKPOINT=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
PI05_BASE=${OPEN_DRAWER_TIMING_PI05_BASE:?set OPEN_DRAWER_TIMING_PI05_BASE}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
RUN=${OPEN_DRAWER_DIRECT_ORACLE_ROOT:-$ROOT/results/open_drawer_grasp_timing_sweep_v1_direct_oracle_retry1}
GPU=${OPEN_DRAWER_DIRECT_ORACLE_GPU:-4}
CPU_SET=${OPEN_DRAWER_DIRECT_ORACLE_CPU_SET:-80-99}
TARGET=${OPEN_DRAWER_DIRECT_ORACLE_TARGET:-3}
MAX_ATTEMPTS=${OPEN_DRAWER_DIRECT_ORACLE_MAX_ATTEMPTS:-10}
SEED_START=${OPEN_DRAWER_DIRECT_ORACLE_SEED_START:-90000}
ANCHORS=(0 50 80 120 160 220)
STATE=$RUN/direct_oracle_video_pipeline_state.json
LOG=$RUN/direct_oracle_video_pipeline.log

mkdir -p "$RUN" "$RUN/logs"
exec > >(tee -a "$LOG") 2>&1

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_direct_oracle_video_sweep_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
}

fail() {
  write_state "$1" failed "${2:-}"
  printf '%s\n' "stage=$1 detail=${2:-}" > "$RUN/DIRECT_ORACLE_VIDEO_SWEEP_FAILED"
  exit 1
}

gpu_is_idle() {
  local gpu=$1 used util uuid apps
  read -r used util < <(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}')
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | grep "$uuid" || true)
  [[ "$used" -le 100 && "$util" -le 5 && -z "$apps" ]]
}

write_state preflight running "gpu=$GPU target=$TARGET oracle_mode=direct_grasp planner_mode=shortest_joint_path"
gpu_is_idle "$GPU" || fail preflight "GPU $GPU is not genuinely idle"
[[ -s "$CHECKPOINT/actor/model_state_dict/full_weights.pt" ]] || fail preflight "missing checkpoint full_weights"
[[ -d "$CHECKPOINT/actor/dcp_checkpoint" ]] || fail preflight "missing checkpoint DCP"
[[ -d "$PI05_BASE" ]] || fail preflight "missing pi05 base"
[[ -d "$NORM" ]] || fail preflight "missing frozen norm"

export PYTHONPATH="$ROOT:$RL"
export ASK4HELP_RLINF_ROOT="$RL"
export PANDA_PLANNER_PYTHON="$PLANNER_PY"
export PANDA_PLANNER_MODE=shortest_joint_path
export PANDA_OBJECT_GRASP_MODE=symmetric_shortest
export CUDA_VISIBLE_DEVICES="$GPU"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

for index in "${!ANCHORS[@]}"; do
  anchor=${ANCHORS[$index]}
  out="$RUN/anchor_${anchor}"
  if [[ -f "$out/summary.json" && -f "$out/DIRECT_ORACLE_COLLECTION_COMPLETE" ]]; then
    write_state "anchor_${anchor}" completed "existing audited diagnostic output"
    continue
  fi
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "anchor_${anchor}" "refusing partial output: $out"
  fi
  mkdir -p "$out"
  write_state "anchor_${anchor}" running "takeover_step=$anchor gpu=$GPU"
  if ! taskset -c "$CPU_SET" "$PY" -u "$ROOT/tools/collect_open_drawer_fixed_timing.py" \
      --checkpoint "$CHECKPOINT" --pi05-base "$PI05_BASE" --norm-stats "$NORM" \
      --output-root "$out" --takeover-step "$anchor" --start-seed "$((SEED_START + index * 100))" \
      --target "$TARGET" --max-attempts "$MAX_ATTEMPTS" --execute-horizon 5 \
      --max-episode-steps 400 --oracle-mode direct_grasp \
      > "$RUN/logs/anchor_${anchor}.log" 2>&1; then
    printf '%s\n' "anchor=$anchor collection command returned nonzero; inspect summary/raw attempts" >> "$RUN/logs/direct_oracle_video_pipeline.log"
  fi
  if [[ -f "$out/summary.json" ]]; then
    accepted=$($PY -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("accepted",0)) )' "$out/summary.json" 2>/dev/null || printf '0')
    if [[ "$accepted" -ge "$TARGET" ]]; then
      printf '%s\n' "direct oracle anchor $anchor accepted=$accepted target=$TARGET" > "$out/DIRECT_ORACLE_COLLECTION_COMPLETE"
    else
      printf '%s\n' "direct oracle anchor $anchor accepted=$accepted target=$TARGET" > "$out/DIRECT_ORACLE_COLLECTION_FAILED"
    fi
  else
    fail "anchor_${anchor}" "missing summary after collection"
  fi
  write_state "anchor_${anchor}" completed "summary_and_video_artifacts_written"
done

printf '%s\n' "direct current-state Oracle videos complete; diagnostic only" > "$RUN/DIRECT_ORACLE_VIDEO_SWEEP_COMPLETE"
write_state complete complete "six takeover anchors; planner=shortest_joint_path; direct grasp mode"
echo OPEN_DRAWER_DIRECT_ORACLE_VIDEO_SWEEP_COMPLETE
