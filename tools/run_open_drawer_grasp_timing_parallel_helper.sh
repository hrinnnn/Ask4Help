#!/usr/bin/env bash
set -u

ROOT=${OPEN_DRAWER_ROOT:-/data/zhaozhixuan/Ask4Help-open-drawer}
RL=${OPEN_DRAWER_RLINF_ROOT:-$ROOT/RLinf}
PY=${OPEN_DRAWER_PYTHON:-/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python}
RUN=${OPEN_DRAWER_TIMING_ROOT:?set OPEN_DRAWER_TIMING_ROOT}
MODEL=${OPEN_DRAWER_TIMING_CHECKPOINT:?set OPEN_DRAWER_TIMING_CHECKPOINT}
ID_DATASET=${OPEN_DRAWER_TIMING_ID_DATASET:?set OPEN_DRAWER_TIMING_ID_DATASET}
NORM=${OPEN_DRAWER_TIMING_NORM:?set OPEN_DRAWER_TIMING_NORM}
BUDGET_ROOT=${OPEN_DRAWER_TIMING_BUDGET_ROOT:-$RUN/formal_budget}
STEPS=${OPEN_DRAWER_TIMING_TRAIN_STEPS:-2500}
# GPU2/5 are pre-registered for the diagnostic OOD20 probe; keep them out of
# the training pool while that controller is active to avoid a race.
GPU_POOL=(${OPEN_DRAWER_TIMING_GPU_POOL:-"0 1 3 4 6 7"})
LOCK_ROOT=${OPEN_DRAWER_TIMING_GPU_LOCK_ROOT:-$RUN/.gpu_locks}
PRIORITY_OOD20_GATE=${OPEN_DRAWER_PRIORITY_OOD20_GATE:-1}
RAY_BASE=${OPEN_DRAWER_TIMING_PARALLEL_RAY_BASE:-/sdd/rod_parallel}
TMP_BASE=${OPEN_DRAWER_TIMING_PARALLEL_TMP_BASE:-/sdd/timod_parallel}
STATE=$RUN/parallel_helper_state.json
LOG=$RUN/parallel_helper.log

# These jobs are deliberately ahead of the old GPU4 serial controller.  The
# first job is reached by that controller only after the anchor-0/50/80 jobs;
# the remaining jobs are reached after it has consumed anchor-120 seed 9302/3.
# Keeping this fixed prevents a partial-output race while retaining frozen
# scientific seeds, anchors, budget, and success protocol.
JOBS=(
  "120:9301"
  "160:9301" "160:9302" "160:9303"
  "220:9301" "220:9302" "220:9303"
)

mkdir -p "$RUN/training" "$RUN/parallel_training_logs" "$RAY_BASE" "$TMP_BASE"

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_grasp_timing_parallel_helper_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
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

wait_for_priority_ood20() {
  local condition=$1 seed=$2
  case "$PRIORITY_OOD20_GATE" in
    1|true|TRUE|yes|YES) ;;
    *) return 0 ;;
  esac
  local marker="$RUN/ood20_probe/$condition/seed_${seed}/OOD20_COMPLETE"
  while [[ ! -f "$marker" ]]; do
    if [[ -e "$RUN/OOD20_PROBE_FAILED" ]]; then
      write_state "waiting_ood20_${condition}_seed_${seed}" failed "priority_probe_failed"
      return 1
    fi
    write_state "waiting_ood20_${condition}_seed_${seed}" waiting "priority OOD20 audit required before next training job"
    sleep 300
  done
  return 0
}

train_one() {
  local anchor=$1 seed=$2 gpu=$3 cpuset=$4
  local condition="anchor_${anchor}"
  local expert="$BUDGET_ROOT/$condition"
  local out="$RUN/training/$condition/seed_$seed"
  local config_path="$RL/examples/sft/config/open_drawer_retrieve_place_dagger_sft_openpi_pi05.yaml"
  local checkpoint="$out/run/checkpoints/global_step_$STEPS/actor/model_state_dict/full_weights.pt"
  local log="$RUN/parallel_training_logs/${condition}_seed_${seed}.log"

  if [[ ! -d "$expert" ]]; then
    printf '%s\n' "missing budget dataset: $expert" >&2
    return 1
  fi
  if ! grep -Fq 'default_prompt: open the drawer, retrieve the blue object, and place it in the green tray' "$config_path"; then
    printf '%s\n' "canonical OpenDrawer prompt missing from training config: $config_path" >&2
    return 1
  fi
  if [[ -f "$checkpoint" ]]; then
    printf '%s\n' "checkpoint already present: $checkpoint"
    printf '%s\n' 'training checkpoint verified' > "$out/TRAINING_COMPLETE"
    wait_for_priority_ood20 "$condition" "$seed" || return 1
    return 0
  fi
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf '%s\n' "refusing to reuse partial training output: $out" >&2
    return 1
  fi

  mkdir -p "$out"
  write_state "parallel_training_${condition}_seed_${seed}" running "gpu=$gpu cpu_set=$cpuset"
  env CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ASK4HELP_RLINF_PLACEMENT="$gpu-$gpu" RLINF_RAY_ADDRESS=local \
    EMBODIED_PATH="$RL/examples/sft" PYTHONPATH="$ROOT:$RL" \
    OPEN_DRAWER_ID_DATASET="$ID_DATASET" OPEN_DRAWER_EXPERT_DATASET="$expert" \
    OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" \
    OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=run \
    OPEN_DRAWER_TRAIN_SEED="$seed" OPEN_DRAWER_RESUME_DIR="" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_CACHE="$RUN/runtime/hf_datasets" HF_HOME="$RUN/runtime/hf_home" \
    RAY_TMPDIR="$RAY_BASE/${condition}_${seed}" \
    TMPDIR="$TMP_BASE/${condition}_${seed}" PYTHONUNBUFFERED=1 \
    taskset -c "$cpuset" "$PY" "$RL/examples/sft/train_vla_sft.py" \
      --config-path "$RL/examples/sft/config" \
      --config-name open_drawer_retrieve_place_dagger_sft_openpi_pi05 \
      runner.max_steps="$STEPS" runner.save_interval=500 \
      actor.optim.total_training_steps="$STEPS" runner.resume_dir=null \
      > "$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$out/train.pid"
  wait "$pid"
  local rc=$?
  if [[ ! -f "$checkpoint" ]]; then
    printf '%s\n' "training failed or checkpoint missing rc=$rc path=$checkpoint" >&2
    return 1
  fi
  printf '%s\n' "training checkpoint verified rc=$rc" > "$out/TRAINING_COMPLETE"
  wait_for_priority_ood20 "$condition" "$seed" || return 1
  return 0
}

write_state preflight running "gpu_pool=${GPU_POOL[*]} jobs=${#JOBS[@]}"
for job in "${JOBS[@]}"; do
  IFS=: read -r anchor seed <<< "$job"
  condition="anchor_${anchor}"
  checkpoint="$RUN/training/$condition/seed_$seed/run/checkpoints/global_step_$STEPS/actor/model_state_dict/full_weights.pt"
  if [[ -f "$checkpoint" ]]; then
    train_one "$anchor" "$seed" "" "" || {
      write_state "parallel_training_anchor_${anchor}_seed_${seed}" failed "priority gate failed"
      printf '%s\n' "anchor=$anchor seed=$seed priority gate failed" > "$RUN/PARALLEL_HELPER_FAILED"
      exit 1
    }
    continue
  fi
  acquire_gpu
  gpu=$SELECTED_GPU
  cpuset=$SELECTED_CPU_SET
  train_one "$anchor" "$seed" "$gpu" "$cpuset" || {
    release_gpu "$gpu"
    write_state "parallel_training_anchor_${anchor}_seed_${seed}" failed "training_failed"
    printf '%s\n' "anchor=$anchor seed=$seed parallel training failed" > "$RUN/PARALLEL_HELPER_FAILED"
    exit 1
  }
  release_gpu "$gpu"
done
write_state all_parallel_jobs complete "all selected jobs checkpointed"
printf '%s\n' 'selected timing models trained on parallel helper GPU' > "$RUN/PARALLEL_HELPER_COMPLETE"
echo OPEN_DRAWER_GRASP_TIMING_PARALLEL_HELPER_COMPLETE
