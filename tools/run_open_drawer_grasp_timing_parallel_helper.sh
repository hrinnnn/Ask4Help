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
GPU=${OPEN_DRAWER_TIMING_PARALLEL_GPU:-0}
CPU_SET=${OPEN_DRAWER_TIMING_PARALLEL_CPU_SET:-0-19}
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

train_one() {
  local anchor=$1 seed=$2
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
    return 0
  fi
  if [[ -e "$out" && -n "$(find "$out" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    printf '%s\n' "refusing to reuse partial training output: $out" >&2
    return 1
  fi

  mkdir -p "$out"
  write_state "parallel_training_${condition}_seed_${seed}" running "gpu=$GPU cpu_set=$CPU_SET"
  env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ASK4HELP_RLINF_PLACEMENT="$GPU-$GPU" RLINF_RAY_ADDRESS=local \
    EMBODIED_PATH="$RL/examples/sft" PYTHONPATH="$ROOT:$RL" \
    OPEN_DRAWER_ID_DATASET="$ID_DATASET" OPEN_DRAWER_EXPERT_DATASET="$expert" \
    OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" \
    OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME=run \
    OPEN_DRAWER_TRAIN_SEED="$seed" OPEN_DRAWER_RESUME_DIR="" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    HF_DATASETS_CACHE="$RUN/runtime/hf_datasets" HF_HOME="$RUN/runtime/hf_home" \
    RAY_TMPDIR="$RAY_BASE/${condition}_${seed}" \
    TMPDIR="$TMP_BASE/${condition}_${seed}" PYTHONUNBUFFERED=1 \
    taskset -c "$CPU_SET" "$PY" "$RL/examples/sft/train_vla_sft.py" \
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
  return 0
}

write_state preflight running "gpu=$GPU cpu_set=$CPU_SET jobs=${#JOBS[@]}"
for job in "${JOBS[@]}"; do
  IFS=: read -r anchor seed <<< "$job"
  train_one "$anchor" "$seed" || {
    write_state "parallel_training_anchor_${anchor}_seed_${seed}" failed "training_failed"
    printf '%s\n' "anchor=$anchor seed=$seed parallel training failed" > "$RUN/PARALLEL_HELPER_FAILED"
    exit 1
  }
done
write_state all_parallel_jobs complete "all selected jobs checkpointed"
printf '%s\n' 'selected timing models trained on parallel helper GPU' > "$RUN/PARALLEL_HELPER_COMPLETE"
echo OPEN_DRAWER_GRASP_TIMING_PARALLEL_HELPER_COMPLETE
