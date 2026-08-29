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
RAY_BASE=${OPEN_DRAWER_TIMING_RAY_BASE:-/tmp/rayod}
TMP_BASE=${OPEN_DRAWER_TIMING_TMP_BASE:-/tmp/timod}
STEPS=${OPEN_DRAWER_TIMING_TRAIN_STEPS:-2500}
SEEDS=(9301 9302 9303)
# A live resource audit found GPU0 occupied by collection and GPUs1--3 owned
# by other workloads; run timing updates sequentially on the verified idle GPU4.
GPU=4
CPU_SET=80-99
PRIORITY_OOD20_GATE=${OPEN_DRAWER_PRIORITY_OOD20_GATE:-1}
ANCHORS=(0 50 80 120 160 220)
STATE=$RUN/training_pipeline_state.json
LOG=$RUN/training_controller.log

if [[ ! -e "$BUDGET_ROOT/BUDGET_AUDIT_PASS" ]]; then
  printf '%s\n' '{"stage":"training","status":"failed","detail":"missing exact budget audit"}' > "$STATE"
  exit 1
fi
mkdir -p "$RUN/training" "$RUN/training_logs" "$RAY_BASE" "$TMP_BASE"

write_state() {
  printf '%s\n' "{\"format\":\"open_drawer_grasp_timing_training_v1\",\"stage\":\"$1\",\"status\":\"$2\",\"detail\":\"${3:-}\",\"updated_at\":\"$(date -Is)\"}" > "$STATE"
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
  local name=run
  local checkpoint="$out/$name/checkpoints/global_step_$STEPS/actor/model_state_dict/full_weights.pt"
  local log="$RUN/training_logs/${condition}_seed_${seed}.log"
  if [[ ! -d "$expert" ]]; then
    printf '%s\n' "missing budget dataset: $expert" >&2
    return 1
  fi
  local config_path="$RL/examples/sft/config/open_drawer_retrieve_place_dagger_sft_openpi_pi05.yaml"
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
  write_state "training_${condition}_seed_${seed}" running "gpu=$gpu"
  env CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ASK4HELP_RLINF_PLACEMENT="$gpu-$gpu" RLINF_RAY_ADDRESS=local \
    EMBODIED_PATH="$RL/examples/sft" PYTHONPATH="$ROOT:$RL" \
    OPEN_DRAWER_ID_DATASET="$ID_DATASET" OPEN_DRAWER_EXPERT_DATASET="$expert" \
    OPEN_DRAWER_ID_NORM_STATS="$NORM" OPEN_DRAWER_PI05_MODEL_PATH="$MODEL" \
    OPEN_DRAWER_RUN_ROOT="$out" OPEN_DRAWER_EXPERIMENT_NAME="$name" \
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
  echo "$pid" > "$out/train.pid"
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

write_state preflight running "exact_budget=$BUDGET_ROOT"
for anchor in "${ANCHORS[@]}"; do
  write_state "training_anchor_${anchor}" running "sequential_seeds=${SEEDS[*]} gpu=$GPU"
  for seed in "${SEEDS[@]}"; do
    train_one "$anchor" "$seed" "$GPU" "$CPU_SET" >> "$LOG" 2>&1 || {
      write_state "training_anchor_${anchor}_seed_${seed}" failed "training failed"
      printf '%s\n' "anchor=$anchor seed=$seed training failed" > "$RUN/TRAINING_FAILED"
      exit 1
    }
  done
done
write_state all_timing_training complete "all anchors and seeds checkpointed"
printf '%s\n' 'all timing anchors trained with three seeds' > "$RUN/TIMING_TRAINING_COMPLETE"
echo OPEN_DRAWER_GRASP_TIMING_TRAINING_COMPLETE
