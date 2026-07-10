#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_path "${ROBO_DOPAMINE_DIR}/train/qwenvl/train/train_qwen.py" "official Robo-Dopamine training code"
TASK_ID="${ONESHOT_TASK_ID:-0}"
TASK_DIR="${GRM_ONESHOT_ROOT}/libero_goal/task_$(printf '%03d' "${TASK_ID}")"
TRAIN_DIR="${TASK_DIR}/train_data"
require_path "${TRAIN_DIR}/train_jsons/finetune_data_final.json" "prepared one-shot SFT data"

source "${ROBO_DOPAMINE_CONDA}"
conda activate "${ROBO_DOPAMINE_ENV}"
cd "${ROBO_DOPAMINE_DIR}"

# The official registry's example_grm_finetune entry expects dataset/train_data.
ln -sfn "${TRAIN_DIR}" "${ROBO_DOPAMINE_DIR}/dataset/train_data"
OUTPUT_DIR="${ONESHOT_OUTPUT_DIR:-${MODELS_DIR}/Robo-Dopamine-GRM-2.0-4B-Preview-libero-goal-task$(printf '%03d' "${TASK_ID}")-oneshot}"
mkdir -p "${OUTPUT_DIR}"

torchrun --nproc_per_node="${GRM_FINETUNE_GPUS:-1}" --nnodes=1 --master_port="${GRM_FINETUNE_PORT:-29514}" \
  qwenvl/train/train_qwen.py \
  --model_name_or_path "${GRM_FINETUNE_BASE_MODEL:-${GRM_MODEL_PATH}}" \
  --tune_mm_llm True \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --dataset_use example_grm_finetune \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${ROBO_DOPAMINE_DIR}/.cache" \
  --bf16 \
  --per_device_train_batch_size "${GRM_FINETUNE_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRM_FINETUNE_GRAD_ACCUM:-4}" \
  --learning_rate 1e-5 \
  --mm_projector_lr 1e-5 \
  --vision_tower_lr 5e-7 \
  --optim adamw_torch \
  --model_max_length 32768 \
  --data_flatten False \
  --data_packing False \
  --max_pixels 76800 \
  --min_pixels 12544 \
  --base_interval 2 \
  --video_max_frames 8 \
  --video_min_frames 4 \
  --video_max_frame_pixels 1304576 \
  --video_min_frame_pixels 200704 \
  --num_train_epochs "${GRM_FINETUNE_EPOCHS:-2}" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 2 \
  --eval_strategy no \
  --deepspeed ./scripts/zero3.json 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo "One-shot adapted GRM saved to ${OUTPUT_DIR}"
