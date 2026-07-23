#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ROBO_DOPAMINE_DIR=${ROBO_DOPAMINE_DIR:-"${ASK4HELP_ROOT}/external/Robo-Dopamine"}
GRM_PYTHON=${GRM_PYTHON:-/opt/conda/envs/robo-dopamine/bin/python}
TRAIN_ROOT="${RESULT_ROOT}/robodopamine_adaptation/train_data_local_video_v2"
ADAPTER_DIR="${RESULT_ROOT}/robodopamine_adaptation/stackcube-grm-lora"
test -x "${GRM_PYTHON}"
test -s "${TRAIN_ROOT}/train_jsons/finetune_data_final.json"
test -d "${GRM_MODEL}"
mkdir -p "${ADAPTER_DIR}"

"${PYTHON}" "${ASK4HELP_ROOT}/tools/patch_robo_dopamine_lora_targets.py" --repo "${ROBO_DOPAMINE_DIR}" --manifest "${ADAPTER_DIR}/official_trainer_patch.json"
# The upstream example config resolves `./dataset/...` from `train/`, not from
# the repository root. Keep the generated pair data outside the checkout and
# expose it at that exact upstream-relative location.
mkdir -p "${ROBO_DOPAMINE_DIR}/train/dataset"
ln -sfn "${TRAIN_ROOT}" "${ROBO_DOPAMINE_DIR}/train/dataset/train_data"
cd "${ROBO_DOPAMINE_DIR}/train"
CUDA_VISIBLE_DEVICES=${GRM_GPU:-1} "${GRM_PYTHON}" qwenvl/train/train_qwen.py \
  --model_name_or_path "${GRM_MODEL}" --dataset_use example_grm_finetune --output_dir "${ADAPTER_DIR}" --cache_dir "${ROBO_DOPAMINE_DIR}/.cache" \
  --bf16 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --learning_rate 1e-5 --optim adamw_torch \
  --model_max_length 32768 --data_flatten False --data_packing False --max_pixels 76800 --min_pixels 12544 --base_interval 2 \
  --video_max_frames 8 --video_min_frames 4 --video_max_frame_pixels 1304576 --video_min_frame_pixels 200704 \
  --num_train_epochs 2 --warmup_ratio 0.03 --lr_scheduler_type cosine --weight_decay 0.01 --logging_steps 10 --save_steps 200 --save_total_limit 2 --eval_strategy no \
  --lora_enable True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
  2>&1 | tee "${ADAPTER_DIR}/train.log"
test -f "${ADAPTER_DIR}/adapter_config.json"
sha256sum "${ADAPTER_DIR}/adapter_model.safetensors" >"${ADAPTER_DIR}/adapter.sha256" 2>/dev/null || sha256sum "${ADAPTER_DIR}/adapter_model.bin" >"${ADAPTER_DIR}/adapter.sha256"
