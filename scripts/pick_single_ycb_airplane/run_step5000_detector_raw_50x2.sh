#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/Ask4Help-pick-airplane-48acc09f}
PYTHON=${PYTHON:-/root/Ask4Help-online-awbc/RLinf/.venv/bin/python}
RESULT_ROOT=${RESULT_ROOT:-/mnt/data/ask4help/results/pick_single_ycb_airplane/yaw_swap_v1}
CHECKPOINT=${CHECKPOINT:-$RESULT_ROOT/id_sft_no180_modelonly_step2000_plus3000_v1/maniskill_stackcube_pi05_id_sft/checkpoints/global_step_3000}
PI05_BASE=${PI05_BASE:-/mnt/data/ask4help/models/pi05_base_torch}
NORM_STATS=${NORM_STATS:-$RESULT_ROOT/assets/id_expert_norm_stats}
DATASET_ROOT=${DATASET_ROOT:-$RESULT_ROOT/id_expert_no180_98_v2/lerobot}
ASSET_DIR=${ASSET_DIR:-$RESULT_ROOT/detector_assets_step5000_all_id98_v1}
ID_OUT=${ID_OUT:-$RESULT_ROOT/detector_eval_step5000_id50_raw_v1}
OOD_OUT=${OOD_OUT:-$RESULT_ROOT/detector_eval_step5000_ood50_raw_v1}
LOG_DIR=${LOG_DIR:-/root/pick_airplane_detector_step5000_50x2_logs}

mkdir -p "$LOG_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/RLinf"

if [[ ! -f "$ASSET_DIR/manifest.json" ]]; then
  if [[ -e "$ASSET_DIR" ]]; then
    echo "partial asset directory exists; refusing to overwrite: $ASSET_DIR" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES=0 "$PYTHON" tools/build_pick_single_ycb_airplane_detector_assets.py \
    --checkpoint "$CHECKPOINT" \
    --pi05-base "$PI05_BASE" \
    --norm-stats "$NORM_STATS" \
    --dataset-root "$DATASET_ROOT" \
    --output-dir "$ASSET_DIR" \
    --knn-k 10 \
    >"$LOG_DIR/assets.log" 2>&1
fi

[[ -f "$ASSET_DIR/detector_assets.pt" ]]
[[ ! -e "$ID_OUT" ]]
[[ ! -e "$OOD_OUT" ]]

CUDA_VISIBLE_DEVICES=0 "$PYTHON" tools/evaluate_pick_single_ycb_airplane_detectors.py \
  --checkpoint "$CHECKPOINT" \
  --pi05-base "$PI05_BASE" \
  --norm-stats "$NORM_STATS" \
  --detector-assets "$ASSET_DIR/detector_assets.pt" \
  --output-dir "$ID_OUT" \
  --split id --episodes 50 --seed 50000 --execute-horizon 5 --max-episode-steps 50 \
  >"$LOG_DIR/id50.log" 2>&1 &
id_pid=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON" tools/evaluate_pick_single_ycb_airplane_detectors.py \
  --checkpoint "$CHECKPOINT" \
  --pi05-base "$PI05_BASE" \
  --norm-stats "$NORM_STATS" \
  --detector-assets "$ASSET_DIR/detector_assets.pt" \
  --output-dir "$OOD_OUT" \
  --split ood --episodes 50 --seed 60000 --execute-horizon 5 --max-episode-steps 50 \
  >"$LOG_DIR/ood50.log" 2>&1 &
ood_pid=$!

printf '%s\n' "$id_pid" >"$LOG_DIR/id50.pid"
printf '%s\n' "$ood_pid" >"$LOG_DIR/ood50.pid"
wait "$id_pid"
wait "$ood_pid"

cp "$LOG_DIR/assets.log" "$ASSET_DIR/pipeline_assets.log"
cp "$LOG_DIR/id50.log" "$ID_OUT/pipeline_eval.log"
cp "$LOG_DIR/ood50.log" "$OOD_OUT/pipeline_eval.log"
printf 'complete\n' >"$LOG_DIR/COMPLETE"
