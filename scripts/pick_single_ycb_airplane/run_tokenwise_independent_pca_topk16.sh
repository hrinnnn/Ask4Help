#!/usr/bin/env bash
# Reproducible stages for airplane independent-token PCA TopK-16.
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
GPU_ID=${GPU_ID:-0}
PCA_COMPUTE_DEVICE=${PCA_COMPUTE_DEVICE:-cuda}
PI05_BASE=${PI05_BASE:?Set PI05_BASE to the pi0.5 pretrained model directory}
CHECKPOINT=${CHECKPOINT:-/mnt/data/ask4help/results/pick_single_ycb_airplane/yaw_swap_v1/id_sft_no180_modelonly_step2000_plus3000_v1/maniskill_stackcube_pi05_id_sft/checkpoints/global_step_3000}
NORM_STATS=${NORM_STATS:-/mnt/data/ask4help/results/pick_single_ycb_airplane/yaw_swap_v1/assets/id_expert_norm_stats}
ID_DATASET=${ID_DATASET:-/mnt/data/ask4help/results/pick_single_ycb_airplane/yaw_swap_v1/id_expert_no180_98_v2/lerobot}
RESULT_ROOT=${RESULT_ROOT:-/mnt/data/ask4help/results/pick_single_ycb_airplane/tokenwise_pca_topk16_v1}
FEATURE_DIR=${FEATURE_DIR:-"${RESULT_ROOT}/id_prefix_features"}
ASSET_DIR=${ASSET_DIR:-"${RESULT_ROOT}/assets_independent_rank1000"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"${RESULT_ROOT}/rollouts_50id_50ood_h250"}
SCAN_PATH=${SCAN_PATH:-"${RESULT_ROOT}/posthoc_threshold_scan.json"}
RENDER_DIR=${RENDER_DIR:-"${RESULT_ROOT}/representative_score_videos"}
STAGE=${1:?Usage: $0 {features|assets|evaluate|scan|render}}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ASK4HELP_ROOT}:${RLINF_ROOT}:${PYTHONPATH:-}"

case "${STAGE}" in
  features)
    exec "${PYTHON}" "${ASK4HELP_ROOT}/tools/build_pick_single_ycb_airplane_tokenwise_pca_features.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" --norm-stats "${NORM_STATS}" \
      --dataset-root "${ID_DATASET}" --output-dir "${FEATURE_DIR}" --expected-observations 9109
    ;;
  assets)
    exec "${PYTHON}" "${ASK4HELP_ROOT}/tools/build_pick_single_ycb_airplane_tokenwise_pca_assets.py" \
      --feature-dir "${FEATURE_DIR}" --output-dir "${ASSET_DIR}" \
      --principal-dim 1000 --min-observations 1001 --token-block-size 8 \
      --compute-device "${PCA_COMPUTE_DEVICE}"
    ;;
  evaluate)
    exec "${PYTHON}" "${ASK4HELP_ROOT}/tools/evaluate_pick_single_ycb_airplane_tokenwise_pca.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" --norm-stats "${NORM_STATS}" \
      --assets-dir "${ASSET_DIR}" --output-dir "${ROLLOUT_DIR}" \
      --episodes-per-split 50 --id-seed 50000 --ood-seed 60000 --execute-horizon 5 --max-episode-steps 250
    ;;
  scan)
    exec "${PYTHON}" "${ASK4HELP_ROOT}/tools/sweep_pick_single_ycb_airplane_tokenwise_pca.py" \
      --episodes "${ROLLOUT_DIR}/episodes.json" --output "${SCAN_PATH}"
    ;;
  render)
    exec "${PYTHON}" "${ASK4HELP_ROOT}/tools/render_pick_single_ycb_airplane_tokenwise_pca_video.py" \
      --episodes "${ROLLOUT_DIR}/episodes.json" --scan "${SCAN_PATH}" --output-dir "${RENDER_DIR}"
    ;;
  *)
    echo "unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac
