#!/usr/bin/env bash
set -euo pipefail

# Strict, single-GPU VLA-FAIL reproduction for the existing StackCube pi0.5
# checkpoint. Set MODE to stats, calibrate, id_eval, or ood_eval.

: "${MODE:?set MODE=stats|calibrate|id_eval|ood_eval}"
: "${CHECKPOINT:?path to the step-7000 checkpoint}"
: "${PI05_BASE:?path to pi05_base_torch}"
: "${NORM_STATS:?path to StackCube norm_stats.json}"
: "${DATASET_ROOT:?path to the 128-demo StackCube ID LeRobot dataset}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/RLinf/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/data/ask4help/results/stackcube_vla_fail/reproduction_v1}"
STATS_PATH="${STATS_PATH:-${RESULT_ROOT}/llmd_statistics.pt}"
CALIBRATION_DIR="${CALIBRATION_DIR:-${RESULT_ROOT}/calibration_id}"
THRESHOLDS_PATH="${THRESHOLDS_PATH:-${CALIBRATION_DIR}/thresholds.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

mkdir -p "${RESULT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export RLINF_ROOT="${ROOT}/RLinf"

case "${MODE}" in
  stats)
    "${PYTHON_BIN}" "${ROOT}/tools/build_stackcube_vla_fail_statistics.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
      --norm-stats "${NORM_STATS}" --dataset-root "${DATASET_ROOT}" \
      --output "${STATS_PATH}" --fixed-prior-seed "${FIXED_PRIOR_SEED:-0}" \
      --ridge "${LLMD_RIDGE:-1e-6}"
    ;;
  calibrate)
    "${PYTHON_BIN}" "${ROOT}/tools/evaluate_stackcube_vla_fail.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
      --norm-stats "${NORM_STATS}" --llmd-statistics "${STATS_PATH}" \
      --output-dir "${CALIBRATION_DIR}" --split id \
      --seed "${CALIBRATION_SEED:-10000}" --calibrate --delta 0.05 \
      --target-successes 20 --max-attempts "${CALIBRATION_MAX_ATTEMPTS:-200}" \
      --execute-horizon 5 --max-episode-steps 100
    ;;
  id_eval|ood_eval)
    split="${MODE%_eval}"
    seed="${EVAL_SEED:-10000}"
    if [[ "${split}" == "ood" ]]; then seed="${EVAL_SEED:-20000}"; fi
    "${PYTHON_BIN}" "${ROOT}/tools/evaluate_stackcube_vla_fail.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
      --norm-stats "${NORM_STATS}" --llmd-statistics "${STATS_PATH}" \
      --output-dir "${RESULT_ROOT}/${split}_eval" --split "${split}" \
      --episodes "${EPISODES:-50}" --seed "${seed}" --execute-horizon 5 \
      --max-episode-steps 100 --thresholds "${THRESHOLDS_PATH}"
    ;;
  *)
    echo "unknown MODE=${MODE}; use stats, calibrate, id_eval, or ood_eval" >&2
    exit 2
    ;;
esac
