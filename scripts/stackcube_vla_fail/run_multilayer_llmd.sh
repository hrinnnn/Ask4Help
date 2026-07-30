#!/usr/bin/env bash
set -euo pipefail

# Offline, single-GPU layer study.  This intentionally leaves the existing
# final-layer VLA-FAIL asset untouched and records three Action Expert probes
# plus VLM-middle and the final VLM-to-action bridge in one forward pass.

: "${MODE:?set MODE=stats|calibrate|id_eval|ood_eval}"
: "${CHECKPOINT:?path to the step-7000 checkpoint}"
: "${PI05_BASE:?path to pi05_base_torch}"
: "${NORM_STATS:?path to StackCube norm_stats.json}"
: "${DATASET_ROOT:?path to the 128-demo StackCube ID LeRobot dataset}"
: "${FINAL_BASELINE_STATS:?strict existing final-layer VLA-FAIL statistics asset}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/RLinf/.venv/bin/python}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/data/ask4help/results/stackcube_vla_fail/multilayer_llmd_step7000}"
ASSET_DIR="${ASSET_DIR:-${RESULT_ROOT}/assets}"
STATISTICS="${STATISTICS:-${ASSET_DIR}/multilayer_statistics.pt}"
THRESHOLDS="${THRESHOLDS:-${RESULT_ROOT}/calibration_id/thresholds.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export RLINF_ROOT="${ROOT}/RLinf"

case "${MODE}" in
  stats)
    "${PYTHON_BIN}" "${ROOT}/tools/build_stackcube_multilayer_llmd_statistics.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
      --norm-stats "${NORM_STATS}" --dataset-root "${DATASET_ROOT}" \
      --final-baseline-statistics "${FINAL_BASELINE_STATS}" --output-dir "${ASSET_DIR}"
    ;;
  calibrate)
    "${PYTHON_BIN}" "${ROOT}/tools/evaluate_stackcube_multilayer_llmd.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
      --norm-stats "${NORM_STATS}" --statistics "${STATISTICS}" \
      --output-dir "${RESULT_ROOT}/calibration_id" --split id --seed "${CALIBRATION_SEED:-10000}" \
      --calibrate --target-successes 20 --max-attempts "${CALIBRATION_MAX_ATTEMPTS:-200}" --delta 0.05
    ;;
  id_eval|ood_eval)
    split="${MODE%_eval}"
    seed="${EVAL_SEED:-10000}"
    if [[ "${split}" == "ood" ]]; then seed="${EVAL_SEED:-20000}"; fi
    "${PYTHON_BIN}" "${ROOT}/tools/evaluate_stackcube_multilayer_llmd.py" \
      --checkpoint "${CHECKPOINT}" --pi05-base "${PI05_BASE}" \
      --norm-stats "${NORM_STATS}" --statistics "${STATISTICS}" \
      --output-dir "${RESULT_ROOT}/${split}_eval" --split "${split}" --seed "${seed}" \
      --episodes "${EPISODES:-50}" --thresholds "${THRESHOLDS}"
    ;;
  *)
    echo "unknown MODE=${MODE}; use stats, calibrate, id_eval, or ood_eval" >&2
    exit 2
    ;;
esac
