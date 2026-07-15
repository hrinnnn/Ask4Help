#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT="${ASK4HELP_ROOT:-/root/Ask4Help}"
RLINF_ROOT="${RLINF_ROOT:-${ASK4HELP_ROOT}/RLinf}"
PYTHON="${PYTHON:-${RLINF_ROOT}/.venv/bin/python}"
: "${PI05_MODEL_PATH:?Set PI05_MODEL_PATH to the in-distribution pi0.5 checkpoint}"
: "${DIFFDAGGER_CALIBRATION_OUTPUT:?Set DIFFDAGGER_CALIBRATION_OUTPUT}"

test -x "${PYTHON}"
test -d "${PI05_MODEL_PATH}"
mkdir -p "$(dirname "${DIFFDAGGER_CALIBRATION_OUTPUT}")"

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"

exec "${PYTHON}" examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name maniskill_openpi_pi05_diffdagger_calibration \
  "$@"
