#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT="${ASK4HELP_ROOT:-/root/Ask4Help}"
RLINF_ROOT="${RLINF_ROOT:-${ASK4HELP_ROOT}/RLinf}"
PYTHON="${PYTHON:-${RLINF_ROOT}/.venv/bin/python}"
: "${PI05_MODEL_PATH:?Set PI05_MODEL_PATH to the student checkpoint}"
: "${DIFFDAGGER_EXPERT_MODEL_PATH:?Set DIFFDAGGER_EXPERT_MODEL_PATH}"
: "${DIFFDAGGER_CALIBRATION_PATH:?Set DIFFDAGGER_CALIBRATION_PATH}"

test -x "${PYTHON}"
test -d "${PI05_MODEL_PATH}"
test -d "${DIFFDAGGER_EXPERT_MODEL_PATH}"
test -f "${DIFFDAGGER_CALIBRATION_PATH}"

cd "${RLINF_ROOT}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}:${PYTHONPATH:-}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"

exec "${PYTHON}" examples/embodiment/train_embodied_agent.py \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name maniskill_ppo_openpi_pi05_flow_sde_diffdagger \
  runner.max_epochs=1 \
  runner.max_steps=1 \
  runner.val_check_interval=-1 \
  runner.save_interval=-1 \
  env.train.total_num_envs=4 \
  env.train.rollout_epoch=1 \
  env.train.max_steps_per_rollout_epoch=10 \
  env.eval.total_num_envs=4 \
  actor.micro_batch_size=1 \
  actor.global_batch_size=4 \
  "$@"
