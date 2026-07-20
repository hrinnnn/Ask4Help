#!/usr/bin/env bash
set -euo pipefail

# Evaluate both possible executing policies at every paired SFT checkpoint.
# Each process loads its two VFD members on one H20, so policy A and policy B
# grids can run concurrently without sharing model memory.
ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
CHECKPOINT_ROOT_0=${CHECKPOINT_ROOT_0:?Set CHECKPOINT_ROOT_0}
CHECKPOINT_ROOT_1=${CHECKPOINT_ROOT_1:?Set CHECKPOINT_ROOT_1}
PI05_BASE=${PI05_BASE:?Set PI05_BASE}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT}
POLICY_MEMBER=${POLICY_MEMBER:?Set POLICY_MEMBER to 0 or 1}
GPU_ID=${GPU_ID:?Set GPU_ID to 0 or 1}
STEPS=${STEPS:-"250"}
EPISODES=${EPISODES:-20}
COMPUTE_VFD=${COMPUTE_VFD:-false}

if [ "${POLICY_MEMBER}" = "0" ]; then
  FIRST_ROOT=${CHECKPOINT_ROOT_0}
  SECOND_ROOT=${CHECKPOINT_ROOT_1}
elif [ "${POLICY_MEMBER}" = "1" ]; then
  FIRST_ROOT=${CHECKPOINT_ROOT_1}
  SECOND_ROOT=${CHECKPOINT_ROOT_0}
else
  echo "POLICY_MEMBER must be 0 or 1" >&2
  exit 2
fi

for step in ${STEPS}; do
  output="${OUTPUT_ROOT}/policy_${POLICY_MEMBER}/step_${step}"
  dataset="${OUTPUT_ROOT}/datasets/policy_${POLICY_MEMBER}_step_${step}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    ASK4HELP_ROOT="${ASK4HELP_ROOT}" \
    PI05_BASE="${PI05_BASE}" \
    MEMBER_0="${FIRST_ROOT}/global_step_${step}" \
    MEMBER_1="${SECOND_ROOT}/global_step_${step}" \
    NORM_STATS="${NORM_STATS}" \
    OUTPUT_DIR="${output}" \
    DATASET="${dataset}" \
    SPLIT=id SEED=10000 EPISODES="${EPISODES}" COMPUTE_VFD="${COMPUTE_VFD}" \
    bash "${ASK4HELP_ROOT}/scripts/online_awbc_plug/evaluate_checkpoint.sh"
done
