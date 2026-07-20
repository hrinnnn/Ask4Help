#!/usr/bin/env bash
set -euo pipefail

# A deliberately unreachable threshold makes online mode run policy-only while
# retaining the same per-chunk pi0.5 rollout path used during VFD collection.
ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
PYTHON=${PYTHON:-"${RLINF_ROOT}/.venv/bin/python"}
MEMBER_0=${MEMBER_0:?Set MEMBER_0}
MEMBER_1=${MEMBER_1:?Set MEMBER_1}
PI05_BASE=${PI05_BASE:?Set PI05_BASE}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}
DATASET=${DATASET:?Set DATASET}
SPLIT=${SPLIT:-id}
SEED=${SEED:-10000}
EPISODES=${EPISODES:-20}
COMPUTE_VFD=${COMPUTE_VFD:-true}

mkdir -p "${OUTPUT_DIR}"
POLICY_ONLY_THRESHOLD="${OUTPUT_DIR}/policy_only_threshold.json"
printf '{"threshold": 1e30, "quantile": 1.0, "calibration_count": 0}\n' > "${POLICY_ONLY_THRESHOLD}"
export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
extra_args=()
if [ "${COMPUTE_VFD}" = "false" ]; then
  extra_args+=(--no-compute-vfd)
fi
"${PYTHON}" "${ASK4HELP_ROOT}/tools/maniskill_pi05_vfd_online_awbc.py" \
  --mode online --task plug --split "${SPLIT}" \
  --member-0 "${MEMBER_0}" --member-1 "${MEMBER_1}" --pi05-base "${PI05_BASE}" --norm-stats "${NORM_STATS}" \
  --threshold-path "${POLICY_ONLY_THRESHOLD}" --repo-id "${DATASET}" --output-dir "${OUTPUT_DIR}" \
  --episodes "${EPISODES}" --seed "${SEED}" --num-action-samples 5 --no-save-videos "${extra_args[@]}"
