#!/usr/bin/env bash
set -euo pipefail

ASK4HELP_ROOT=${ASK4HELP_ROOT:-/root/Ask4Help}
RLINF_ROOT=${RLINF_ROOT:-"${ASK4HELP_ROOT}/RLinf"}
ID_DATASET=${ID_DATASET:?Set ID_DATASET}
NORM_STATS=${NORM_STATS:?Set NORM_STATS}
PI05_BASE=${PI05_BASE:?Set PI05_BASE}
RESULT_ROOT=${RESULT_ROOT:?Set RESULT_ROOT to a new OSS result directory}
MAX_STEPS=${MAX_STEPS:-2000}
SAVE_INTERVAL=${SAVE_INTERVAL:-250}
MEMBER_0_RESUME=${MEMBER_0_RESUME:-}
MEMBER_1_RESUME=${MEMBER_1_RESUME:-}

test -d "${ID_DATASET}"
test -s "${NORM_STATS}"
test -s "${PI05_BASE}/model.safetensors"
test ! -e "${RESULT_ROOT}/member_0"
test ! -e "${RESULT_ROOT}/member_1"
mkdir -p "${RESULT_ROOT}" /root/ask4help_stackcube_sft

export PYTHONPATH="${RLINF_ROOT}:${ASK4HELP_ROOT}:${PYTHONPATH:-}"
"${RLINF_ROOT}/.venv/bin/ray" stop --force || true
"${RLINF_ROOT}/.venv/bin/ray" start --head --num-cpus=32 --num-gpus=2 \
  --disable-usage-stats > /root/ask4help_stackcube_sft/ray_start.log 2>&1

"${RLINF_ROOT}/.venv/bin/python" - <<'PY'
import ray


@ray.remote(num_cpus=0)
def import_rlinf():
    import rlinf

    return rlinf.__file__


ray.init(address="auto")
print(f"Ray worker RLinf import: {ray.get(import_rlinf.remote())}")
ray.shutdown()
PY

launch_member() {
  local member=$1
  local gpu=$2
  local seed=$3
  local resume=$4
  local output="${RESULT_ROOT}/member_${member}"
  local log="/root/ask4help_stackcube_sft/member_${member}.log"
  local pid="/tmp/ask4help_stackcube_member_${member}.pid"
  local resume_args=()
  if [ -n "${resume}" ]; then
    resume_args+=(RESUME_DIR="${resume}")
  fi
  nohup env \
    ASK4HELP_ROOT="${ASK4HELP_ROOT}" RLINF_ROOT="${RLINF_ROOT}" EXTERNAL_RAY=1 \
    GPU_ID="${gpu}" SEED="${seed}" ID_DATASET="${ID_DATASET}" \
    NORM_STATS="${NORM_STATS}" PI05_BASE="${PI05_BASE}" OUTPUT_DIR="${output}" \
    MAX_STEPS="${MAX_STEPS}" SAVE_INTERVAL="${SAVE_INTERVAL}" \
    "${resume_args[@]}" \
    bash "${ASK4HELP_ROOT}/scripts/stackcube_id/run_member_sft.sh" > "${log}" 2>&1 &
  echo $! > "${pid}"
}

launch_member 0 0 1000 "${MEMBER_0_RESUME}"
launch_member 1 1 1001 "${MEMBER_1_RESUME}"

cat > "${RESULT_ROOT}/launch_manifest.json" <<EOF
{
  "start": "$(date -Iseconds)",
  "dataset": "${ID_DATASET}",
  "norm_stats": "${NORM_STATS}",
  "pi05_base": "${PI05_BASE}",
  "max_steps": ${MAX_STEPS},
  "save_interval": ${SAVE_INTERVAL},
  "member_0_pid": $(cat /tmp/ask4help_stackcube_member_0.pid),
  "member_1_pid": $(cat /tmp/ask4help_stackcube_member_1.pid),
  "member_0_seed": 1000,
  "member_1_seed": 1001,
  "member_0_resume": "${MEMBER_0_RESUME}",
  "member_1_resume": "${MEMBER_1_RESUME}"
}
EOF

printf 'member0 pid=%s log=%s\n' "$(cat /tmp/ask4help_stackcube_member_0.pid)" /root/ask4help_stackcube_sft/member_0.log
printf 'member1 pid=%s log=%s\n' "$(cat /tmp/ask4help_stackcube_member_1.pid)" /root/ask4help_stackcube_sft/member_1.log
