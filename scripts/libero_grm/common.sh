#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ASK4HELP_ROOT:-}" ]]; then
  if [[ -d /root/Ask4Help ]]; then
    ASK4HELP_ROOT=/root/Ask4Help
  elif [[ -d /root/vla_rl_workspace/RLinf ]]; then
    ASK4HELP_ROOT=/root/vla_rl_workspace
  else
    ASK4HELP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  fi
fi

RLINF_DIR="${RLINF_DIR:-${ASK4HELP_ROOT}/RLinf}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/ask4help}"
RESULTS_DIR="${RESULTS_DIR:-${DATA_ROOT}/results}"
MODELS_DIR="${MODELS_DIR:-${DATA_ROOT}/models}"
ASSETS_DIR="${ASSETS_DIR:-${DATA_ROOT}/assets}"
LIBERO_DEMO_ROOT="${LIBERO_DEMO_ROOT:-${DATA_ROOT}/libero_demos}"
LIBERO_SPATIAL_DEMO_DIR="${LIBERO_SPATIAL_DEMO_DIR:-${LIBERO_DEMO_ROOT}/libero_spatial}"
LIBERO_GOAL_DEMO_DIR="${LIBERO_GOAL_DEMO_DIR:-${LIBERO_DEMO_ROOT}/libero_goal}"
GRM_GOAL_BANK_DIR="${GRM_GOAL_BANK_DIR:-${ASSETS_DIR}/grm_goal_bank/libero_spatial_demo_final}"
GRM_LIBERO_GOAL_BANK_DIR="${GRM_LIBERO_GOAL_BANK_DIR:-${ASSETS_DIR}/grm_goal_bank/libero_goal_demo_final}"
GRM_ONESHOT_ROOT="${GRM_ONESHOT_ROOT:-${ASSETS_DIR}/grm_oneshot}"

PI05_MODEL_PATH="${PI05_MODEL_PATH:-${MODELS_DIR}/RLinf-Pi05-LIBERO-SFT}"
GRM_MODEL_PATH="${GRM_MODEL_PATH:-${MODELS_DIR}/Robo-Dopamine-GRM-2.0-4B-Preview}"
GRM_MODEL_NAME="${GRM_MODEL_NAME:-tanhuajie2001/Robo-Dopamine-GRM-2.0-4B-Preview}"
GRM_HOST="${GRM_HOST:-0.0.0.0}"
GRM_PORT="${GRM_PORT:-8000}"
GRM_ENDPOINT="${GRM_ENDPOINT:-http://127.0.0.1:${GRM_PORT}/v1/chat/completions}"
GRM_GPU="${GRM_GPU:-0}"
TRAIN_GPU_RANK="${TRAIN_GPU_RANK:-1}"
ROBO_DOPAMINE_DIR="${ROBO_DOPAMINE_DIR:-${ASK4HELP_ROOT}/external/Robo-Dopamine}"

ROBO_DOPAMINE_ENV="${ROBO_DOPAMINE_ENV:-robo-dopamine}"
ROBO_DOPAMINE_CONDA="${ROBO_DOPAMINE_CONDA:-/opt/conda/etc/profile.d/conda.sh}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

activate_rlinf() {
  require_path "${RLINF_DIR}" "RLinf checkout"
  require_path "${RLINF_DIR}/.venv/bin/activate" "RLinf venv"
  export PYTHONPATH="${PYTHONPATH:-}"
  # shellcheck disable=SC1091
  source "${RLINF_DIR}/.venv/bin/activate"
}

export_rll_env() {
  export PATH="${RLINF_DIR}/.venv/bin:${PATH}"
  export PYTHONPATH="${RLINF_DIR}:${PYTHONPATH:-}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export RAY_DEBUG="${RAY_DEBUG:-legacy}"
  export ROBOT_PLATFORM="${ROBOT_PLATFORM:-LIBERO}"
  export EMBODIED_PATH="${EMBODIED_PATH:-${RLINF_DIR}/examples/embodiment}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
}
