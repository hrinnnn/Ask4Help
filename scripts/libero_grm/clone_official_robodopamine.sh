#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

mkdir -p "$(dirname "${ROBO_DOPAMINE_DIR}")"
if [[ -d "${ROBO_DOPAMINE_DIR}/.git" ]]; then
  git -C "${ROBO_DOPAMINE_DIR}" pull --ff-only
else
  git clone https://github.com/FlagOpen/Robo-Dopamine.git "${ROBO_DOPAMINE_DIR}"
fi
echo "Official Robo-Dopamine checkout: ${ROBO_DOPAMINE_DIR}"
