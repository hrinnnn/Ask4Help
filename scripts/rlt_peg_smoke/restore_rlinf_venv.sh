#!/usr/bin/env bash
set -euo pipefail

RLINF_ROOT=${RLINF_ROOT:-/root/Ask4Help/RLinf}
ARCHIVE=${ARCHIVE:-/mnt/data/ask4help/env_snapshots/20260707_114911/rlinf_venv.tar.zst}
DESTINATION="${RLINF_ROOT}/.venv"

command -v zstd >/dev/null || {
  echo "zstd is required to restore ${ARCHIVE}" >&2
  exit 1
}
[[ -d "${RLINF_ROOT}" ]] || {
  echo "RLinf source directory does not exist: ${RLINF_ROOT}" >&2
  exit 1
}
[[ -f "${ARCHIVE}" ]] || {
  echo "RLinf environment archive does not exist: ${ARCHIVE}" >&2
  exit 1
}
[[ ! -e "${DESTINATION}" ]] || {
  echo "Refusing to overwrite existing environment: ${DESTINATION}" >&2
  exit 1
}

staging_dir=$(mktemp -d)
trap 'rm -rf "${staging_dir}"' EXIT

tar --use-compress-program=unzstd -xf "${ARCHIVE}" -C "${staging_dir}"
restored_venv=$(find "${staging_dir}" -type d -name .venv -print -quit)
[[ -n "${restored_venv}" ]] || {
  echo "The archive did not contain a .venv directory" >&2
  exit 1
}

mv "${restored_venv}" "${DESTINATION}"
echo "Restored ${DESTINATION} from ${ARCHIVE}"
