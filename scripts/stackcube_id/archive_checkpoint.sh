#!/usr/bin/env bash
set -euo pipefail

SOURCE=${SOURCE:?Set SOURCE to a complete RLinf global_step checkpoint directory}
ARCHIVE=${ARCHIVE:?Set ARCHIVE to a never-before-used OSS directory}
test -d "${SOURCE}/actor"
test -s "${SOURCE}/actor/model_state_dict/full_weights.pt"
test ! -e "${ARCHIVE}"

mkdir -p "$(dirname "${ARCHIVE}")"
cp -a "${SOURCE}" "${ARCHIVE}"

source_hash=$(sha256sum "${SOURCE}/actor/model_state_dict/full_weights.pt" | awk '{print $1}')
archive_hash=$(sha256sum "${ARCHIVE}/actor/model_state_dict/full_weights.pt" | awk '{print $1}')
if [ "${source_hash}" != "${archive_hash}" ]; then
  echo "checkpoint archive hash mismatch" >&2
  exit 1
fi
printf '%s  actor/model_state_dict/full_weights.pt\n' "${archive_hash}" \
  > "${ARCHIVE}/full_weights.pt.sha256"
printf '{"source":"%s","archive":"%s","sha256":"%s","archived_at":"%s"}\n' \
  "${SOURCE}" "${ARCHIVE}" "${archive_hash}" "$(date -Iseconds)" \
  > "${ARCHIVE}/archive_manifest.json"

