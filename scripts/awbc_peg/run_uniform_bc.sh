#!/usr/bin/env bash
set -euo pipefail

export AWBC_MODE=uniform
exec "$(dirname "$0")/run_awbc_sft.sh"
