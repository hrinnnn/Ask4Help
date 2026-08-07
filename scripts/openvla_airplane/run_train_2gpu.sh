#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENVLA_ROOT="${OPENVLA_ROOT:?set OPENVLA_ROOT to the official OpenVLA checkout}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the 98-episode LeRobot dataset}"
RUN_DIR="${RUN_DIR:?set RUN_DIR to the durable experiment output}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$OPENVLA_ROOT:${PYTHONPATH:-}"
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  -m openvla_airplane.train \
  --vla-path "${VLA_PATH:-openvla/openvla-7b}" \
  --data-dir "$DATA_DIR" \
  --run-dir "$RUN_DIR" \
  --batch-size 16 \
  --max-steps 10000 \
  --save-steps 500 \
  --learning-rate 5e-4 \
  --lora-rank 32 \
  --lora-alpha 16 \
  --lora-dropout 0.0 \
  --image-aug
