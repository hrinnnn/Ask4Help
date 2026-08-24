#!/usr/bin/env python3
"""Wait for the next complete ID checkpoint and run one 20-episode probe."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1")
CHECKPOINT = RUN / "id_training_v1/formal_10000_retry8/id_sft_10000_retry8_weights_only/checkpoints/global_step_4000"
OUTPUT = RUN / "id_checkpoint_probe_20id/step_4000_retry8"
PYTHON = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python")
MODEL = Path("/data/zhaozhixuan/Ask4Help-open-drawer/results/model_cache/pi05_base_pytorch_v1")
NORM = RUN / "datasets/id_v1_retry1/norm_stats.json"
LOG = RUN / "logs/id_probe_step4000_retry8.log"
PID_FILE = RUN / "id_checkpoint_probe_20id/step_4000_retry8.pid"


def checkpoint_ready() -> bool:
    weights = CHECKPOINT / "actor/model_state_dict/full_weights.pt"
    dcp = CHECKPOINT / "actor/dcp_checkpoint"
    if not weights.is_file() or weights.stat().st_size < 1024 * 1024 or not dcp.is_dir():
        return False
    size = weights.stat().st_size
    time.sleep(2)
    return weights.is_file() and weights.stat().st_size == size


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    while not checkpoint_ready():
        time.sleep(300)
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite existing probe output: {OUTPUT}")
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "3",
            "PYTHONPATH": f"{ROOT}:{ROOT / 'RLinf'}:{env.get('PYTHONPATH', '')}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        "taskset",
        "-c",
        "40-59",
        str(PYTHON),
        str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
        "--checkpoint",
        str(CHECKPOINT),
        "--pi05-base",
        str(MODEL),
        "--norm-stats",
        str(NORM),
        "--output-dir",
        str(OUTPUT),
        "--split",
        "id",
        "--episodes",
        "20",
        "--seed",
        "24000",
        "--execute-horizon",
        "5",
        "--max-episode-steps",
        "200",
    ]
    with LOG.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    summary = OUTPUT / "summary.json"
    videos = list((OUTPUT / "videos").glob("*.mp4")) if (OUTPUT / "videos").is_dir() else []
    if not summary.is_file() or len(videos) != 20:
        raise RuntimeError(f"incomplete 20-ID probe evidence: summary={summary.is_file()} videos={len(videos)}")
    payload = json.loads(summary.read_text())
    (OUTPUT / "ID_PROBE_COMPLETE").write_text(
        json.dumps({"step": 4000, "episodes": payload.get("episodes"), "successes": payload.get("successes"), "videos": len(videos), "returncode": result.returncode}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
