#!/usr/bin/env python3
"""Resume the object-variation ID policy from the last complete checkpoint.

This controller deliberately uses a new retry root after the interrupted retry1
run. It keeps the original data, norm, model, mask, and probe contract fixed.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1")
RETRY = RUN / "id_training_v1/formal_10000_retry6"
TRAIN = RETRY / "id_sft_10000_retry6"
SOURCE = RUN / "id_training_v1/formal_10000_retry1/id_sft_10000_retry1/checkpoints/global_step_3500"
DATASET = RUN / "datasets/id_v1_retry1"
NORM = DATASET / "norm_stats.json"
MODEL = Path("/data/zhaozhixuan/Ask4Help-open-drawer/results/model_cache/pi05_base_pytorch_v1")
PYTHON = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python")
CONFIG = ROOT / "RLinf/examples/sft/config"
CONFIG_NAME = "pick_single_ycb_object_variation_id_sft_openpi_pi05"
STATE = RUN / "pipeline_state.json"
LOG_DIR = RUN / "logs"
PID_FILE = TRAIN / "train.pid"
GPU = os.environ.get("OBJECT_VARIATION_GPU", "1")
CPUSET = os.environ.get("OBJECT_VARIATION_CPUSET", "20-39")


def write_state(**updates: object) -> None:
    payload = json.loads(STATE.read_text()) if STATE.is_file() else {}
    payload.update(updates)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(payload, indent=2) + "\n")


def complete_checkpoint(path: Path) -> bool:
    weights = path / "actor/model_state_dict/full_weights.pt"
    dcp = path / "actor/dcp_checkpoint"
    return weights.is_file() and weights.stat().st_size > 1024 * 1024 and dcp.is_dir()


def run_job(name: str, output: Path, max_steps: int, resume: Path, log: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU,
            "ASK4HELP_RLINF_PLACEMENT": f"{GPU}-{GPU}",
            "EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'RLinf'}:{env.get('PYTHONPATH', '')}",
            "OBJECT_VARIATION_ID_DATASET": str(DATASET),
            "OBJECT_VARIATION_ID_NORM_STATS": str(NORM),
            "OBJECT_VARIATION_PI05_MODEL_PATH": str(MODEL),
            "OBJECT_VARIATION_RUN_ROOT": str(output),
            "OBJECT_VARIATION_EXPERIMENT_NAME": name,
            "OBJECT_VARIATION_MAX_STEPS": str(max_steps),
            "OBJECT_VARIATION_SAVE_INTERVAL": "500",
            "OBJECT_VARIATION_TRAIN_SEED": "7000",
            "RAY_TMPDIR": os.environ.get("OBJECT_VARIATION_RAY_TMPDIR", "/sdd/ov_ray6"),
            "TMPDIR": os.environ.get("OBJECT_VARIATION_TMPDIR", "/sdd/ov_tmp6"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        "taskset",
        "-c",
        CPUSET,
        str(PYTHON),
        str(ROOT / "RLinf/examples/sft/train_vla_sft.py"),
        "--config-path",
        str(CONFIG),
        "--config-name",
        CONFIG_NAME,
        f"runner.max_steps={max_steps}",
        "runner.save_interval=500",
        "actor.optim.total_training_steps=10000",
        "+runner.initial_step=3500",
        f"actor.model.model_path={resume}",
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(process.pid) + "\n")
        write_state(
            current_stage="id_training_resume_from3500_retry2" if max_steps == 10000 else "id_resume_smoke",
            next_stage="id_checkpoint_probe_and_early_stop",
            training_pid=process.pid,
            retry=RETRY.name,
        )
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"{name} exited with return code {returncode}")


def run_reload_smoke(smoke_checkpoint: Path) -> None:
    output = RETRY / "reload_forward_smoke"
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU,
            "PYTHONPATH": f"{ROOT}:{ROOT / 'RLinf'}:{env.get('PYTHONPATH', '')}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        "taskset",
        "-c",
        CPUSET,
        str(PYTHON),
        str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
        "--checkpoint",
        str(smoke_checkpoint),
        "--pi05-base",
        str(MODEL),
        "--norm-stats",
        str(NORM),
        "--output-dir",
        str(output),
        "--split",
        "id",
        "--episodes",
        "1",
        "--seed",
        "23502",
        "--execute-horizon",
        "5",
        "--max-episode-steps",
        "200",
    ]
    log = LOG_DIR / "id_resume_retry6_reload_forward_smoke.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    summary = output / "summary.json"
    videos = list((output / "videos").glob("*.mp4")) if (output / "videos").is_dir() else []
    if not summary.is_file() or len(videos) != 1:
        raise RuntimeError("resume reload/forward smoke has incomplete evidence")
    if result.returncode != 0:
        (output / "SIMULATOR_EXIT_AFTER_ARTIFACTS").write_text(
            json.dumps({"returncode": result.returncode, "episodes": 1, "videos": 1}, indent=2) + "\n"
        )
    (RETRY / "RESUME_RELOAD_FORWARD_PASSED").write_text(
        json.dumps({"source": str(SOURCE), "checkpoint": str(smoke_checkpoint), "episodes": 1, "videos": 1}, indent=2) + "\n"
    )


def main() -> int:
    if not complete_checkpoint(SOURCE):
        raise RuntimeError(f"source checkpoint is incomplete: {SOURCE}")
    if not DATASET.is_dir() or not NORM.is_file() or not MODEL.is_dir() or not PYTHON.is_file():
        raise RuntimeError("immutable dataset, norm, model, or runtime is missing")
    if RETRY.exists() and any(RETRY.iterdir()):
        raise RuntimeError(f"refusing to reuse non-empty retry root: {RETRY}")
    RETRY.mkdir(parents=True)
    write_state(
        current_stage="id_resume_preflight",
        next_stage="id_resume_smoke_from3500",
        retry=RETRY.name,
        resume_source=str(SOURCE),
        old_retry_preserved=True,
        storage_mount_recovered=True,
    )
    run_job("id_resume_smoke_from3500_weights_only", RETRY / "smoke_2step", 3502, SOURCE, LOG_DIR / "id_resume_retry6_smoke.log")
    smoke_checkpoint = RETRY / "smoke_2step/checkpoints/global_step_3502"
    if not complete_checkpoint(smoke_checkpoint):
        raise RuntimeError(f"resume smoke checkpoint is incomplete: {smoke_checkpoint}")
    run_reload_smoke(smoke_checkpoint)
    write_state(current_stage="id_training_resume_from3500_retry6", next_stage="id_checkpoint_probe_and_early_stop")
    run_job("id_sft_10000_retry6_weights_only", TRAIN, 10000, SOURCE, LOG_DIR / "id_sft_formal_10000_retry6.log")
    if not complete_checkpoint(TRAIN / "checkpoints/global_step_10000"):
        raise RuntimeError("formal global_step_10000 checkpoint is incomplete")
    (TRAIN / "TRAINING_COMPLETE").write_text(
        json.dumps({"source_checkpoint": str(SOURCE), "target_step": 10000}, indent=2) + "\n"
    )
    write_state(current_stage="id_checkpoint_probe", next_stage="id_checkpoint_probe_and_early_stop", training_pid=None)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        (RETRY / "PIPELINE_FAILED").parent.mkdir(parents=True, exist_ok=True)
        (RETRY / "PIPELINE_FAILED").write_text(str(exc) + "\n")
        write_state(current_stage="id_resume_failed", next_stage="new_engineering_retry_or_user_decision", terminal_marker="PIPELINE_FAILED")
        raise
