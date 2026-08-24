#!/usr/bin/env python3
"""Run an isolated four-GPU object-variation retry without touching single-GPU training."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1")
SOURCE = RUN / "id_training_v1/formal_10000_retry8/id_sft_10000_retry8/id_sft_10000_retry7_weights_only/checkpoints/global_step_4000"
RETRY = RUN / "id_training_v1/formal_10000_4gpu_retry3"
SMOKE = RETRY / "smoke_2step"
FORMAL = RETRY / "id_sft_10000_4gpu_retry1"
DATASET = RUN / "datasets/id_v1_retry1"
NORM = DATASET / "norm_stats.json"
MODEL = Path("/data/zhaozhixuan/Ask4Help-open-drawer/results/model_cache/pi05_base_pytorch_v1")
PYTHON = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python")
CONFIG = ROOT / "RLinf/examples/sft/config"
CONFIG_NAME = "pick_single_ycb_object_variation_id_sft_openpi_pi05"
GPU_IDS = os.environ.get("OBJECT_VARIATION_4GPU_IDS", "4,5,6,7")
PLACEMENT = os.environ.get("OBJECT_VARIATION_4GPU_PLACEMENT", "4-7")
CPUSET = os.environ.get("OBJECT_VARIATION_4GPU_CPUSET", "80-159")
STATE = RETRY / "pipeline_state.json"


def write_state(stage: str, **extra: object) -> None:
    payload = json.loads(STATE.read_text()) if STATE.is_file() else {}
    payload.update({"pipeline": "pick_single_ycb_object_variation_4gpu_retry3", "stage": stage, **extra})
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    STATE.write_text(json.dumps(payload, indent=2) + "\n")


def complete_checkpoint(path: Path) -> bool:
    weights = path / "actor/model_state_dict/full_weights.pt"
    dcp = path / "actor/dcp_checkpoint"
    return weights.is_file() and weights.stat().st_size > 1024 * 1024 and dcp.is_dir()


def train(output: Path, experiment: str, max_steps: int, save_interval: int, log: Path) -> Path:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": GPU_IDS,
            "ASK4HELP_RLINF_PLACEMENT": PLACEMENT,
            "EMBODIED_PATH": str(ROOT / "RLinf/examples/sft"),
            "PYTHONPATH": f"{ROOT}:{ROOT / 'RLinf'}:{env.get('PYTHONPATH', '')}",
            "OBJECT_VARIATION_ID_DATASET": str(DATASET),
            "OBJECT_VARIATION_ID_NORM_STATS": str(NORM),
            "OBJECT_VARIATION_PI05_MODEL_PATH": str(MODEL),
            "OBJECT_VARIATION_RUN_ROOT": str(output),
            "OBJECT_VARIATION_EXPERIMENT_NAME": experiment,
            "OBJECT_VARIATION_MAX_STEPS": str(max_steps),
            "OBJECT_VARIATION_SAVE_INTERVAL": str(save_interval),
            "OBJECT_VARIATION_TRAIN_SEED": "7000",
            "RAY_TMPDIR": "/sdd/ov_ray_4gpu_retry3",
            "TMPDIR": "/sdd/ov_tmp_4gpu_retry3",
            "RLINF_RAY_ADDRESS": "local",
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
        f"runner.save_interval={save_interval}",
        "actor.optim.total_training_steps=10000",
        "+runner.initial_step=4000",
        f"actor.model.model_path={SOURCE}",
    ]
    output.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        (output / "train.pid").write_text(str(process.pid) + "\n")
        write_state("smoke_running" if max_steps == 4002 else "formal_training_running", train_pid=process.pid, experiment=experiment)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"{experiment} exited with return code {returncode}")
    checkpoint = output / experiment / "checkpoints" / f"global_step_{max_steps}"
    if not complete_checkpoint(checkpoint):
        raise RuntimeError(f"incomplete checkpoint: {checkpoint}")
    return checkpoint


def reload_forward(checkpoint: Path) -> None:
    output = RETRY / "reload_forward_smoke"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "4", "PYTHONPATH": f"{ROOT}:{ROOT / 'RLinf'}:{env.get('PYTHONPATH', '')}", "PYTHONUNBUFFERED": "1"})
    command = [
        "taskset", "-c", "80-99", str(PYTHON), str(ROOT / "tools/evaluate_pick_single_ycb_object_variation_pi05.py"),
        "--checkpoint", str(checkpoint), "--pi05-base", str(MODEL), "--norm-stats", str(NORM),
        "--output-dir", str(output), "--split", "id", "--episodes", "1", "--seed", "24002",
        "--execute-horizon", "5", "--max-episode-steps", "200",
    ]
    log = RETRY / "reload_forward_smoke.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    summary = output / "summary.json"
    videos = list((output / "videos").glob("*.mp4")) if (output / "videos").is_dir() else []
    if not summary.is_file() or len(videos) != 1:
        raise RuntimeError("four-GPU reload/forward smoke evidence is incomplete")
    if result.returncode != 0:
        (output / "SIMULATOR_EXIT_AFTER_ARTIFACTS").write_text(json.dumps({"returncode": result.returncode, "episodes": 1, "videos": 1}, indent=2) + "\n")
    (RETRY / "SMOKE_RELOAD_PASSED").write_text(json.dumps({"checkpoint": str(checkpoint), "episodes": 1, "videos": 1, "gpu_ids": GPU_IDS}, indent=2) + "\n")


def main() -> int:
    if not complete_checkpoint(SOURCE):
        raise RuntimeError(f"source checkpoint incomplete: {SOURCE}")
    if not DATASET.is_dir() or not NORM.is_file() or not MODEL.is_dir() or not PYTHON.is_file():
        raise RuntimeError("immutable data, norm, model, or runtime missing")
    if RETRY.exists() and any(RETRY.iterdir()):
        raise RuntimeError(f"refusing to overwrite retry root: {RETRY}")
    RETRY.mkdir(parents=True)
    (RETRY / "provenance.json").write_text(json.dumps({"source_checkpoint": str(SOURCE), "gpu_ids": GPU_IDS, "placement": PLACEMENT, "cpuset": CPUSET, "global_batch_size": 128, "micro_batch_size": 32, "initial_step": 4000, "optimizer": "reset", "single_gpu_training_untouched": True}, indent=2) + "\n")
    write_state("preflight", source_checkpoint=str(SOURCE), gpu_ids=GPU_IDS, single_gpu_training_untouched=True)
    smoke_name = "id_resume_smoke_4gpu_from4000"
    smoke_checkpoint = train(SMOKE, smoke_name, 4002, 2, RETRY / "smoke_2step.log")
    write_state("reload_forward_smoke", smoke_checkpoint=str(smoke_checkpoint))
    reload_forward(smoke_checkpoint)
    write_state("formal_training_starting", next_stage="formal_10000")
    formal_name = "id_sft_10000_4gpu_retry1_weights_only"
    final_checkpoint = train(FORMAL, formal_name, 10000, 500, RETRY / "formal_training.log")
    (FORMAL / "TRAINING_COMPLETE").write_text(json.dumps({"checkpoint": str(final_checkpoint), "source_checkpoint": str(SOURCE), "gpu_ids": GPU_IDS}, indent=2) + "\n")
    write_state("formal_training_complete", checkpoint=str(final_checkpoint))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        RETRY.mkdir(parents=True, exist_ok=True)
        (RETRY / "PIPELINE_FAILED").write_text(str(exc) + "\n")
        write_state("failed", error=str(exc))
        raise
