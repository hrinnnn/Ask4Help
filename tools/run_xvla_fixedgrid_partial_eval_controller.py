#!/usr/bin/env python3
"""Run a small, isolated evaluator diagnostic before formal Stage-B utility eval.

The diagnostic is deliberately downstream of Stage-B training completion so it
cannot compete with the approved training process for GPU5.  It uses separate
seeds and output directories, and its summaries are never consumed as formal
100-episode utility results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EPISODES = 20
JOBS = (
    {
        "task": "stackcube",
        "split": "id",
        "seed": 190000,
    },
    {
        "task": "stackcube",
        "split": "ood",
        "seed": 190100,
    },
    {
        "task": "airplane",
        "split": "id",
        "seed": 191000,
    },
    {
        "task": "airplane",
        "split": "ood",
        "seed": 191100,
    },
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def gpu_idle(gpu: int) -> bool:
    """Require both low memory and no compute app on the selected GPU."""

    try:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        values = {
            int(index.strip()): int(used.strip())
            for index, used in (line.split(",") for line in rows.splitlines())
        }
        if values.get(gpu, 2048) > 1024:
            return False
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        uuid = subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                str(gpu),
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        return not any(line.startswith(uuid) for line in apps.splitlines() if line.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return False


def summary_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        summary = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    if int(summary.get("episodes", -1)) != EXPECTED_EPISODES:
        return False
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != EXPECTED_EPISODES:
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        for key in ("video", "actions"):
            artifact = row.get(key)
            if not artifact or not Path(str(artifact)).is_file():
                return False
    return True


def wait_for_gpu(args: argparse.Namespace, state: dict[str, Any], state_path: Path, label: str) -> None:
    while not gpu_idle(args.gpu):
        state.update(
            {
                "stage": "partial_eval_waiting_for_gpu",
                "waiting_for": label,
                "updated_at": now(),
            }
        )
        write_json(state_path, state)
        time.sleep(args.interval_seconds)


def job_command(job: dict[str, Any], checkpoint: Path, output: Path, args: argparse.Namespace) -> list[str]:
    if job["task"] == "stackcube":
        evaluator = args.repo / "tools/evaluate_stackcube_xvla.py"
        return [
            args.python,
            str(evaluator),
            "--checkpoint",
            str(checkpoint),
            "--xvla-root",
            str(args.xvla_root),
            "--output-dir",
            str(output),
            "--episodes",
            str(EXPECTED_EPISODES),
            "--seed",
            str(job["seed"]),
            "--split",
            job["split"],
            "--max-episode-steps",
            "150",
            "--flow-steps",
            "10",
        ]
    evaluator = args.repo / "tools/evaluate_pick_single_ycb_airplane_xvla.py"
    return [
        args.python,
        str(evaluator),
        "--checkpoint",
        str(checkpoint),
        "--xvla-root",
        str(args.xvla_root),
        "--output-dir",
        str(output),
        "--episodes",
        str(EXPECTED_EPISODES),
        "--seed",
        str(job["seed"]),
        "--split",
        job["split"],
        "--max-episode-steps",
        "150",
        "--flow-steps",
        "10",
    ]


def checkpoint_for(task: str, args: argparse.Namespace) -> Path:
    if task == "stackcube":
        return args.training_root / "stackcube/step_20/seed_17001/train/ckpt-2500"
    return args.training_root / "airplane/step_20/seed_17001/train/ckpt-2500"


def run_job(
    *,
    job: dict[str, Any],
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint_for(job["task"], args)
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint / "model.safetensors")
    output = args.run_root / job["task"] / f"step_20_seed_17001" / job["split"]
    summary = output / "summary.json"
    key = f"{job['task']}/step_20_seed_17001/{job['split']}"
    if summary_complete(summary):
        state.setdefault("completed_jobs", []).append(key)
        write_json(state_path, state)
        return read_json(summary)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"partial diagnostic output exists: {output}")
    wait_for_gpu(args, state, state_path, key)
    command = job_command(job, checkpoint, output, args)
    log = args.run_root / "logs" / f"{job['task']}_{job['split']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    state.update(
        {
            "stage": "partial_eval_running",
            "current_job": key,
            "command": command,
            "updated_at": now(),
        }
    )
    write_json(state_path, state)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "20",
        "MKL_NUM_THREADS": "20",
    }
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            ["taskset", "-c", args.cpu_set, *command],
            cwd=args.repo,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode not in (0, -6) or not summary_complete(summary):
        state.update(
            {
                "stage": "partial_eval_failed",
                "failed_job": key,
                "returncode": result.returncode,
                "updated_at": now(),
            }
        )
        write_json(state_path, state)
        raise RuntimeError(f"partial evaluation failed: {key}; see {log}")
    state.setdefault("completed_jobs", []).append(key)
    state.update({"stage": "partial_eval_running", "updated_at": now()})
    write_json(state_path, state)
    return read_json(summary)


def run(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    state_path = args.run_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": "xvla_fixedgrid_taskpolicy_knee_v1",
        "diagnostic": True,
        "formal": False,
        "stage": "partial_eval_waiting_for_stage_b_training",
        "started_at": now(),
        "completed_jobs": [],
    }
    state.setdefault("completed_jobs", [])
    write_json(state_path, state)
    training_marker = args.training_root / "STAGE_B_TRAINING_COMPLETE"
    if not args.start_now:
        while not training_marker.is_file():
            state.update(
                {
                    "stage": "partial_eval_waiting_for_stage_b_training",
                    "updated_at": now(),
                }
            )
            write_json(state_path, state)
            time.sleep(args.interval_seconds)
    else:
        state.update(
            {
                "stage": "partial_eval_waiting_for_gpu",
                "start_condition": "required knee checkpoints present and selected GPU idle",
                "updated_at": now(),
            }
        )
        write_json(state_path, state)

    summaries: dict[str, Any] = {}
    for job in JOBS:
        key = f"{job['task']}/step_20_seed_17001/{job['split']}"
        if key in state["completed_jobs"]:
            summary = args.run_root / job["task"] / "step_20_seed_17001" / job["split"] / "summary.json"
            summaries[key] = read_json(summary)
            continue
        summaries[key] = run_job(job=job, args=args, state=state, state_path=state_path)

    report = {
        "format": "xvla_fixedgrid_partial_evaluation_diagnostic_v1",
        "formal": False,
        "episodes_per_task_split": EXPECTED_EPISODES,
        "not_used_for_formal_denominators": True,
        "created_at": now(),
        "summaries": summaries,
    }
    write_json(args.run_root / "partial_report.json", report)
    state.update({"stage": "partial_eval_complete", "completed_at": now(), "updated_at": now()})
    write_json(state_path, state)
    (args.run_root / "PARTIAL_EVAL_COMPLETE").write_text(
        "diagnostic partial evaluation complete; excluded from formal utility denominators\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument(
        "--start-now",
        action="store_true",
        help="run as soon as the selected checkpoints and GPU are available, without waiting for Stage-B training completion",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
