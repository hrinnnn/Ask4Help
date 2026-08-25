#!/usr/bin/env python3
"""Durable Stage-B supervisor: training -> evaluation -> utility summaries.

The training and evaluation controllers own their respective jobs.  This
small supervisor only advances the approved stage graph after completion
markers and denominator checks pass, so a paused Codex session is not needed
for stage transitions.
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
MANIFEST_DEFAULT = ROOT / "configs/pipelines/xvla_fixedgrid_taskpolicy_knee_v1.json"
EXPECTED_TRAINING_JOBS = 27
EXPECTED_EVAL_JOBS = 54


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pid_alive(pid: int) -> bool:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            if proc_stat.read_text(encoding="utf-8").split()[2] == "Z":
                return False
        except (OSError, IndexError):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def training_complete(training_root: Path) -> bool:
    marker = training_root / "STAGE_B_TRAINING_COMPLETE"
    state_path = training_root / "pipeline_state.json"
    if not marker.is_file() or not state_path.is_file():
        return False
    state = read_json(state_path)
    return state.get("stage") == "stage_b_training_complete" and len(state.get("completed_jobs", [])) == EXPECTED_TRAINING_JOBS


def evaluation_complete(evaluation_root: Path) -> bool:
    marker = evaluation_root / "STAGE_B_EVAL_COMPLETE"
    state_path = evaluation_root / "pipeline_state.json"
    if not marker.is_file() or not state_path.is_file():
        return False
    state = read_json(state_path)
    return state.get("stage") == "stage_b_evaluation_complete" and len(state.get("completed_evals", [])) == EXPECTED_EVAL_JOBS


def launch_evaluation(args: argparse.Namespace, evaluation_root: Path, log_path: Path) -> int:
    command = [
        args.python,
        str(args.repo / "tools/run_xvla_fixedgrid_stage_b_eval_controller.py"),
        "--task", "both",
        "--training-root", str(args.training_root),
        "--run-root", str(evaluation_root),
        "--repo", str(args.repo),
        "--xvla-root", str(args.xvla_root),
        "--python", args.python,
        "--gpu", str(args.gpu),
        "--cpu-set", args.cpu_set,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            start_new_session=True,
        )
    return process.pid


def run_utility_summaries(args: argparse.Namespace, evaluation_root: Path, supervisor_root: Path) -> None:
    summary_root = evaluation_root / "utility_summaries"
    commands = [
        [
            args.python,
            str(args.repo / "tools/summarize_xvla_stage_b_utility.py"),
            "--root", str(evaluation_root),
            "--task", "stackcube",
            "--anchors", "0", "10", "20", "30", "45",
            "--knee-set", "10", "20",
            "--output", str(summary_root / "stackcube.json"),
        ],
        [
            args.python,
            str(args.repo / "tools/summarize_xvla_stage_b_utility.py"),
            "--root", str(evaluation_root),
            "--task", "airplane",
            "--anchors", "0", "10", "20", "30",
            "--knee-set", "20",
            "--output", str(summary_root / "airplane.json"),
        ],
    ]
    for command in commands:
        subprocess.run(command, cwd=args.repo, check=True)
    write_json(
        summary_root / "combined.json",
        {
            "format": "xvla_stage_b_timing_utility_combined_v1",
            "tasks": ["stackcube", "airplane"],
            "summaries": {name: str(summary_root / f"{name}.json") for name in ("stackcube", "airplane")},
            "created_at": now(),
        },
    )


def run(args: argparse.Namespace) -> None:
    supervisor_root = args.supervisor_root
    evaluation_root = args.evaluation_root
    supervisor_root.mkdir(parents=True, exist_ok=True)
    evaluation_root.mkdir(parents=True, exist_ok=True)
    state_path = supervisor_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": "xvla_fixedgrid_taskpolicy_knee_v1",
        "stage": "stage_b_waiting_for_training",
        "started_at": now(),
        "training_root": str(args.training_root),
        "evaluation_root": str(evaluation_root),
    }
    log_path = supervisor_root / "supervisor.log"

    while True:
        if not training_complete(args.training_root):
            training_state_path = args.training_root / "pipeline_state.json"
            training_state = read_json(training_state_path) if training_state_path.exists() else {}
            if training_state.get("stage") == "stage_b_failed":
                state.update({"stage": "stage_b_training_failed", "failed_job": training_state.get("failed_job"), "updated_at": now()})
                write_json(state_path, state)
                raise RuntimeError(f"training controller failed: {training_state.get('failed_job')}")
            state.update({"stage": "stage_b_waiting_for_training", "updated_at": now()})
            write_json(state_path, state)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} waiting for training completion\n")
            time.sleep(args.interval_seconds)
            continue

        if not evaluation_complete(evaluation_root):
            eval_state_path = evaluation_root / "pipeline_state.json"
            eval_state = read_json(eval_state_path) if eval_state_path.exists() else {}
            if eval_state.get("stage") == "stage_b_evaluation_failed":
                state.update({"stage": "stage_b_evaluation_failed", "failed_eval": eval_state.get("failed_eval"), "updated_at": now()})
                write_json(state_path, state)
                raise RuntimeError(f"evaluation controller failed: {eval_state.get('failed_eval')}")
            pid_path = supervisor_root / "evaluation.pid"
            pid = int(pid_path.read_text(encoding="utf-8")) if pid_path.exists() else 0
            if not pid or not pid_alive(pid):
                pid = launch_evaluation(args, evaluation_root, supervisor_root / "evaluation.log")
                pid_path.write_text(f"{pid}\n", encoding="utf-8")
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"{now()} launched evaluation pid={pid}\n")
            state.update({"stage": "stage_b_evaluation", "evaluation_pid": pid, "updated_at": now()})
            write_json(state_path, state)
            time.sleep(args.interval_seconds)
            continue

        utility_marker = supervisor_root / "STAGE_B_UTILITY_COMPLETE"
        if not utility_marker.is_file():
            run_utility_summaries(args, evaluation_root, supervisor_root)
            utility_marker.write_text("complete\n", encoding="utf-8")
            state.update({"stage": "stage_b_utility_complete", "completed_at": now(), "updated_at": now()})
            write_json(state_path, state)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} utility summaries complete\n")
        break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--supervisor-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--interval-seconds", type=int, default=900)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
