#!/usr/bin/env python3
"""Restart-tolerant matched-budget Stage-B timing training controller."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DEFAULT = ROOT / "configs/pipelines/xvla_fixedgrid_taskpolicy_knee_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_gpu(gpu: int, max_memory_mib: int = 1024) -> None:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = {int(i.strip()): int(m.strip()) for i, m in (line.split(",") for line in rows.splitlines())}
    if values.get(gpu, max_memory_mib + 1) > max_memory_mib:
        raise RuntimeError(f"GPU {gpu} is not idle: {values.get(gpu)} MiB")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    uuid = subprocess.check_output(
        ["nvidia-smi", "--id", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if any(line.startswith(uuid) for line in apps.splitlines() if line.strip()):
        raise RuntimeError(f"GPU {gpu} has a compute process")


def jobs_for_task(task: str) -> tuple[list[int], Path, int]:
    base = Path("/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1")
    if task == "stackcube":
        return [0, 10, 20, 30, 45], base / "formal_calibration_merged_v2/timing_datasets_budget_520", 520
    return [0, 10, 20, 30], base / "airplane_calibration_merged_v2/timing_datasets_budget_2820", 2820


def run(args: argparse.Namespace) -> None:
    manifest = read_json(args.manifest)
    task_order = ["stackcube", "airplane"] if args.task == "both" else [args.task]
    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": manifest["pipeline_id"],
        "stage": "stage_b_training",
        "started_at": now(),
        "tasks": task_order,
        "training_seeds": [17001, 17002, 17003],
        "completed_jobs": [],
    }
    write_json(state_path, state)
    for task in task_order:
        anchors, dataset_root, budget = jobs_for_task(task)
        for anchor in anchors:
            for seed in (17001, 17002, 17003):
                job_id = f"{task}/step_{anchor}/seed_{seed}"
                if job_id in state["completed_jobs"]:
                    continue
                output = run_root / task / f"step_{anchor}" / f"seed_{seed}"
                if output.exists():
                    raise FileExistsError(f"partial training output exists: {output}")
                check_gpu(args.gpu)
                command = [
                    args.python,
                    str(args.repo / "tools/run_xvla_fixedgrid_timing_training.py"),
                    "--task", task,
                    "--anchor", str(anchor),
                    "--seed", str(seed),
                    "--dataset-root", str(dataset_root / f"step_{anchor}"),
                    "--expected-budget", str(budget),
                    "--output", str(output),
                    "--repo", str(args.repo),
                    "--xvla-root", str(args.xvla_root),
                    "--python", args.python,
                    "--gpu", str(args.gpu),
                    "--cpu-set", args.cpu_set,
                    "--steps", str(args.steps),
                    "--save-interval", str(args.save_interval),
                ]
                log = run_root / "logs" / f"{task}_step_{anchor}_seed_{seed}.log"
                state.update({"current_job": job_id, "command": command, "updated_at": now()})
                write_json(state_path, state)
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as handle:
                    result = subprocess.run(
                        command,
                        cwd=args.repo,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if result.returncode != 0 or not (output / "TRAINING_COMPLETE").is_file():
                    state.update({"stage": "stage_b_failed", "failed_job": job_id, "returncode": result.returncode, "updated_at": now()})
                    write_json(state_path, state)
                    raise RuntimeError(f"Stage-B job failed: {job_id}; see {log}")
                state["completed_jobs"].append(job_id)
                state.update({"stage": "stage_b_training", "updated_at": now()})
                write_json(state_path, state)
    state.update({"stage": "stage_b_training_complete", "completed_at": now()})
    write_json(state_path, state)
    (run_root / "STAGE_B_TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--task", choices=("stackcube", "airplane", "both"), default="both")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--save-interval", type=int, default=500)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
