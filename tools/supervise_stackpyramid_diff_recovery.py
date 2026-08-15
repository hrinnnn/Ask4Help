#!/usr/bin/env python3
"""Keep the StackPyramid Diff collection and stage transition restart-tolerant."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_state(path: Path, **values: object) -> None:
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(values)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def controller_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python), "-u", str(args.controller),
        "--output-root", str(args.root),
        "--repo-root", str(args.repo_root),
        "--xvla-root", str(args.xvla_root),
        "--python", str(args.python),
        "--base-model", str(args.base_model),
        "--id-h5", str(args.id_h5),
        "--pca-asset", str(args.pca_asset),
        "--pca-threshold", str(args.pca_threshold),
        "--gpus", args.gpus,
        "--cpu-sets", args.cpu_sets,
        "--training-steps", str(args.training_steps),
        "--batch-size", str(args.batch_size),
    ]


def diff_command(args: argparse.Namespace, output: Path, attempts: int) -> list[str]:
    return [
        str(args.python), "-u", str(args.collector),
        "--method", "diffdagger",
        "--checkpoint", str(args.base_model),
        "--xvla-root", str(args.xvla_root),
        "--output-dir", str(output),
        "--split", "stage1_ood",
        "--target", "100",
        "--id-seed", "70000",
        "--ood-seed", "80000",
        "--max-attempts", str(attempts),
        "--flow-steps", "5",
        "--diff-timesteps", "16",
        "--diff-patience", "2",
        "--diff-threshold", str(args.diff_threshold),
        "--sim-backend", "cpu",
        "--render-backend", "cpu",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--current-output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--id-h5", type=Path, required=True)
    parser.add_argument("--pca-asset", type=Path, required=True)
    parser.add_argument("--pca-threshold", type=float, required=True)
    parser.add_argument("--diff-threshold", type=float, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--cpu-sets", default="80-99,100-119")
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-retry", type=int, default=3)
    parser.add_argument("--attempts-per-retry", type=int, default=1500)
    args = parser.parse_args()

    logs = args.root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = args.root / "diff_recovery_state.json"
    pid = args.wait_pid
    output = args.current_output
    retry = args.start_retry
    while True:
        write_state(state_path, phase="waiting_or_collecting", pid=pid, output=str(output), retry=retry)
        while alive(pid):
            time.sleep(60)
        if (output / "COLLECTION_COMPLETE").is_file():
            write_state(state_path, phase="handoff", output=str(output))
            os.execv(str(args.python), controller_command(args))

        while True:
            candidate = args.root / "collections" / "stage1_ood" / f"diffdagger_retry{retry}"
            retry += 1
            if not candidate.exists():
                break
        log = logs / f"collect_stage1_ood_diffdagger_retry{retry - 1}.log"
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "5",
            "PYTHONPATH": os.pathsep.join([str(args.repo_root), str(args.xvla_root)]),
            "TOKENIZERS_PARALLELISM": "false",
        })
        command = diff_command(args, candidate, args.attempts_per_retry)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command, "retry": retry - 1}) + "\n")
            stream.flush()
            process = subprocess.Popen(
                ["taskset", "-c", "100-119", *command],
                cwd=args.repo_root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            pid = process.pid
        output = candidate
        write_state(state_path, phase="collecting", pid=pid, output=str(output), retry=retry - 1, log=str(log))


if __name__ == "__main__":
    main()
