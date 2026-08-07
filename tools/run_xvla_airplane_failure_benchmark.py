#!/usr/bin/env python3
"""Restart-tolerant driver for the X-VLA airplane failure benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--internal-pid", type=Path, required=True)
    parser.add_argument("--external-pid", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def pid_alive(path: Path) -> bool:
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return False


def wait_for(path: Path, pid_path: Path, interval: int) -> None:
    while not path.exists():
        if not pid_alive(pid_path):
            raise RuntimeError(f"producer exited before writing {path}")
        print(f"[pipeline] waiting for {path}", flush=True)
        time.sleep(interval)


def env_for(gpu: int, repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{repo}:{repo / 'RLinf'}"
    return env


def completed(output: Path) -> bool:
    if (output / "summary.json").exists():
        return True
    if output.exists():
        raise RuntimeError(f"partial output requires diagnosis and a new retry path: {output}")
    return False


def evaluation_command(
    args: argparse.Namespace,
    *,
    split: str,
    episodes: int,
    seed: int,
    output: Path,
    internal: Path,
    external: Path,
) -> list[str]:
    return [
        str(args.python),
        str(args.repo / "tools/evaluate_xvla_airplane_failure_detectors.py"),
        "--checkpoint", str(args.checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--multilayer-assets", str(internal),
        "--external-assets", str(external),
        "--output-dir", str(output),
        "--split", split,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--execute-horizon", "5",
        "--max-episode-steps", "150",
        "--flow-steps", "10",
        "--probe-steps", "5",
        "--probe-seed", "0",
        "--diff-timesteps", "16",
        "--diff-noise-samples", "1",
    ]


def launch(
    command: list[str], *, log: Path, gpu: int, cpus: str, repo: Path
) -> tuple[subprocess.Popen, object]:
    handle = log.open("w")
    process = subprocess.Popen(
        ["taskset", "-c", cpus, *command],
        cwd=repo,
        env=env_for(gpu, repo),
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def main() -> None:
    args = parse_args()
    logs = args.result_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    internal = args.result_root / "assets_internal/multilayer_detector_assets.pt"
    external = args.result_root / "assets_external_retry1/external_detector_assets.pt"
    wait_for(internal, args.internal_pid, args.poll_seconds)
    wait_for(external, args.external_pid, args.poll_seconds)
    print("[pipeline] detector assets ready", flush=True)

    calibration_rollouts = args.result_root / "calibration_id25"
    if not completed(calibration_rollouts):
        command = evaluation_command(
            args,
            split="id",
            episodes=25,
            seed=51000,
            output=calibration_rollouts,
            internal=internal,
            external=external,
        )
        process, handle = launch(
            command, log=logs / "calibration_id25.log", gpu=2, cpus="40-59", repo=args.repo
        )
        status = process.wait()
        handle.close()
        if status and not completed(calibration_rollouts):
            raise RuntimeError(f"calibration rollout exited with status {status}")
        if status:
            print(
                f"[pipeline] calibration exited with status {status} after writing a complete summary; continuing",
                flush=True,
            )
    calibration_summary = json.loads((calibration_rollouts / "summary.json").read_text())
    if int(calibration_summary["ever_grasped_successes"]) < 20:
        raise RuntimeError("fewer than 20 independent successful ID calibration trajectories")

    calibration = args.result_root / "calibration_q95.json"
    if not calibration.exists():
        subprocess.run(
            [
                str(args.python),
                str(args.repo / "tools/summarize_xvla_airplane_failure_detection.py"),
                "--calibrate", str(calibration_rollouts / "summary.json"),
                "--q", "0.95",
                "--output", str(calibration),
            ],
            cwd=args.repo,
            env=env_for(2, args.repo),
            check=True,
            stdout=(logs / "calibration_scores.log").open("w"),
            stderr=subprocess.STDOUT,
        )

    jobs = []
    for split, gpu, cpus, seed in (("id", 0, "0-19", 50000), ("ood", 1, "20-39", 60000)):
        output = args.result_root / f"eval_{split}_100"
        if completed(output):
            continue
        command = evaluation_command(
            args,
            split=split,
            episodes=100,
            seed=seed,
            output=output,
            internal=internal,
            external=external,
        )
        process, handle = launch(
            command,
            log=logs / f"eval_{split}_100.log",
            gpu=gpu,
            cpus=cpus,
            repo=args.repo,
        )
        jobs.append((split, process, handle))
    for split, process, handle in jobs:
        status = process.wait()
        handle.close()
        output = args.result_root / f"eval_{split}_100"
        if status and not completed(output):
            raise RuntimeError(f"{split} evaluation exited with status {status}")
        if status:
            print(
                f"[pipeline] {split} evaluation exited with status {status} after writing a complete summary; continuing",
                flush=True,
            )

    metrics = args.result_root / "metrics"
    if not (metrics / "metrics.json").exists():
        if metrics.exists():
            raise RuntimeError(f"partial metrics output exists: {metrics}")
        subprocess.run(
            [
                str(args.python),
                str(args.repo / "tools/summarize_xvla_airplane_failure_detection.py"),
                "--id-summary", str(args.result_root / "eval_id_100/summary.json"),
                "--ood-summary", str(args.result_root / "eval_ood_100/summary.json"),
                "--calibration", str(calibration),
                "--output", str(metrics),
            ],
            cwd=args.repo,
            env=env_for(2, args.repo),
            check=True,
            stdout=(logs / "metrics.log").open("w"),
            stderr=subprocess.STDOUT,
        )
    print(f"[pipeline] complete: {metrics / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
