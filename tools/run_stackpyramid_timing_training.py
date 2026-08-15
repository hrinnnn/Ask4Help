#!/usr/bin/env python3
"""Durable two-GPU controller for the StackPyramid timing-sweep updates."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path


STAGES = ("stage1_ood", "stage2_ood", "stage3_ood")
CONDITIONS = ("immediate", "pre_stage", "capability_boundary", "failure_recovery")


def write_state(path: Path, **updates: object) -> None:
    state = {"format": "stackpyramid_timing_training_v1", **updates}
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def fresh_output(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 100):
        candidate = path.with_name(f"{path.name}_retry{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many partial outputs for {path}")


def run_one(
    label: str,
    command: list[str],
    gpu: str,
    cpu_set: str,
    repo_root: Path,
    log_path: Path,
    state_path: Path,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root), env.get("PYTHONPATH", "")])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"label": label, "command": command, "gpu": gpu, "cpu_set": cpu_set}) + "\n")
        stream.flush()
        process = subprocess.Popen(
            ["taskset", "-c", cpu_set, *command],
            cwd=repo_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        write_state(state_path, phase="training", running={"label": label, "pid": process.pid, "gpu": gpu, "log": str(log_path)})
        return_code = process.wait()
        stream.write(json.dumps({"return_code": return_code}) + "\n")
    if return_code not in (0, -6):
        raise RuntimeError(f"{label} exited with {return_code}; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--id-h5", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--cpu-sets", default="0-19,20-39")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8200)
    args = parser.parse_args()

    if len(args.gpus.split(",")) != 2 or len(args.cpu_sets.split(",")) != 2:
        raise ValueError("exactly two GPUs and two CPU sets are required")
    gpus = args.gpus.split(",")
    cpu_sets = args.cpu_sets.split(",")
    root = args.output_root
    logs = root / "training_logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_path = root / "training_pipeline_state.json"
    write_state(state_path, phase="starting", output_root=root, steps=args.steps, batch_size=args.batch_size)

    jobs: list[tuple[str, Path, str, str, int]] = []
    for stage_index, stage in enumerate(STAGES):
        for condition_index, condition in enumerate(CONDITIONS):
            output = fresh_output(root / "training" / stage / condition)
            output.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((stage, output, gpus[len(jobs) % 2], cpu_sets[len(jobs) % 2], args.seed + stage_index * 100 + condition_index))

    train_script = args.repo_root / "tools" / "run_stackpyramid_gated_training.py"
    for wave_start in range(0, len(jobs), 2):
        wave = jobs[wave_start : wave_start + 2]
        write_state(
            state_path,
            phase="training",
            wave=wave_start // 2 + 1,
            total_waves=(len(jobs) + 1) // 2,
            active=[{"stage": stage, "output": str(output), "gpu": gpu} for stage, output, gpu, _cpu, _seed in wave],
        )

        commands: list[tuple[str, list[str], str, str, Path]] = []
        for stage, output, gpu, cpu_set, seed in wave:
            condition = CONDITIONS[(wave_start + len(commands)) % len(CONDITIONS)]
            # The jobs list is ordered stage-major, so recover the condition from its output path.
            condition = output.name
            command = [
                str(args.python),
                str(train_script),
                "--xvla-root", str(args.xvla_root),
                "--model", str(args.base_model),
                "--id-h5", str(args.id_h5),
                "--expert-h5", str(root / "selected" / stage / condition / "accepted_suffixes.h5"),
                "--output", str(output),
                "--steps", str(args.steps),
                "--save-interval", "500",
                "--batch-size", str(args.batch_size),
                "--seed", str(seed),
            ]
            commands.append((f"{stage}_{condition}", command, gpu, cpu_set, output))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_one, label, command, gpu, cpu_set, args.repo_root, logs / f"{label}.log", state_path)
                for label, command, gpu, cpu_set, _output in commands
            ]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
        if errors:
            write_state(state_path, phase="failed", error="; ".join(str(error) for error in errors))
            raise RuntimeError("; ".join(str(error) for error in errors))
        for label, _command, _gpu, _cpu_set, output in commands:
            expected = [output / f"ckpt-{step}" for step in (500, 1000, 1500, 2000)]
            if not (output / "TRAINING_COMPLETE").is_file() or not all(path.is_dir() for path in expected):
                write_state(state_path, phase="failed", error=f"incomplete training output: {output}")
                raise RuntimeError(f"incomplete training output: {output}")

    (root / "TIMING_TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    write_state(state_path, phase="complete", marker=str(root / "TIMING_TRAINING_COMPLETE"))


if __name__ == "__main__":
    main()
