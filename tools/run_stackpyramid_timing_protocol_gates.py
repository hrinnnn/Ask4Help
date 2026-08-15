#!/usr/bin/env python3
"""Run the protocol gates required before a StackPyramid timing sweep."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SPLITS = ("id", "stage1_ood", "stage2_ood", "stage3_ood")


def _run_pair(commands: list[tuple[list[str], Path, str]]) -> None:
    processes: list[tuple[subprocess.Popen[bytes], Path, str]] = []
    for command, log_path, label in commands:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("wb")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, log_path, label))
    failures: list[str] = []
    for process, log_path, label in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append(f"{label}: rc={return_code}, log={log_path}")
    if failures:
        raise RuntimeError("; ".join(failures))


def _command(
    python: Path,
    script: Path,
    xvla_root: Path,
    gpu: str,
    cpu_set: str,
    args: list[str],
) -> list[str]:
    repo_root = script.parents[1]
    env_prefix = [
        "env",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        "PYTHONPATH=" + os.pathsep.join((str(repo_root), str(xvla_root))),
    ]
    return ["taskset", "-c", cpu_set, *env_prefix, str(python), str(script), *args]


def _summary(path: Path) -> dict[str, object]:
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("episodes", -1)) != 100:
        raise RuntimeError(f"expected 100 episodes in {summary_path}: {summary}")
    return summary


def _run_oracle_gates(args: argparse.Namespace, root: Path) -> dict[str, dict[str, object]]:
    script = args.repo_root / "tools" / "collect_stackpyramid_oracle.py"
    output = root / "oracle"
    output.mkdir(parents=True, exist_ok=True)
    starts = {"id": 91000, "stage1_ood": 92000, "stage2_ood": 93000, "stage3_ood": 94000}
    results: dict[str, dict[str, object]] = {}
    for pair in (("id", "stage1_ood"), ("stage2_ood", "stage3_ood")):
        commands: list[tuple[list[str], Path, str]] = []
        for gpu_index, split in enumerate(pair):
            split_output = output / split
            if split_output.exists():
                raise FileExistsError(split_output)
            command = _command(
                args.python,
                script,
                args.xvla_root,
                str(gpu_index),
                args.cpu_sets[gpu_index],
                [
                    "--output", str(split_output),
                    "--split", split,
                    "--episodes", "100",
                    "--start-seed", str(starts[split]),
                    "--sim-backend", "gpu",
                    "--render-backend", "gpu",
                ],
            )
            commands.append((command, root / "logs" / f"oracle_{split}.log", f"oracle_{split}"))
        _run_pair(commands)
        for split in pair:
            summary = _summary(output / split / "motionplanning")
            success_rate = float(summary["success_rate"])
            if success_rate < 0.90:
                raise RuntimeError(f"oracle success gate failed for {split}: {success_rate}")
            results[split] = {
                "episodes": int(summary["episodes"]),
                "strict_successes": int(summary["strict_successes"]),
                "success_rate": success_rate,
                "summary": str((output / split / f"oracle_summary_{split}.json").resolve()),
            }
    return results


def _run_base_policy_gates(args: argparse.Namespace, root: Path) -> dict[str, dict[str, object]]:
    script = args.repo_root / "tools" / "evaluate_stackpyramid_xvla.py"
    output = root / "base_policy"
    output.mkdir(parents=True, exist_ok=True)
    starts = {"id": 95000, "stage1_ood": 96000, "stage2_ood": 97000, "stage3_ood": 98000}
    results: dict[str, dict[str, object]] = {}
    for pair in (("id", "stage1_ood"), ("stage2_ood", "stage3_ood")):
        commands: list[tuple[list[str], Path, str]] = []
        for gpu_index, split in enumerate(pair):
            split_output = output / split
            if split_output.exists():
                raise FileExistsError(split_output)
            command = _command(
                args.python,
                script,
                args.xvla_root,
                str(gpu_index),
                args.cpu_sets[gpu_index],
                [
                    "--checkpoint", str(args.checkpoint),
                    "--xvla-root", str(args.xvla_root),
                    "--output", str(split_output),
                    "--split", split,
                    "--episodes", "100",
                    "--start-seed", str(starts[split]),
                    "--max-episode-steps", "250",
                    "--execute-horizon", "5",
                    "--flow-steps", "5",
                    "--device", "cuda",
                    "--sim-backend", "gpu",
                    "--render-backend", "gpu",
                ],
            )
            commands.append((command, root / "logs" / f"base_{split}.log", f"base_{split}"))
        _run_pair(commands)
        for split in pair:
            summary = _summary(output / split)
            results[split] = {
                "episodes": int(summary["episodes"]),
                "ever_grasped": int(summary["ever_grasped"]),
                "ever_base_completed": int(summary["ever_base_completed"]),
                "strict_success": int(summary["strict_success"]),
                "success_rate": float(summary["strict_success"]) / 100.0,
                "summary": str((output / split / "summary.json").resolve()),
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--cpu-sets", nargs=2, default=("0-7", "8-15"))
    args = parser.parse_args()
    root = args.output_root
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root / "pipeline_state.json").write_text(
        json.dumps({"phase": "oracle", "updated_at": time.time()}, indent=2) + "\n"
    )
    oracle = _run_oracle_gates(args, root)
    (root / "pipeline_state.json").write_text(
        json.dumps({"phase": "base_policy", "oracle": oracle, "updated_at": time.time()}, indent=2) + "\n"
    )
    base_policy = _run_base_policy_gates(args, root)
    id_success = float(base_policy["id"]["success_rate"])
    ood_success = {split: float(base_policy[split]["success_rate"]) for split in SPLITS if split != "id"}
    if id_success < 0.80:
        raise RuntimeError(f"ID base policy success gate failed: {id_success}")
    if any(value >= id_success for value in ood_success.values()):
        raise RuntimeError(f"OOD capability-gap gate failed: ID={id_success}, OOD={ood_success}")
    report = {
        "format": "stackpyramid_timing_protocol_gates_v1",
        "oracle": oracle,
        "base_policy": base_policy,
        "gates": {
            "oracle_each_split_at_least_100_and_90_percent": True,
            "id_base_at_least_80_percent": True,
            "each_ood_below_id_base": True,
        },
    }
    (root / "protocol_gates.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "PROTOCOL_GATES_COMPLETE").write_text("complete\n", encoding="utf-8")
    (root / "pipeline_state.json").write_text(
        json.dumps({"phase": "complete", "report": str((root / "protocol_gates.json").resolve()), "updated_at": time.time()}, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
