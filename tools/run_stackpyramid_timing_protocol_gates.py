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


def _seed_values(spec: object) -> list[int]:
    if isinstance(spec, dict):
        start = int(spec["start"])
        count = int(spec["count"])
        return list(range(start, start + count))
    return [int(seed) for seed in spec]


def _validate_versioned_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    geometry = manifest.get("geometry")
    expected_format = f"stackpyramid_timing_protocol_seed_manifest_{geometry}"
    if geometry not in {"v2", "v3", "v4"} or manifest.get("format") != expected_format:
        raise ValueError("versioned recovery requires a frozen v2, v3, or v4 seed manifest")
    if not manifest.get("declared_before_execution"):
        raise ValueError("versioned manifest must be declared before execution")
    if not manifest.get("paired_reset", {}).get("enabled"):
        raise ValueError("versioned manifest must explicitly enable paired resets")
    predicates = {
        "stage1_ood": {"prefix": "red_grasped", "target": "red_lifted"},
        "stage2_ood": {"prefix": "red_lifted", "target": "red_placed"},
        "stage3_ood": {"prefix": "red_placed", "target": "blue_lifted"},
    }
    if manifest.get("stage_predicate") != predicates:
        raise ValueError("versioned manifest stage_predicate does not match the frozen evaluator contract")
    oracle = manifest.get("oracle", {})
    oracle_seeds = [_seed_values(oracle[split]) for split in SPLITS]
    if any(len(seeds) != 100 for seeds in oracle_seeds) or any(seeds != oracle_seeds[0] for seeds in oracle_seeds[1:]):
        raise ValueError("versioned Oracle seeds must be 100 continuous paired seeds across all splits")
    base = manifest.get("base_policy", {})
    base_seeds = [_seed_values(base[split]) for split in SPLITS]
    if any(len(seeds) != 100 for seeds in base_seeds) or any(seeds != base_seeds[0] for seeds in base_seeds[1:]):
        raise ValueError("versioned base-policy seeds must be 100 continuous paired seeds across all splits")
    final = manifest.get("final_evaluation", {})
    final_seeds = [_seed_values(final[split]) for split in SPLITS]
    if any(len(seeds) != 100 for seeds in final_seeds) or any(seeds != final_seeds[0] for seeds in final_seeds[1:]):
        raise ValueError("versioned final-evaluation seeds must be 100 continuous paired seeds across all splits")
    timing = manifest.get("timing_collection", {})
    if any(len(_seed_values(timing[split])) != 100 for split in SPLITS[1:]):
        raise ValueError("versioned timing-collection candidate seeds must each contain 100 seeds")


def _write_seed_manifest(root: Path, source: Path | None = None) -> Path:
    manifest = {
        "format": "stackpyramid_timing_protocol_seed_manifest_v1",
        "declared_before_execution": True,
        "oracle": {
            "id": list(range(70000, 70100)),
            "stage1_ood": list(range(71000, 71100)),
            "stage2_ood": list(range(72000, 72100)),
            "stage3_ood": list(range(73000, 73100)),
        },
        "base_policy": {
            "id": {"start": 74000, "count": 100},
            "stage1_ood": {"start": 75000, "count": 100},
            "stage2_ood": {"start": 76000, "count": 100},
            "stage3_ood": {"start": 77000, "count": 100},
        },
    }
    path = root / "seed_manifest.json"
    if source is not None:
        path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


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
    script = args.repo_root / "tools" / "run_stackpyramid_oracle_gate.py"
    output = root / "oracle_gate"
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    seed_manifest = root / "seed_manifest.json"
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
                    "--repo-root", str(args.repo_root),
                    "--xvla-root", str(args.xvla_root),
                    "--output", str(split_output),
                    "--split", split,
                    "--episodes", "100",
                    "--start-seed", "0",
                    "--seed-manifest", str(seed_manifest),
                    "--sim-backend", "cpu",
                    "--render-backend", "cpu",
                ],
            )
            commands.append((command, root / "logs" / f"oracle_{split}.log", f"oracle_{split}"))
        _run_pair(commands)
        for split in pair:
            summary = _summary(output / split)
            success_rate = float(summary["success_rate"])
            if success_rate < 0.90:
                raise RuntimeError(f"oracle success gate failed for {split}: {success_rate}")
            results[split] = {
                "episodes": int(summary["episodes"]),
                "strict_successes": int(summary["strict_successes"]),
                "success_rate": success_rate,
                "summary": str((output / split / "summary.json").resolve()),
            }
    return results


def _run_base_policy_gates(args: argparse.Namespace, root: Path) -> dict[str, dict[str, object]]:
    script = args.repo_root / "tools" / "evaluate_stackpyramid_xvla.py"
    output = root / "base_policy"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "seed_manifest.json").read_text(encoding="utf-8"))
    starts = {
        split: int(manifest["base_policy"][split]["start"])
        if isinstance(manifest["base_policy"][split], dict)
        else int(manifest["base_policy"][split][0])
        for split in SPLITS
    }
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
                    "--sim-backend", "cpu",
                    "--render-backend", "cpu",
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
    parser.add_argument("--seed-manifest", type=Path)
    args = parser.parse_args()
    root = args.output_root
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    seed_manifest = _write_seed_manifest(root, args.seed_manifest)
    if args.seed_manifest is not None:
        _validate_versioned_manifest(seed_manifest)
    (root / "pipeline_state.json").write_text(
        json.dumps(
            {
                "phase": "oracle",
                "seed_manifest": str(seed_manifest.resolve()),
                "updated_at": time.time(),
            },
            indent=2,
        ) + "\n"
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
        "benchmark_version": json.loads(seed_manifest.read_text(encoding="utf-8")).get("benchmark_version"),
        "geometry": json.loads(seed_manifest.read_text(encoding="utf-8")).get("geometry", "v1"),
        "seed_manifest": str(seed_manifest.resolve()),
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
