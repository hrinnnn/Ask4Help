#!/usr/bin/env python3
"""Audit prefix completion and target-stage reachability for StackPyramid splits."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path


SPLITS = ("stage1_ood", "stage2_ood", "stage3_ood")
DEFAULT_STARTS = {"stage1_ood": 75000, "stage2_ood": 76000, "stage3_ood": 77000}
EXPECTED_PREDICATES = {
    "stage1_ood": {"prefix": "red_grasped", "target": "red_lifted"},
    "stage2_ood": {"prefix": "red_lifted", "target": "red_placed"},
    "stage3_ood": {"prefix": "red_placed", "target": "blue_lifted"},
}


def _start(spec: object, default: int) -> int:
    if isinstance(spec, dict):
        return int(spec["start"])
    return int(spec[0])


def run_one(
    split: str,
    command: list[str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode not in (0, -6):
        raise RuntimeError(f"{split} locality audit failed; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--gpus", nargs=2, default=("0", "1"))
    parser.add_argument("--cpu-sets", nargs=2, default=("0-7", "8-15"))
    parser.add_argument("--geometry", choices=("v1", "v2"), default="v2")
    args = parser.parse_args()
    root = args.output_root
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    manifest = json.loads(args.seed_manifest.read_text(encoding="utf-8"))
    (root / "seed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest.get("geometry") != args.geometry or not manifest.get("paired_reset", {}).get("enabled"):
        raise ValueError("locality audit requires the declared geometry and paired_reset")
    if manifest.get("format", "").endswith("_v2"):
        if manifest.get("stage_predicate") != EXPECTED_PREDICATES:
            raise ValueError("seed manifest stage_predicate does not match the frozen evaluator contract")
    starts = {
        split: _start(manifest.get("base_policy", {}).get(split), DEFAULT_STARTS[split])
        for split in SPLITS
    }
    script = args.repo_root / "tools" / "evaluate_stackpyramid_xvla.py"
    commands: list[tuple[str, list[str], Path]] = []
    for index, split in enumerate(SPLITS):
        output = root / split
        command = [
            "taskset", "-c", args.cpu_sets[index % 2],
            "env", f"CUDA_VISIBLE_DEVICES={args.gpus[index % 2]}",
            f"STACKPYRAMID_OOD_GEOMETRY={args.geometry}",
            "PYTHONPATH=" + os.pathsep.join((str(args.repo_root), str(args.xvla_root))),
            str(args.python), str(script),
            "--checkpoint", str(args.checkpoint),
            "--xvla-root", str(args.xvla_root),
            "--output", str(output),
            "--split", split,
            "--episodes", "100",
            "--start-seed", str(starts[split]),
            "--max-episode-steps", "250",
            "--execute-horizon", "5",
            "--flow-steps", "5",
            "--device", "cuda",
            "--sim-backend", "cpu",
            "--render-backend", "cpu",
        ]
        commands.append((split, command, root / "logs" / f"{split}.log"))

    for start in range(0, len(commands), 2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(run_one, split, command, log) for split, command, log in commands[start:start + 2]]
            for future in futures:
                future.result()

    report: dict[str, object] = {
        "format": "stackpyramid_stage_locality_gate_v1",
        "geometry": args.geometry,
        "seed_manifest": str((root / "seed_manifest.json").resolve()),
        "id_base_summary": str((root.parent / "base_policy" / "id" / "summary.json").resolve()),
        "splits": {},
    }
    failures: list[str] = []
    for split in SPLITS:
        summary_path = root / split / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("episodes", -1)) != 100:
            failures.append(f"{split}: expected 100 episodes")
        prefix_rate = float(summary["prefix_completion_rate"])
        target_rate = float(summary["target_stage_reached_rate"])
        if prefix_rate < 0.80:
            failures.append(f"{split}: prefix completion {prefix_rate:.3f} < 0.80")
        if target_rate <= 0.0:
            failures.append(f"{split}: target-stage reach is zero")
        report["splits"][split] = {
            "episodes": int(summary["episodes"]),
            "strict_success": int(summary["strict_success"]),
            "success_rate": float(summary["strict_success"]) / 100.0,
            "stage_event_counts": summary["stage_event_counts"],
            "stage_locality_contract": summary.get("stage_locality_contract"),
            "prefix_completion": summary.get("prefix_completion"),
            "prefix_completion_rate": summary.get("prefix_completion_rate"),
            "target_stage_reached": summary.get("target_stage_reached"),
            "target_stage_reached_rate": summary.get("target_stage_reached_rate"),
            "summary": str(summary_path.resolve()),
        }

    report["passed"] = not failures
    report["failures"] = failures
    (root / "stage_locality_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        (root / "STAGE_LOCALITY_GATE_DIAGNOSTIC").write_text("\n".join(failures) + "\n", encoding="utf-8")
        raise RuntimeError("; ".join(failures))
    (root / "STAGE_LOCALITY_GATE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
