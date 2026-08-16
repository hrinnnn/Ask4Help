#!/usr/bin/env python3
"""Advance the fresh StackPyramid ID gate after checkpoint selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def write_state(path: Path, **updates: Any) -> None:
    current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "format": "stackpyramid_id_gate_after_selection_v1"
    }
    current.update(updates)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def wait_for_selection(selection_root: Path, state: Path) -> dict[str, Any]:
    while True:
        complete = selection_root / "SELECTION_COMPLETE"
        failed = selection_root / "SELECTION_FAILED"
        if complete.is_file():
            report = selection_root / "selection_summary.json"
            if not report.is_file():
                raise RuntimeError("selection completion marker has no selection_summary.json")
            return json.loads(report.read_text(encoding="utf-8"))
        if failed.is_file():
            raise RuntimeError(f"checkpoint selection failed: {failed.read_text(encoding='utf-8').strip()}")
        write_state(state, phase="waiting_for_selection", selection_root=str(selection_root))
        time.sleep(60)


def run_evaluation(args: argparse.Namespace, checkpoint: Path, output: Path) -> None:
    command = [
        "taskset", "-c", args.cpu_set, "env",
        f"CUDA_VISIBLE_DEVICES={args.gpu}",
        "STACKPYRAMID_OOD_GEOMETRY=v4",
        "PYTHONPATH=" + os.pathsep.join((str(args.repo_root), str(args.xvla_root))),
        str(args.python), str(args.repo_root / "tools/evaluate_stackpyramid_xvla.py"),
        "--checkpoint", str(checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--output", str(output),
        "--split", "id",
        "--episodes", "100",
        "--start-seed", "84400",
        "--max-episode-steps", "300",
        "--execute-horizon", "5",
        "--flow-steps", "5",
        "--device", "cuda",
        "--sim-backend", "gpu",
        "--render-backend", "gpu",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log = args.output_root / "logs" / "formal_id_gate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}, indent=2) + "\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"formal ID gate failed with return code {result.returncode}")


def run_locality(args: argparse.Namespace, checkpoint: Path, output: Path) -> None:
    command = [
        "taskset", "-c", args.locality_cpu_set, "env",
        "STACKPYRAMID_OOD_GEOMETRY=v4",
        "PYTHONPATH=" + os.pathsep.join((str(args.repo_root), str(args.xvla_root))),
        str(args.python), str(args.repo_root / "tools/run_stackpyramid_stage_locality_gate.py"),
        "--output-root", str(output),
        "--repo-root", str(args.repo_root),
        "--xvla-root", str(args.xvla_root),
        "--checkpoint", str(checkpoint),
        "--python", str(args.python),
        "--seed-manifest", str(args.seed_manifest),
        "--gpus", "0", "1",
        "--cpu-sets", "0-7", "8-15",
        "--geometry", "v4",
    ]
    log = args.output_root / "logs" / "prefix_locality_gate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command}, indent=2) + "\n")
        stream.flush()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"prefix/locality gate failed with return code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--locality-cpu-set", default="0-19")
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    state = args.output_root / "pipeline_state.json"
    try:
        selection = wait_for_selection(args.selection_root, state)
        selected = selection.get("selected", {})
        checkpoint = Path(str(selected.get("checkpoint", ""))).resolve()
        training_root = args.training_root.resolve()
        if checkpoint.parent != training_root or not checkpoint.is_dir():
            raise RuntimeError(f"selected checkpoint is outside fresh training root: {checkpoint}")
        write_state(state, phase="formal_id_gate", selected=selected, checkpoint=str(checkpoint))
        formal_output = args.output_root / "formal_id_gate"
        run_evaluation(args, checkpoint, formal_output)
        summary_path = formal_output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("episodes", -1)) != 100 or int(summary.get("video_count", -1)) != 100:
            raise RuntimeError(f"formal ID gate is incomplete: {summary}")
        report = {
            "format": "stackpyramid_formal_id_gate_v1",
            "geometry": "v4",
            "checkpoint": str(checkpoint),
            "seed_manifest": str(args.seed_manifest.resolve()),
            "episodes": int(summary["episodes"]),
            "video_count": int(summary["video_count"]),
            "ever_grasped": int(summary["ever_grasped"]),
            "ever_base_completed": int(summary["ever_base_completed"]),
            "strict_success": int(summary["strict_success"]),
            "summary": str(summary_path.resolve()),
            "minimum_strict_success": 80,
            "passed": int(summary["strict_success"]) >= 80,
        }
        (args.output_root / "formal_id_gate.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["passed"]:
            write_state(state, phase="diagnostic_id_gate_failed", formal_id_gate=report)
            (args.output_root / "ID_GATE_FAILED").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return
        (args.output_root / "ID_GATE_COMPLETE").write_text("complete\n", encoding="utf-8")
        write_state(state, phase="prefix_locality_gate", formal_id_gate=report)
        locality_output = args.output_root / "prefix_locality_gate"
        run_locality(args, checkpoint, locality_output)
        if not (locality_output / "STAGE_LOCALITY_GATE_COMPLETE").is_file():
            raise RuntimeError("prefix/locality gate did not produce completion marker")
        write_state(state, phase="downstream_unlocked", formal_id_gate=report, locality=str(locality_output))
        (args.output_root / "DOWNSTREAM_UNLOCKED").write_text("ID and prefix/locality gates passed\n", encoding="utf-8")
    except Exception as exc:
        write_state(state, phase="failed", error=repr(exc))
        (args.output_root / "PIPELINE_FAILED").write_text(repr(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
