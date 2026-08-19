#!/usr/bin/env python3
"""Run an independent ID checkpoint-selection probe for StackPyramid v4."""

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
        "format": "stackpyramid_id_checkpoint_selection_v1"
    }
    current.update(updates)
    current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def checkpoint_steps(root: Path) -> list[int]:
    steps = []
    for path in root.glob("ckpt-*"):
        try:
            steps.append(int(path.name.removeprefix("ckpt-")))
        except ValueError:
            continue
    return sorted(steps)


def run_probe(args: argparse.Namespace, checkpoint: Path, output: Path, log: Path) -> dict[str, Any]:
    command = [
        str(args.python),
        str(args.repo_root / "tools/evaluate_stackpyramid_xvla.py"),
        "--checkpoint", str(checkpoint),
        "--xvla-root", str(args.xvla_root),
        "--output", str(output),
        "--split", "id",
        "--episodes", str(args.episodes),
        "--start-seed", str(args.start_seed),
        "--max-episode-steps", "250",
        "--execute-horizon", "5",
        "--flow-steps", "5",
        "--device", "cuda",
        "--sim-backend", "gpu",
        "--render-backend", "gpu",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["STACKPYRAMID_OOD_GEOMETRY"] = "v4"
    env["PYTHONPATH"] = os.pathsep.join([str(args.repo_root), str(args.xvla_root)])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"command": command, "gpu": args.gpu}) + "\n")
        stream.flush()
        result = subprocess.run(
            ["taskset", "-c", args.cpu_set, *command],
            cwd=args.repo_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    summary_path = output / "summary.json"
    marker = output / "EVAL_COMPLETE"
    if result.returncode != 0 or not summary_path.is_file() or not marker.is_file():
        raise RuntimeError(f"selection probe failed for {checkpoint}: rc={result.returncode}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("episodes", -1)) != args.episodes or int(summary.get("video_count", -1)) != args.episodes:
        raise RuntimeError(f"incomplete selection probe for {checkpoint}: {summary}")
    return {
        "checkpoint": str(checkpoint),
        "step": int(checkpoint.name.removeprefix("ckpt-")),
        "episodes": int(summary["episodes"]),
        "ever_grasped": int(summary["ever_grasped"]),
        "ever_base_completed": int(summary["ever_base_completed"]),
        "strict_success": int(summary["strict_success"]),
        "stage_event_counts": summary.get("stage_event_counts", {}),
        "summary": str(summary_path),
        "videos": int(summary["video_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=88400)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument(
        "--checkpoint-steps",
        default="",
        help="Comma-separated checkpoint steps; empty preserves the original 2000..20000 schedule.",
    )
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    state_path = args.output_root / "selection_state.json"
    log_dir = args.output_root / "logs"
    write_state(state_path, phase="starting", training_root=str(args.training_root), start_seed=args.start_seed, episodes=args.episodes)
    results: list[dict[str, Any]] = []
    try:
        steps = checkpoint_steps(args.training_root)
        expected = (
            [int(value) for value in args.checkpoint_steps.split(",") if value.strip()]
            if args.checkpoint_steps
            else list(range(2000, 20001, 2000))
        )
        expected = sorted(expected)
        if steps != expected:
            raise RuntimeError(f"unexpected checkpoint set: {steps}; expected {expected}")
        for index, step in enumerate(steps, start=1):
            checkpoint = args.training_root / f"ckpt-{step}"
            output = args.output_root / f"ckpt-{step}"
            log = log_dir / f"ckpt-{step}.log"
            write_state(state_path, phase="running", index=index, total=len(steps), checkpoint=str(checkpoint), results=results)
            result = run_probe(args, checkpoint, output, log)
            results.append(result)
            write_state(state_path, phase="running", index=index, total=len(steps), checkpoint=str(checkpoint), results=results)
        selected = max(results, key=lambda row: (row["strict_success"], row["ever_base_completed"], row["ever_grasped"], row["step"]))
        report = {
            "format": "stackpyramid_id_checkpoint_selection_v1",
            "geometry": "v4",
            "split": "id",
            "seed_manifest": {"start": args.start_seed, "count": args.episodes},
            "selection_is_not_formal_gate": True,
            "checkpoints": results,
            "selected": selected,
            "selection_rule": "max strict_success, then ever_base_completed, then ever_grasped, then later step",
        }
        (args.output_root / "selection_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (args.output_root / "SELECTED_CHECKPOINT.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
        (args.output_root / "SELECTION_COMPLETE").write_text("complete\n", encoding="utf-8")
        write_state(state_path, phase="complete", selected=selected, results=results)
    except Exception as exc:
        write_state(state_path, phase="failed", error=repr(exc), results=results)
        (args.output_root / "SELECTION_FAILED").write_text(repr(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
