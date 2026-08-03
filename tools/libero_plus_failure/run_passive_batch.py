#!/usr/bin/env python3
"""Resume-safe launcher for passive LIBERO(-Plus) failure traces.

The evaluator intentionally owns one environment episode at a time.  This
launcher owns the experiment schedule, so an interrupted leaderboard resumes
from immutable completed episode directories rather than creating ambiguous
partial aggregates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


def calibration_schedule(*, task_count: int, max_attempts: int, seed_base: int) -> list[dict[str, int]]:
    if task_count < 1 or max_attempts < 1:
        raise ValueError("task_count and max_attempts must be positive")
    return [
        {"ordinal": ordinal, "task_index": ordinal % task_count, "seed": seed_base + ordinal}
        for ordinal in range(max_attempts)
    ]


def manifest_schedule(manifest: Mapping[str, Any], *, task_field: str) -> list[dict[str, Any]]:
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("official manifest must contain a nonempty rows list")
    if task_field not in {"plus_task_index", "clean_task_index"}:
        raise ValueError("task_field must be plus_task_index or clean_task_index")
    schedule = []
    seen = set()
    for ordinal, row in enumerate(rows):
        configuration_id = int(row["plus_task_id"])
        if configuration_id in seen:
            raise ValueError("official manifest contains a duplicate plus_task_id")
        seen.add(configuration_id)
        schedule.append(
            {
                "ordinal": ordinal,
                "configuration_id": configuration_id,
                "task_index": int(row[task_field]),
                "category": str(row["category"]),
                # A fixed seed makes the matching clean control directly
                # comparable to its corresponding Plus configuration.
                "seed": 200000 + configuration_id,
            }
        )
    return schedule


def episode_dir(root: Path, row: Mapping[str, Any]) -> Path:
    if "configuration_id" in row:
        return root / "episodes" / ("config_%06d" % int(row["configuration_id"]))
    return root / "episodes" / ("attempt_%05d_task_%02d_seed_%06d" % (
        int(row["ordinal"]), int(row["task_index"]), int(row["seed"])
    ))


def completed_rollout(path: Path) -> bool:
    required = (path / "episode.json", path / "features.npz", path / "rollout.mp4")
    if not all(item.is_file() and item.stat().st_size > 0 for item in required):
        return False
    try:
        episode = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(episode.get("success"), bool) and bool(episode.get("timeline"))


def completed_successes(rows: Iterable[Mapping[str, Any]], root: Path) -> int:
    count = 0
    for row in rows:
        path = episode_dir(root, row)
        if completed_rollout(path):
            episode = json.loads((path / "episode.json").read_text(encoding="utf-8"))
            count += int(bool(episode["success"]))
    return count


def completed_successes_by_task(rows: Iterable[Mapping[str, Any]], root: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        path = episode_dir(root, row)
        if completed_rollout(path):
            episode = json.loads((path / "episode.json").read_text(encoding="utf-8"))
            task_index = int(row["task_index"])
            counts[task_index] = counts.get(task_index, 0) + int(bool(episode["success"]))
    return counts


def evaluator_command(args: argparse.Namespace, row: Mapping[str, Any], output_dir: Path) -> list[str]:
    command = [
        args.python,
        str(args.evaluator),
        "--suite", args.suite,
        "--task-index", str(row["task_index"]),
        "--seed", str(row["seed"]),
        "--output-dir", str(output_dir),
        "--host", args.host,
        "--port", str(args.port),
        "--resize-size", str(args.resize_size),
        "--render-gpu-device-id", str(args.render_gpu_device_id),
        "--source", args.source,
    ]
    if row.get("category") is not None:
        command.extend(("--category", str(row["category"])))
    if row.get("configuration_id") is not None:
        command.extend(("--configuration-id", str(row["configuration_id"])))
    return command


def append_event(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def execute(
    args: argparse.Namespace,
    schedule: list[dict[str, Any]],
    *,
    stop_after_successes: int | None,
    successes_per_task: int | None = None,
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    event_log = args.output_root / "launcher_events.jsonl"
    successes = completed_successes(schedule, args.output_root)
    per_task = completed_successes_by_task(schedule, args.output_root)
    for row in schedule:
        if stop_after_successes is not None and successes >= stop_after_successes:
            break
        if successes_per_task is not None and all(per_task.get(task, 0) >= successes_per_task for task in range(args.task_count)):
            break
        if successes_per_task is not None and per_task.get(int(row["task_index"]), 0) >= successes_per_task:
            continue
        target = episode_dir(args.output_root, row)
        if completed_rollout(target):
            continue
        if target.exists():
            raise RuntimeError("refusing to overwrite incomplete rollout " + str(target))
        command = evaluator_command(args, row, target)
        append_event(event_log, {"event": "launch", "row": row, "command": command})
        completed = subprocess.run(command, text=True, capture_output=True)
        (args.output_root / "logs").mkdir(exist_ok=True)
        log = args.output_root / "logs" / (target.name + ".log")
        log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            append_event(event_log, {"event": "failed", "row": row, "returncode": completed.returncode, "log": str(log)})
            raise RuntimeError("passive rollout failed; see " + str(log))
        if not completed_rollout(target):
            raise RuntimeError("rollout process exited without complete artifacts " + str(target))
        episode = json.loads((target / "episode.json").read_text(encoding="utf-8"))
        successes += int(bool(episode["success"]))
        task_index = int(row["task_index"])
        per_task[task_index] = per_task.get(task_index, 0) + int(bool(episode["success"]))
        append_event(event_log, {"event": "complete", "row": row, "success": bool(episode["success"]), "path": str(target)})
    summary = {
        "format": "libero_plus_passive_batch_v1",
        "source": args.source,
        "suite": args.suite,
        "scheduled": len(schedule),
        "completed": sum(completed_rollout(episode_dir(args.output_root, row)) for row in schedule),
        "successful": completed_successes(schedule, args.output_root),
        "required_successes": stop_after_successes,
        "successful_by_task": completed_successes_by_task(schedule, args.output_root),
        "required_successes_per_task": successes_per_task,
    }
    (args.output_root / "batch_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if stop_after_successes is not None and summary["successful"] < stop_after_successes:
        raise RuntimeError("calibration exhausted attempts before required policy successes")
    if successes_per_task is not None and any(summary["successful_by_task"].get(str(task), summary["successful_by_task"].get(task, 0)) < successes_per_task for task in range(args.task_count)):
        raise RuntimeError("calibration exhausted attempts before every task reached its success quota")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibration", "manifest"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", choices=("clean", "libero_plus"), required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--evaluator", type=Path, default=Path(__file__).with_name("evaluate_passive_rollouts.py"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--render-gpu-device-id", type=int, default=1)
    parser.add_argument("--required-successes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--successes-per-task", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-field", choices=("plus_task_index", "clean_task_index"))
    args = parser.parse_args()
    if args.mode == "calibration":
        schedule = calibration_schedule(task_count=args.task_count, max_attempts=args.max_attempts, seed_base=args.seed_base)
        if args.successes_per_task < 0:
            raise ValueError("successes-per-task must be non-negative")
        quota = args.successes_per_task or None
        if quota is not None and args.required_successes != quota * args.task_count:
            raise ValueError("required-successes must equal successes-per-task * task-count when task-balanced calibration is enabled")
        execute(args, schedule, stop_after_successes=args.required_successes, successes_per_task=quota)
        return
    if args.manifest is None or args.task_field is None:
        raise ValueError("manifest mode requires --manifest and --task-field")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    execute(args, manifest_schedule(manifest, task_field=args.task_field), stop_after_successes=None)


if __name__ == "__main__":
    main()
