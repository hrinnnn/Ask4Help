#!/usr/bin/env python3
"""Build the auditable clean/Plus task pairing used by the main table."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from libero_plus_failure_protocol import build_libero_plus_manifest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def task_balanced_hash_subset(rows: list[dict], *, count: int, seed: int) -> tuple[list[dict], dict]:
    """Select a reproducible task-balanced subset without inspecting rollouts.

    This is deliberately based only on official configuration identifiers.  It
    never looks at a policy outcome or detector score, so selecting a practical
    100-episode leaderboard subset cannot leak failure labels into the protocol.
    """

    if count < 1:
        raise ValueError("count must be positive")
    groups: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        groups[int(row["clean_task_index"])].append(dict(row))
    task_ids = sorted(groups)
    if not task_ids:
        raise ValueError("cannot select from an empty manifest")
    base, remainder = divmod(count, len(task_ids))
    selected: list[dict] = []
    per_task: dict[str, int] = {}
    for position, task_id in enumerate(task_ids):
        quota = base + int(position < remainder)
        ranked = sorted(
            groups[task_id],
            key=lambda row: hashlib.sha256(f"{seed}:{int(row['plus_task_id'])}".encode("ascii")).hexdigest(),
        )
        if len(ranked) < quota:
            raise ValueError(f"task {task_id} has only {len(ranked)} configurations, needs {quota}")
        selected.extend(ranked[:quota])
        per_task[str(task_id)] = quota
    selected.sort(key=lambda row: int(row["plus_task_id"]))
    difficulty_counts = collections.Counter(int(row["difficulty_level"]) for row in selected)
    return selected, {
        "mode": "task_balanced_hash",
        "seed": int(seed),
        "requested_configurations": int(count),
        "selected_configurations": len(selected),
        "per_clean_task": per_task,
        "difficulty_counts": {str(key): int(value) for key, value in sorted(difficulty_counts.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--clean-tasks", type=Path, required=True, help="JSON list from the unmodified LIBERO-10 install")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["Camera Viewpoints", "Robot Initial States", "Objects Layout"])
    parser.add_argument("--min-difficulty-level", type=int)
    parser.add_argument("--max-difficulty-level", type=int)
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--selection-seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite " + str(args.output))
    classifications = json.loads(args.classification.read_text(encoding="utf-8"))
    clean = json.loads(args.clean_tasks.read_text(encoding="utf-8"))
    if args.suite not in classifications:
        raise KeyError("classification has no suite " + args.suite)
    rows = build_libero_plus_manifest(
        classifications=classifications[args.suite],
        clean_tasks=clean,
        categories=args.categories,
        min_difficulty_level=args.min_difficulty_level,
        max_difficulty_level=args.max_difficulty_level,
    )
    selection = {"mode": "all_official_configurations", "selected_configurations": len(rows)}
    if args.max_configurations is not None:
        rows, selection = task_balanced_hash_subset(rows, count=args.max_configurations, seed=args.selection_seed)
    payload = {
        "format": "libero_plus_failure_task_manifest_v1",
        "suite": args.suite,
        "categories": args.categories,
        "min_difficulty_level": args.min_difficulty_level,
        "max_difficulty_level": args.max_difficulty_level,
        "classification": str(args.classification),
        "classification_sha256": sha256(args.classification),
        "clean_tasks": str(args.clean_tasks),
        "clean_tasks_sha256": sha256(args.clean_tasks),
        "selection": selection,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
