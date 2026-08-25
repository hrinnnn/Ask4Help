#!/usr/bin/env python3
"""Independent evidence audit for fixed-step timing calibration collections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(
    root: Path,
    anchors: list[int],
    *,
    seeds: list[int],
    endpoint: str = "success",
    minimum_recoverability: float = 0.9,
) -> dict[str, Any]:
    expected = set(int(seed) for seed in seeds)
    by_anchor: dict[str, Any] = {}
    recoverable_sets: list[set[int]] = []
    errors: list[str] = []
    for step in anchors:
        anchor = root / f"step_{step}"
        summary_path = anchor / "summary.json"
        episodes_path = anchor / "episodes.jsonl"
        if not summary_path.is_file() or not episodes_path.is_file():
            errors.append(f"step_{step}:missing_summary_or_episodes")
            continue
        summary = read_json(summary_path)
        rows = read_jsonl(episodes_path)
        row_seeds = [int(row["seed"]) for row in rows]
        seed_set = set(row_seeds)
        anchor_errors: list[str] = []
        if len(rows) != len(seeds) or seed_set != expected or len(seed_set) != len(row_seeds):
            anchor_errors.append("seed_denominator_or_uniqueness")
        if int(summary.get("raw_total", -1)) != len(seeds):
            anchor_errors.append("summary_raw_total")
        recoverable: set[int] = set()
        for row in rows:
            seed = int(row["seed"])
            state_path = Path(row["task_states"])
            action_path = Path(row.get("actions", ""))
            if not row.get("actions"):
                action_path = anchor / "raw_archive" / "actions" / f"episode_{int(row.get('attempt_index', 0)):06d}_seed_{seed:06d}.npy"
            video_path = Path(row.get("video", ""))
            if not state_path.is_file():
                anchor_errors.append(f"seed_{seed}:missing_task_states")
                continue
            if not action_path.is_file():
                anchor_errors.append(f"seed_{seed}:missing_actions")
            if not video_path.is_file():
                anchor_errors.append(f"seed_{seed}:missing_video")
            try:
                state_count = int(np.load(state_path, mmap_mode="r").shape[0])
                action_count = int(row.get("steps", -1))
                if state_count != action_count + 1:
                    anchor_errors.append(f"seed_{seed}:state_action_length_mismatch")
            except (OSError, ValueError):
                anchor_errors.append(f"seed_{seed}:invalid_task_states")
            value = row.get(endpoint)
            if value is None and endpoint == "success":
                value = row.get("strict_success", row.get("success", False))
            if bool(value):
                recoverable.add(seed)
        recoverable_sets.append(recoverable)
        rate = len(recoverable) / max(1, len(seeds))
        if rate < minimum_recoverability:
            anchor_errors.append("UNRECOVERABLE_REGION")
        errors.extend(f"step_{step}:{error}" for error in anchor_errors)
        by_anchor[str(step)] = {
            "raw_total": len(rows),
            "recoverable": len(recoverable),
            "recoverability": rate,
            "recoverable_seeds": sorted(recoverable),
            "errors": anchor_errors,
        }
    common = sorted(set.intersection(*recoverable_sets)) if recoverable_sets else []
    return {
        "format": "xvla_fixed_timing_calibration_audit_v1",
        "root": str(root.resolve()),
        "anchors": anchors,
        "expected_seed_count": len(seeds),
        "endpoint": endpoint,
        "minimum_recoverability": minimum_recoverability,
        "anchor_audit": by_anchor,
        "common_recoverable_seed_count": len(common),
        "common_recoverable_seeds": common,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--endpoint", choices=("success", "strict_success", "ever_grasped"), default="success")
    parser.add_argument("--minimum-recoverability", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    payload = audit(
        args.root,
        args.anchors,
        seeds=list(range(args.seed_start, args.seed_end + 1)),
        endpoint=args.endpoint,
        minimum_recoverability=args.minimum_recoverability,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
