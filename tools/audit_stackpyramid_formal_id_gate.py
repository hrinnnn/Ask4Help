#!/usr/bin/env python3
"""Audit complete formal StackPyramid ID gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def audit(root: Path, *, expected: int, minimum_successes: int) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    if len(rows) != expected or int(summary.get("episodes", -1)) != expected:
        errors.append(f"episodes denominator mismatch: rows={len(rows)} summary={summary.get('episodes')}")
    videos = list((root / "videos").glob("*.mp4"))
    actions = list((root / "actions").glob("*.npy"))
    states = list((root / "states").glob("*.json"))
    if len(videos) != expected:
        errors.append(f"video_count={len(videos)} expected={expected}")
    if len(actions) != expected:
        errors.append(f"action_count={len(actions)} expected={expected}")
    if len(states) != expected:
        errors.append(f"state_count={len(states)} expected={expected}")
    for row in rows:
        for key in ("video", "actions", "state_timeline"):
            path = Path(row.get(key, ""))
            if not path.is_file():
                errors.append(f"{row.get('seed')}: missing {key} {path}")
        if Path(row.get("state_timeline", "")).is_file():
            state_rows = json.loads(Path(row["state_timeline"]).read_text(encoding="utf-8"))
            if len(state_rows) != int(row.get("steps", -1)) + 1:
                errors.append(f"{row.get('seed')}: state length mismatch")
        if Path(row.get("actions", "")).is_file():
            action_array = np.load(row["actions"])
            if action_array.shape[0] != int(row.get("steps", -1)):
                errors.append(f"{row.get('seed')}: action length mismatch")
        required_events = ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")
        if not all(name in row.get("stage_events", {}) for name in required_events):
            errors.append(f"{row.get('seed')}: incomplete stage events")
    report = {
        "format": "stackpyramid_formal_id_gate_audit_v1",
        "root": str(root),
        "expected_episodes": expected,
        "episodes": len(rows),
        "videos": len(videos),
        "actions": len(actions),
        "states": len(states),
        "strict_success": int(summary.get("strict_success", 0)),
        "strict_success_threshold": minimum_successes,
        "errors": errors,
        "audit_pass": not errors and int(summary.get("strict_success", 0)) >= minimum_successes,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--minimum-successes", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, expected=args.expected, minimum_successes=args.minimum_successes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
