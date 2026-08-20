#!/usr/bin/env python3
"""Audit first failure bottlenecks for the StackPyramid grasp-recovery baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


EVENTS = ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")


def _max_repeat(actions: np.ndarray) -> int:
    if len(actions) == 0:
        return 0
    best = current = 1
    for left, right in zip(actions[:-1], actions[1:]):
        if np.allclose(left, right, atol=1e-5, rtol=0.0):
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _first_bottleneck(row: dict[str, Any], actions: np.ndarray) -> tuple[str, dict[str, Any]]:
    events = row.get("stage_events", {})
    steps = int(row.get("steps", len(actions)))
    close_steps = int(np.sum(actions[:, -1] < 0.0)) if actions.ndim == 2 and actions.shape[1] else 0
    max_repeat = _max_repeat(actions)
    delta = np.abs(np.diff(actions, axis=0)).mean() if len(actions) > 1 else 0.0
    evidence = {
        "steps": steps,
        "horizon_reached": steps >= 300,
        "close_action_steps": close_steps,
        "closed_action_fraction": close_steps / max(1, len(actions)),
        "max_identical_action_run": max_repeat,
        "mean_action_delta": float(delta),
        "last_action": actions[-1].tolist() if len(actions) else None,
    }
    if bool(row.get("strict_success")):
        return "success", evidence
    if not events.get("red_grasped", False):
        if steps >= 300 and (max_repeat >= 8 or delta < 1e-3):
            return "pre_grasp_hover_or_repeated_action", evidence
        if close_steps > 0:
            return "gripper_close_without_red_contact", evidence
        return "pre_grasp_no_contact", evidence
    if not events.get("red_lifted", False):
        return "red_grasp_without_stable_lift", evidence
    if not events.get("red_placed", False):
        return "red_placement_failure", evidence
    if not events.get("blue_lifted", False):
        return "blue_grasp_or_lift_failure", evidence
    return "other_failure", evidence


def audit(root: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = []
    counts: Counter[str] = Counter()
    for row in rows:
        actions = np.load(row["actions"])
        bottleneck, evidence = _first_bottleneck(row, actions)
        counts[bottleneck] += 1
        records.append({
            "seed": row.get("seed"),
            "strict_success": bool(row.get("strict_success")),
            "events": row.get("stage_events", {}),
            "first_bottleneck": bottleneck,
            "evidence": evidence,
        })
    summary = {
        "format": "stackpyramid_grasp_recovery_baseline_audit_v1",
        "root": str(root),
        "episodes": len(rows),
        "first_bottleneck_counts": dict(counts),
        "strict_success": sum(int(bool(row.get("strict_success"))) for row in rows),
        "event_counts": {
            event: sum(int(bool(row.get("stage_events", {}).get(event))) for row in rows)
            for event in EVENTS
        },
        "records": records,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("episodes", "strict_success", "first_bottleneck_counts", "event_counts")}, indent=2))


if __name__ == "__main__":
    main()
