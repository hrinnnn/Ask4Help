#!/usr/bin/env python3
"""Relate frozen X-VLA gate alarms to a task-policy timing knee.

This is an audit-only utility.  It consumes a passive detector summary and a
threshold calibration produced on validation ID data; it never chooses a
threshold from OOD rows and it never changes the recorded policy actions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METHOD_SCORE_KEYS = {
    "input_pca": "vlm_input_pool_pca",
    "bridge_pca": "vlm_action_bridge_pca",
    "action_pca": "action_block_01_pca",
    "diffdagger": "diffdagger",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def first_alarm(row: dict[str, Any], score_key: str, threshold: float) -> int | None:
    for point in row.get("timeline", []):
        value = point.get("scores", {}).get(score_key)
        if value is None or not math.isfinite(float(value)):
            continue
        if float(value) > threshold:
            return int(point.get("env_step", point.get("decision_index", 0)))
    return None


def summarize_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    knee_set: list[int],
    threshold: float | None,
    knee_tolerance: int,
    fixed_step: int | None = None,
) -> dict[str, Any]:
    if not knee_set:
        raise ValueError("knee_set must not be empty")
    alarms: list[dict[str, Any]] = []
    for row in rows:
        alarm = fixed_step if fixed_step is not None else first_alarm(row, METHOD_SCORE_KEYS[method], float(threshold))
        if alarm is None:
            alarms.append(
                {
                    "episode_index": row.get("episode_index"),
                    "seed": row.get("seed"),
                    "alarm_observed": False,
                    "alarm_step": None,
                    "knee_distance": None,
                    "knee_hit": None,
                }
            )
            continue
        distance = min(abs(alarm - knee) for knee in knee_set)
        alarms.append(
            {
                "episode_index": row.get("episode_index"),
                "seed": row.get("seed"),
                "alarm_observed": True,
                "alarm_step": alarm,
                "knee_distance": distance,
                "knee_hit": bool(distance <= knee_tolerance),
            }
        )
    observed = [row for row in alarms if row["alarm_observed"]]
    distances = [row["knee_distance"] for row in observed]
    hits = [row for row in observed if row["knee_hit"]]
    return {
        "method": method,
        "score_key": None if fixed_step is not None else METHOD_SCORE_KEYS[method],
        "threshold": threshold,
        "fixed_step": fixed_step,
        "episodes": len(rows),
        "alarms_observed": len(observed),
        "alarm_miss_rate": 1.0 - len(observed) / max(1, len(rows)),
        "knee_distance_mean": None if not distances else sum(distances) / len(distances),
        "knee_distance_median": None if not distances else sorted(distances)[len(distances) // 2],
        "knee_hit_rate_conditional": None if not observed else len(hits) / len(observed),
        "knee_hit_rate_all_episodes": None if not rows else len(hits) / len(rows),
        "episodes_detail": alarms,
    }


def summarize(
    summary: Path,
    calibration: Path,
    *,
    task: str,
    knee_set: list[int],
    methods: list[str],
    knee_tolerance: int,
    failure_recovery_step: int,
) -> dict[str, Any]:
    payload = read_json(summary)
    rows = list(payload.get("rows", []))
    if not rows:
        raise ValueError(f"summary has no rows: {summary}")
    calibration_payload = read_json(calibration)
    calibrated = calibration_payload.get("methods", {})
    result_methods: dict[str, Any] = {}
    for method in methods:
        if method == "failure_recovery":
            result_methods[method] = summarize_method(
                rows,
                method=method,
                knee_set=knee_set,
                threshold=None,
                knee_tolerance=knee_tolerance,
                fixed_step=failure_recovery_step,
            )
            continue
        if method not in METHOD_SCORE_KEYS:
            raise ValueError(f"unsupported method: {method}")
        score_key = METHOD_SCORE_KEYS[method]
        threshold_info = calibrated.get(score_key)
        if threshold_info is None:
            raise KeyError(f"calibration has no threshold for {score_key}")
        threshold = float(threshold_info["threshold"])
        result_methods[method] = summarize_method(
            rows,
            method=method,
            knee_set=knee_set,
            threshold=threshold,
            knee_tolerance=knee_tolerance,
        )
    return {
        "format": "xvla_gate_to_knee_audit_v1",
        "task": task,
        "summary": str(summary.resolve()),
        "calibration": str(calibration.resolve()),
        "knee_confidence_set": knee_set,
        "knee_tolerance_steps": knee_tolerance,
        "failure_recovery_step": failure_recovery_step,
        "methods": result_methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--knee-set", type=int, nargs="+", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["input_pca", "bridge_pca", "action_pca", "diffdagger", "failure_recovery"],
    )
    parser.add_argument("--knee-tolerance", type=int, default=5)
    parser.add_argument("--failure-recovery-step", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.knee_tolerance < 0 or args.failure_recovery_step < 0:
        raise ValueError("timing tolerances must be non-negative")
    result = summarize(
        args.summary,
        args.calibration,
        task=args.task,
        knee_set=args.knee_set,
        methods=args.methods,
        knee_tolerance=args.knee_tolerance,
        failure_recovery_step=args.failure_recovery_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
