#!/usr/bin/env python3
"""Machine-check the repaired StackPyramid Oracle trace contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_EVENTS = ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")
MAX_ACTION_STEPS = 300


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    events = row["event_first_steps"]
    calls = row["planner_calls"]
    real_opens = [call for call in calls if call["method"] == "open_gripper" and not call["dry_run"]]
    real_closes = [call for call in calls if call["method"] == "close_gripper" and not call["dry_run"]]
    red_closes = [call for call in real_closes if call["after"]["grasped"][0]]
    red_releases = [
        call
        for call in real_opens
        if call["before"]["grasped"][0] and not call["after"]["grasped"][0]
    ]
    blue_closes = [call for call in real_closes if call["after"]["grasped"][2]]
    event_order_pass = (
        all(name in events for name in REQUIRED_EVENTS)
        and all(events[left] < events[right] for left, right in zip(REQUIRED_EVENTS, REQUIRED_EVENTS[1:]))
    )
    red_close = red_closes[0] if red_closes else None
    red_release = red_releases[0] if red_releases else None
    blue_close = blue_closes[0] if blue_closes else None
    transport_closed = bool(
        red_close is not None
        and red_release is not None
        and red_close["action_step_end"] <= red_release["action_step_start"]
        and all(
            transition["to_closed"]
            for transition in row["gripper_transitions"]
            if red_close["action_step_end"] <= transition["action_step"] < red_release["action_step_start"]
        )
    )
    release_at_target = bool(
        red_release is not None
        and red_release["before"]["red_green_xy_distance"]
        <= red_release["before"]["red_green_xy_tolerance"]
        and events.get("red_placed", -1) >= red_release["action_step_start"]
    )
    blue_after_red = bool(
        blue_close is not None
        and events.get("red_placed", MAX_ACTION_STEPS + 1) < blue_close["action_step_start"]
    )
    checks = {
        "strict_success": bool(row["strict_success"]),
        "exactly_one_red_grasp": len(red_closes) == 1,
        "red_release_count_one": len(red_releases) == 1,
        "closed_during_red_transport": transport_closed,
        "release_at_target": release_at_target,
        "event_order": event_order_pass,
        "blue_after_red_placement": blue_after_red,
        "bounded_action_steps": int(row["action_steps"]) <= MAX_ACTION_STEPS,
        "evidence_paths_present": all(row.get(key) for key in ("actions", "state", "video")),
    }
    return {
        "seed": row["seed"],
        "checks": checks,
        "passed": all(checks.values()),
        "red_close_action_step": None if red_close is None else red_close["action_step_start"],
        "red_release_action_step": None if red_release is None else red_release["action_step_start"],
        "blue_close_action_step": None if blue_close is None else blue_close["action_step_start"],
        "event_first_steps": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = [validate_row(row) for row in summary["rows"]]
    report = {
        "format": "stackpyramid_oracle_trace_validation_v1",
        "source_summary": str(args.summary),
        "episodes": len(rows),
        "passed_episodes": sum(int(row["passed"]) for row in rows),
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
