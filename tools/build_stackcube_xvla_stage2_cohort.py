#!/usr/bin/env python3
"""Freeze policy-failure seeds for the controlled StackCube timing study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def select_cohort(rows: list[dict], count: int) -> list[dict]:
    selected = [
        row
        for row in rows
        if not bool(row["success"])
        and bool(row["grasped_once"])
        and bool(row["lifted_once"])
        and bool(row["stable_lift_boundary_once"])
        and bool(row["dropped_after_lift_two_boundaries"])
    ]
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads(args.policy_summary.read_text(encoding="utf-8"))
    selected = select_cohort(summary["rows"], args.count)
    if len(selected) < args.count:
        raise RuntimeError(
            f"only {len(selected)} target-related failures satisfy the cohort rule; "
            f"requested {args.count}"
        )
    payload = {
        "format": "xvla_stackcube_stage2_timing_cohort_v1",
        "source": str(args.policy_summary.resolve()),
        "rule": "policy failure after a stable grasped lift boundary and two dropped decision boundaries",
        "count": len(selected),
        "seeds": [int(row["seed"]) for row in selected],
        "rows": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
