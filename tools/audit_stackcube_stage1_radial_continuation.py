#!/usr/bin/env python3
"""Audit radial continuation without changing the smoke decision or thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ID_DISTANCE_RANGE_M = (0.08, 0.10)
CHUNK_SIZE = 5


def audit(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    audited = []
    for row in rows:
        returns = [
            event for event in row["score_timeline"] if event.get("return_after_chunk")
        ]
        first = returns[0] if returns else None
        handoff = None
        if first is not None:
            state_index = min(
                first["env_step"] + CHUNK_SIZE, len(row["state_timeline"]) - 1
            )
            handoff = row["state_timeline"][state_index]
        distance = None
        outside_id_range = None
        if handoff is not None:
            distance = float(
                np.linalg.norm(
                    np.asarray(handoff["red_xy"], dtype=np.float64)
                    - np.asarray(handoff["green_xy"], dtype=np.float64)
                )
            )
            outside_id_range = not (
                ID_DISTANCE_RANGE_M[0] <= distance <= ID_DISTANCE_RANGE_M[1]
            )
        audited.append(
            {
                "seed": row["seed"],
                "continuation_success": bool(row["continuation_success"]),
                "return_to_policy": bool(row["real_return_to_policy"]),
                "false_release": bool(row["false_release"]),
                "handoff_red_green_distance_m": distance,
                "handoff_outside_id_distance_range": outside_id_range,
                "handoff_red_lifted": None
                if handoff is None
                else bool(handoff["predicates"]["red_lifted"]),
                "handoff_red_grasped": None
                if handoff is None
                else bool(handoff["predicates"]["red_grasped"]),
            }
        )
    returned = [row for row in audited if row["return_to_policy"]]
    outside = [row for row in returned if row["handoff_outside_id_distance_range"]]
    return {
        "source_summary": str(summary_path),
        "episodes": len(audited),
        "return_to_policy_episodes": len(returned),
        "continuation_successes": sum(int(row["continuation_success"]) for row in audited),
        "returned_with_carried_ood_distance": len(outside),
        "id_distance_range_m": list(ID_DISTANCE_RANGE_M),
        "conclusion": (
            "RADIAL_CONTINUATION_GATE_FAILED_PERSISTENT_CARRIED_OOD"
            if outside
            else "NO_RETURN_HANDOFF_DISTANCE_EVIDENCE"
        ),
        "interpretation": "The red-only initial shift remains in the carried object pose at the stable-lift handoff; this diagnostic does not alter the frozen radial task, threshold, or success predicate.",
        "rows": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
