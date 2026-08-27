#!/usr/bin/env python3
"""Diagnostic threshold sweep for the frozen ERD-Pose summaries.

This utility does not change the formal X-VLA pipeline.  It replays the
already-emitted per-episode distance timelines in an ERD summary and reports
how the persistent-crossing timing changes with the expert calibration
quantile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_QUANTILES = (0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99)


def _first_crossing(values: np.ndarray, steps: np.ndarray, threshold: float, persistence: int) -> int | None:
    for index in range(0, len(values) - persistence + 1):
        if np.all(values[index : index + persistence] > threshold):
            return int(steps[index])
    return None


def sweep(summary_path: Path, quantiles: tuple[float, ...], horizon: int, persistence: int) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = np.asarray(payload["threshold"]["calibration_values"], dtype=np.float64)
    rows = payload["rows"]
    supported_total = sum(bool(row.get("context_supported", False)) for row in rows)
    stride = int(payload.get("learner", {}).get("decision_stride", 5))
    steps = np.arange(0, horizon + stride, stride, dtype=np.int64)
    steps = steps[steps <= horizon]

    candidates: list[dict[str, Any]] = []
    for quantile in quantiles:
        threshold = float(np.quantile(calibration, quantile))
        alarms: list[int] = []
        supported_alarms = 0
        pre_grasp = 0
        post_grasp = 0
        leads: list[int] = []
        late = 0
        event_covered = 0
        for row in rows:
            values = np.asarray(row["distance_at_decision_steps"], dtype=np.float64)[: len(steps)]
            alarm = _first_crossing(values, steps, threshold, persistence)
            if alarm is None:
                continue
            alarms.append(alarm)
            supported_alarms += int(bool(row.get("context_supported", False)))
            first_grasp = row.get("first_grasp_step")
            if first_grasp is None or alarm < int(first_grasp):
                pre_grasp += 1
            else:
                post_grasp += 1
            irreversibility = row.get("irreversibility", {})
            if not bool(irreversibility.get("censored", False)):
                lead = int(irreversibility["step"]) - alarm
                leads.append(lead)
                late += int(lead < 0)
                event_covered += int(lead >= 0)

        candidates.append(
            {
                "quantile": float(quantile),
                "threshold": threshold,
                "expert_point_exceedance_rate": float(1.0 - quantile),
                "episodes": len(rows),
                "observed": len(alarms),
                "observed_rate": float(len(alarms) / len(rows)) if rows else None,
                "supported_observed": supported_alarms,
                "supported_observed_rate": float(supported_alarms / len(rows)) if rows else None,
                "supported_episodes": supported_total,
                "supported_detection_rate": (
                    float(supported_alarms / supported_total) if supported_total else None
                ),
                "mean_alarm_step": float(np.mean(alarms)) if alarms else None,
                "median_alarm_step": float(np.median(alarms)) if alarms else None,
                "p25_alarm_step": float(np.quantile(alarms, 0.25)) if alarms else None,
                "p75_alarm_step": float(np.quantile(alarms, 0.75)) if alarms else None,
                "pre_grasp": pre_grasp,
                "post_grasp": post_grasp,
                "identifiable_events": len(leads),
                "event_covered": event_covered,
                "late_alarms": late,
                "late_rate_among_identifiable": float(late / len(leads)) if leads else None,
                "median_lead_step": float(np.median(leads)) if leads else None,
            }
        )

    distance_matrix = np.asarray(
        [np.asarray(row["distance_at_decision_steps"], dtype=np.float64)[: len(steps)] for row in rows],
        dtype=np.float64,
    )
    time_distribution = [
        {
            "step": int(step),
            "p25": float(np.quantile(distance_matrix[:, index], 0.25)),
            "median": float(np.median(distance_matrix[:, index])),
            "p75": float(np.quantile(distance_matrix[:, index], 0.75)),
            "mean": float(np.mean(distance_matrix[:, index])),
        }
        for index, step in enumerate(steps)
    ]
    return {
        "format": "xvla_erd_threshold_sweep_v1",
        "source_summary": str(summary_path),
        "task": payload.get("task"),
        "horizon": int(horizon),
        "decision_stride": int(stride),
        "persistence_decisions": int(persistence),
        "candidates": candidates,
        "time_distribution": time_distribution,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--persistence", type=int, default=2)
    args = parser.parse_args()

    result = sweep(args.summary, DEFAULT_QUANTILES, args.horizon, args.persistence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = list(result["candidates"][0].keys()) if result["candidates"] else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["candidates"])
    print(json.dumps({"json": str(args.output), "csv": str(csv_path), "task": result["task"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
