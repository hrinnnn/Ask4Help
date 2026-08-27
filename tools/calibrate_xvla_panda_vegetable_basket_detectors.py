#!/usr/bin/env python3
"""Calibrate fixed detector thresholds from successful ID policy rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--minimum-successes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    summary = json.loads((args.rollouts / "summary.json").read_text(encoding="utf-8"))
    rows = [row for row in summary.get("rows", []) if bool(row.get("strict_success", row.get("success")))]
    if len(rows) < args.minimum_successes:
        raise RuntimeError(
            f"need {args.minimum_successes} successful ID rollouts, got {len(rows)}"
        )
    method_names = sorted(
        {
            name
            for row in rows
            for point in row.get("timeline", [])
            for name, value in point.get("scores", {}).items()
            if value is not None and np.isfinite(float(value))
        }
    )
    thresholds: dict[str, dict] = {}
    for method in method_names:
        maxima = []
        for row in rows:
            values = [
                float(point["scores"][method])
                for point in row.get("timeline", [])
                if point.get("scores", {}).get(method) is not None
                and np.isfinite(float(point["scores"][method]))
            ]
            if values:
                maxima.append(max(values))
        if len(maxima) < args.minimum_successes:
            continue
        rank = min(len(maxima) - 1, max(0, math.ceil((len(maxima) + 1) * args.quantile) - 1))
        ordered = sorted(maxima)
        thresholds[method] = {
            "threshold": float(ordered[rank]),
            "q": args.quantile,
            "successful_trajectory_count": len(maxima),
            "trajectory_maxima_min": float(min(maxima)),
            "trajectory_maxima_max": float(max(maxima)),
        }
    if not thresholds:
        raise RuntimeError("no finite detector thresholds were produced")
    payload = {
        "format": "xvla_panda_vegetable_basket_detector_calibration_v1",
        "calibration_rollouts": str((args.rollouts / "summary.json").resolve()),
        "calibration_split": "successful ID policy rollouts only",
        "quantile": args.quantile,
        "successful_id_rollouts": len(rows),
        "methods": thresholds,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "thresholds.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "CALIBRATION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
