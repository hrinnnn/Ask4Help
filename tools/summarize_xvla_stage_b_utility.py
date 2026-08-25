#!/usr/bin/env python3
"""Aggregate Stage-B ID/OOD policy utility across three training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path, task: str, anchors: list[int], *, threshold: float = 0.95, knee_set: list[int] | None = None) -> dict[str, Any]:
    metric = "success_rate" if task == "stackcube" else "ever_grasped_rate"
    strict_metric = "success_rate" if task == "stackcube" else "strict_success_rate"
    rows = []
    for anchor in anchors:
        id_values, ood_values, strict_ood = [], [], []
        for seed in (17001, 17002, 17003):
            id_summary = read_json(root / task / f"step_{anchor}/seed_{seed}/id/summary.json")
            ood_summary = read_json(root / task / f"step_{anchor}/seed_{seed}/ood/summary.json")
            id_values.append(float(id_summary[metric]))
            ood_values.append(float(ood_summary[metric]))
            strict_ood.append(float(ood_summary[strict_metric]))
        rows.append(
            {
                "anchor": anchor,
                "id_mean": float(np.mean(id_values)),
                "id_std": float(np.std(id_values, ddof=1)),
                "ood_mean": float(np.mean(ood_values)),
                "ood_std": float(np.std(ood_values, ddof=1)),
                "ood_strict_mean": float(np.mean(strict_ood)),
                "seed_values": {"id": id_values, "ood": ood_values, "ood_strict": strict_ood},
            }
        )
    best = max(row["ood_mean"] for row in rows)
    near = [row["anchor"] for row in rows if row["ood_mean"] >= threshold * best]
    knee = [] if knee_set is None else list(knee_set)
    return {
        "format": "xvla_stage_b_timing_utility_summary_v1",
        "task": task,
        "metric": metric,
        "anchors": anchors,
        "relative_threshold": threshold,
        "anchor_summary": rows,
        "best_utility": best,
        "utility_best_anchor_set": near,
        "calibration_knee_set": knee,
        "knee_utility_overlap": sorted(set(near).intersection(knee)),
        "utility_consistent_with_calibration": bool(set(near).intersection(knee)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("stackcube", "airplane"), required=True)
    parser.add_argument("--anchors", type=int, nargs="+", required=True)
    parser.add_argument("--knee-set", type=int, nargs="*", default=[])
    parser.add_argument("--relative-threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root, args.task, args.anchors, threshold=args.relative_threshold, knee_set=args.knee_set)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
