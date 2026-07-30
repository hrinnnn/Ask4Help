#!/usr/bin/env python3
"""Summarize a fixed-threshold StackCube internal-detector comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _metric_row(name: str, id_metrics: dict[str, Any], ood_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector": name,
        "id_success_false_positive_rate": id_metrics["success_false_positive_rate"],
        "id_success_false_positive_rate_95_ci": id_metrics["success_false_positive_rate_95_ci"],
        "id_failure_recall": id_metrics["failure_recall"],
        "id_failure_recall_95_ci": id_metrics["failure_recall_95_ci"],
        "ood_failure_recall": ood_metrics["failure_recall"],
        "ood_failure_recall_95_ci": ood_metrics["failure_recall_95_ci"],
        "id_successes": id_metrics["successes"],
        "id_failures": id_metrics["failures"],
        "ood_failures": ood_metrics["failures"],
    }


def build_summary(id_result: dict[str, Any], ood_result: dict[str, Any]) -> dict[str, Any]:
    """Return a comparable nine-detector summary from passive evaluations."""
    if id_result.get("format") != "stackcube_internal_detector_rollout_v1":
        raise ValueError("ID result has an unexpected format")
    if ood_result.get("format") != "stackcube_internal_detector_rollout_v1":
        raise ValueError("OOD result has an unexpected format")
    id_metrics = id_result["metrics"]
    ood_metrics = ood_result["metrics"]
    if set(id_metrics) != set(ood_metrics):
        raise ValueError("ID and OOD result detector sets differ")
    return {
        "format": "stackcube_internal_detector_summary_v1",
        "id_result": id_result.get("checkpoint"),
        "ood_result": ood_result.get("checkpoint"),
        "detector_assets_sha256": id_result.get("detector_assets_sha256"),
        "rows": [_metric_row(name, id_metrics[name], ood_metrics[name]) for name in sorted(id_metrics)],
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "detector",
        "id_success_false_positive_rate",
        "id_success_false_positive_rate_95_ci",
        "id_failure_recall",
        "id_failure_recall_95_ci",
        "ood_failure_recall",
        "ood_failure_recall_95_ci",
        "id_successes",
        "id_failures",
        "ood_failures",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [row["detector"].replace("__", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, axis = plt.subplots(figsize=(16, 6), constrained_layout=True)
    axis.bar(x - width, [row["id_success_false_positive_rate"] for row in rows], width, label="ID success FPR")
    axis.bar(x, [row["id_failure_recall"] for row in rows], width, label="ID failure recall")
    axis.bar(x + width, [row["ood_failure_recall"] for row in rows], width, label="OOD failure recall")
    axis.set_ylabel("Trajectory rate")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, names, rotation=25, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-result", type=Path, required=True)
    parser.add_argument("--ood-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary directory: {args.output_dir}")
    summary = build_summary(
        json.loads(args.id_result.read_text(encoding="utf-8")),
        json.loads(args.ood_result.read_text(encoding="utf-8")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "results_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(summary["rows"], args.output_dir / "results_summary.csv")
    _plot(summary["rows"], args.output_dir / "detector_comparison.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
