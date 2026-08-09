#!/usr/bin/env python3
"""Calibrate and summarize the X-VLA 100-ID/100-OOD failure benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from pick_single_ycb_airplane_detector_protocol import (  # noqa: E402
    summary_for_method,
    threshold_free_summary,
    threshold_from_success_maxima,
    union_trace,
)
from xvla_airplane_failure_detection import trajectory_score_rows  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    return trajectory_score_rows(json.loads(path.read_text())["rows"])


def method_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({name for row in rows for name in row.get("scores", {})})


def policy_success(row: dict[str, Any]) -> bool:
    """Use the benchmark's registered task outcome without split labels."""

    if "ever_grasped" in row:
        return bool(row["ever_grasped"])
    if "success" in row:
        return bool(row["success"])
    raise KeyError("rollout row has neither ever_grasped nor success")


def calibrate(summary: Path, output: Path, q: float) -> dict[str, Any]:
    rows = [row for row in load_rows(summary) if policy_success(row)]
    if not rows:
        raise ValueError("calibration needs successful ID policy rollouts")
    methods: dict[str, Any] = {}
    for name in method_names(rows):
        traces = [row["scores"][name] for row in rows if row["scores"].get(name)]
        methods[name] = threshold_from_success_maxima(traces, q=q)
    payload = {
        "format": "xvla_airplane_failure_calibration_v1",
        "source": str(summary.resolve()),
        "success_definition": (
            "ever_grasped" if all("ever_grasped" in row for row in rows) else "task_success"
        ),
        "successful_id_trajectories": len(rows),
        "q": q,
        "methods": methods,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def add_union(rows: list[dict[str, Any]], thresholds: dict[str, Any]) -> None:
    final_name = "action_block_24_llmd"
    if final_name not in thresholds["methods"] or "acc" not in thresholds["methods"]:
        return
    final_threshold = float(thresholds["methods"][final_name]["threshold"])
    acc_threshold = float(thresholds["methods"]["acc"]["threshold"])
    for row in rows:
        final_trace = row["scores"].get(final_name, [])
        acc_trace = row["scores"].get("acc", [])
        if final_trace:
            row["scores"]["final_llmd_or_acc"] = union_trace(
                final_trace,
                acc_trace,
                final_threshold=final_threshold,
                acc_threshold=acc_threshold,
            )
    thresholds["methods"]["final_llmd_or_acc"] = {
        "threshold": 1.0,
        "q": thresholds["q"],
        "definition": "normalized logical OR of action_block_24_llmd and ACC",
    }


def summarize(
    id_summary: Path, ood_summary: Path, calibration_path: Path, output_dir: Path
) -> dict[str, Any]:
    rows = load_rows(id_summary) + load_rows(ood_summary)
    airplane = all("ever_grasped" in row for row in rows)
    calibration = json.loads(calibration_path.read_text())
    add_union(rows, calibration)
    table = []
    for name in method_names(rows):
        threshold_info = calibration["methods"].get(name)
        threshold = None if threshold_info is None else float(threshold_info["threshold"])
        free = threshold_free_summary(rows, name)
        fixed = {"episodes": 0} if threshold is None else summary_for_method(rows, name, threshold)
        table.append(
            {
                "method": name,
                "auprc": free.get("auprc"),
                "auroc": free.get("auroc"),
                "aucpdt_lower_is_better": free.get("aucpdt"),
                "threshold": threshold,
                "balanced_accuracy": fixed.get("balanced_accuracy"),
                "precision": fixed.get("precision"),
                "failure_recall": fixed.get("failure_recall"),
                "f1": fixed.get("f1"),
                "success_false_alarm_rate": fixed.get("success_conditioned_false_alarm_rate"),
                "episodes": free.get("episodes"),
                "successes": free.get("successes"),
                "failures": free.get("failures"),
            }
        )
    table.sort(key=lambda row: (-1 if row["auprc"] is None else -row["auprc"], row["method"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": (
            "xvla_airplane_failure_metrics_v1"
            if airplane
            else "xvla_stackcube_failure_metrics_v1"
        ),
        "failure_definition": "not ever_grasped" if airplane else "not task_success",
        "trajectory_score": "maximum over decision scores",
        "id_summary": str(id_summary.resolve()),
        "ood_summary": str(ood_summary.resolve()),
        "calibration": str(calibration_path.resolve()),
        "methods": table,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        "| Method | AUPRC | AUROC | AUCPDT (lower) | Bal. Acc. | Recall | F1 | False alarm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        def percent(key: str) -> str:
            value = row[key]
            return "-" if value is None else f"{100 * value:.2f}%"

        aucpdt_value = row["aucpdt_lower_is_better"]
        aucpdt_text = "-" if aucpdt_value is None else f"{aucpdt_value:.4f}"
        lines.append(
            f"| {row['method']} | {percent('auprc')} | {percent('auroc')} | "
            f"{aucpdt_text} | {percent('balanced_accuracy')} | "
            f"{percent('failure_recall')} | {percent('f1')} | "
            f"{percent('success_false_alarm_rate')} |"
        )
    (output_dir / "metrics.md").write_text("\n".join(lines) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", type=Path)
    parser.add_argument("--q", type=float, default=0.95)
    parser.add_argument("--id-summary", type=Path)
    parser.add_argument("--ood-summary", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calibrate:
        payload = calibrate(args.calibrate, args.output, args.q)
    else:
        if args.id_summary is None or args.ood_summary is None or args.calibration is None:
            raise ValueError("summary mode requires ID, OOD, and calibration inputs")
        payload = summarize(args.id_summary, args.ood_summary, args.calibration, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
