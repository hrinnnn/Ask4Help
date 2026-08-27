#!/usr/bin/env python3
"""Compute threshold-free and frozen-threshold detector metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.libero_plus_failure_protocol import (
    aucpdt,
    average_precision,
    evaluate_fixed_threshold,
    fixed_threshold_metrics,
    roc_auc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-rollouts", type=Path, required=True)
    parser.add_argument("--ood-rollouts", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"no rollout rows in {path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    rows = read_rows(args.id_rollouts) + read_rows(args.ood_rollouts)
    calibration = json.loads((args.thresholds / "thresholds.json").read_text(encoding="utf-8"))
    threshold_specs = calibration.get("methods", {})
    methods = sorted(
        {
            name
            for row in rows
            for point in row.get("timeline", [])
            for name, value in point.get("scores", {}).items()
            if value is not None
        }
        & set(threshold_specs)
    )
    if not methods:
        raise RuntimeError("no common detector score and threshold")

    result: dict[str, dict] = {}
    for method in methods:
        episodes = []
        for index, row in enumerate(rows):
            scores = [
                float(point["scores"][method])
                for point in row.get("timeline", [])
                if point.get("scores", {}).get(method) is not None
            ]
            if not scores or not np.isfinite(scores).all():
                continue
            episodes.append(
                {
                    "episode_id": str(row.get("episode_index", index)),
                    "split": row.get("split"),
                    "success": bool(row.get("strict_success", row.get("success", False))),
                    "scores": scores,
                    "execute_horizon": int(row.get("execute_horizon", 5)),
                }
            )
        labels = [not bool(item["success"]) for item in episodes]
        maxima = [max(item["scores"]) for item in episodes]
        threshold = float(threshold_specs[method]["threshold"])
        frozen = evaluate_fixed_threshold(episodes, threshold=threshold)
        fixed = fixed_threshold_metrics(frozen)
        failures = [item for item in frozen if not item["success"]]
        leads = [
            (len(item["scores"]) - 1 - int(item["first_alert_index"])) * int(item["execute_horizon"])
            for item in failures
            if item["first_alert_index"] is not None
        ]
        result[method] = {
            "episodes": len(episodes),
            "successes": int(sum(not label for label in labels)),
            "failures": int(sum(labels)),
            "threshold": threshold,
            "q": threshold_specs[method].get("q", calibration.get("quantile", 0.95)),
            "auprc": average_precision(labels, maxima),
            "auroc": roc_auc(labels, maxima),
            "aucpdt": aucpdt(episodes),
            "balanced_accuracy": fixed["balanced_accuracy"],
            "precision": fixed["precision"],
            "recall": fixed["recall"],
            "f1": fixed["f1"],
            "success_conditioned_false_alarm_rate": (
                None
                if fixed["tn"] + fixed["fp"] == 0
                else fixed["fp"] / (fixed["tn"] + fixed["fp"])
            ),
            "true_positive": fixed["tp"],
            "false_positive": fixed["fp"],
            "true_negative": fixed["tn"],
            "false_negative": fixed["fn"],
            "first_alarm_lead_low_level_steps": {
                "count": len(leads),
                "mean": None if not leads else float(np.mean(leads)),
                "median": None if not leads else float(np.median(leads)),
            },
            "trajectory_max_score_min": float(min(maxima)),
            "trajectory_max_score_max": float(max(maxima)),
        }

    policy = {}
    for name, path in (("id", args.id_rollouts), ("ood", args.ood_rollouts)):
        payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        policy[name] = {
            "episodes": payload.get("episodes"),
            "strict_successes": payload.get("strict_successes", payload.get("successes")),
            "ever_grasped_successes": payload.get("ever_grasped_successes"),
        }
    payload = {
        "format": "xvla_panda_vegetable_basket_failure_metrics_v1",
        "failure_definition": "not strict_success",
        "rollout_roots": {"id": str(args.id_rollouts.resolve()), "ood": str(args.ood_rollouts.resolve())},
        "thresholds": str(args.thresholds.resolve()),
        "policy_outcomes": policy,
        "methods": result,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["method"] + [
        "auprc", "auroc", "aucpdt", "balanced_accuracy", "precision", "recall", "f1",
        "success_conditioned_false_alarm_rate", "threshold", "first_alarm_lead_mean",
    ]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, values in result.items():
            writer.writerow({
                "method": name,
                **{key: values.get(key) for key in fieldnames if key != "method" and key != "first_alarm_lead_mean"},
                "first_alarm_lead_mean": values["first_alarm_lead_low_level_steps"]["mean"],
            })
    lines = [
        "# Panda Eggplant Failure Detection",
        "",
        "Failure label: `not strict_success`; thresholds are frozen from successful ID calibration only.",
        "",
        "| Method | AUPRC | AUROC | AUCPDT | BA | Precision | Recall | F1 | ID false alarm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in result.items():
        lines.append(
            "| {name} | {auprc:.4f} | {auroc:.4f} | {aucpdt:.4f} | {balanced_accuracy:.4f} | "
            "{precision:.4f} | {recall:.4f} | {f1:.4f} | {far:.4f} |".format(
                name=name,
                auprc=values["auprc"],
                auroc=values["auroc"],
                aucpdt=values["aucpdt"],
                balanced_accuracy=values["balanced_accuracy"],
                precision=values["precision"],
                recall=values["recall"],
                f1=values["f1"],
                far=values["success_conditioned_false_alarm_rate"],
            )
        )
    (args.output_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "METRICS_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
