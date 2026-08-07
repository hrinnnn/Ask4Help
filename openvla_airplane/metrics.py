"""Threshold-free and calibrated trajectory-level failure metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score


def _rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["rows"]


def calibrate_thresholds(id_summary: Path, output: Path, quantile: float = 0.95) -> dict:
    rows = _rows(id_summary)
    methods = sorted({name for row in rows for point in row["timeline"] for name in point.get("scores", {})})
    result = {"quantile": quantile, "calibration_split": "ID success policy rollouts", "methods": {}}
    for method in methods:
        scores = [float(point["scores"][method]) for row in rows for point in row["timeline"] if method in point.get("scores", {})]
        result["methods"][method] = {"threshold": float(np.quantile(scores, quantile)), "scores": {"count": len(scores), "min": min(scores), "max": max(scores), "q95": float(np.quantile(scores, quantile))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


def _pdt_auc(rows: list[dict], method: str) -> float | None:
    failures = [not bool(row["ever_grasped"]) for row in rows]
    if not rows or not any(failures):
        return None
    events = sorted(
        (float(point["scores"][method]), index, time_index)
        for index, row in enumerate(rows)
        for time_index, point in enumerate(row["timeline"])
        if method in point.get("scores", {})
    )[::-1]
    horizons = [max(1, len(row["timeline"])) for row in rows]
    first = [None] * len(rows)
    points = [(0.0, 1.0)]
    for threshold, index, time_index in events:
        if first[index] is None or time_index < first[index]:
            first[index] = time_index
        predicted = [value is not None for value in first]
        tp = sum(predicted[i] and failures[i] for i in range(len(rows)))
        fp = sum(predicted[i] and not failures[i] for i in range(len(rows)))
        precision = tp / max(1, tp + fp)
        pdt = np.mean([1.0 if first[i] is None else first[i] / horizons[i] for i, flag in enumerate(failures) if flag])
        points.append((precision, float(pdt)))
    points.sort()
    area = 0.0
    previous_precision = sum(failures) / len(rows)
    for precision, pdt in points:
        if precision >= previous_precision:
            area += (precision - previous_precision) * pdt
            previous_precision = precision
    return float(area)


def summarize(id_summary: Path, ood_summary: Path, thresholds: Path | None = None) -> dict:
    rows = _rows(id_summary) + _rows(ood_summary)
    methods = sorted({name for row in rows for point in row["timeline"] for name in point.get("scores", {})})
    threshold_payload = json.loads(thresholds.read_text()) if thresholds else None
    table = []
    for method in methods:
        labels = np.asarray([not bool(row["ever_grasped"]) for row in rows], dtype=bool)
        scores = np.asarray([max([float(point["scores"][method]) for point in row["timeline"] if method in point.get("scores", {})], default=float("nan")) for row in rows])
        valid = np.isfinite(scores)
        labels, scores = labels[valid], scores[valid]
        fixed = None
        if threshold_payload and method in threshold_payload["methods"]:
            fixed = float(threshold_payload["methods"][method]["threshold"])
        if fixed is None:
            fixed = float(np.quantile(scores, 0.95))
        predicted = scores >= fixed
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, average="binary", zero_division=0)
        table.append({
            "method": method,
            "auprc": float(average_precision_score(labels, scores)) if labels.any() else None,
            "auroc": float(roc_auc_score(labels, scores)) if labels.any() and (~labels).any() else None,
            "aucpdt": _pdt_auc(rows, method),
            "threshold": fixed,
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_alarm_rate_id": float(np.mean([max([float(point["scores"][method]) for point in row["timeline"] if method in point.get("scores", {})], default=-np.inf) >= fixed for row in _rows(id_summary)])),
            "episodes": int(len(scores)),
        })
    return {"failure_definition": "not ever_grasped", "id_episodes": len(_rows(id_summary)), "ood_episodes": len(_rows(ood_summary)), "methods": table}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate-id", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--id-summary", type=Path)
    parser.add_argument("--ood-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.calibrate_id:
        payload = calibrate_thresholds(args.calibrate_id, args.output)
    else:
        if args.id_summary is None or args.ood_summary is None:
            raise ValueError("summarization requires --id-summary and --ood-summary")
        payload = summarize(args.id_summary, args.ood_summary, args.thresholds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
