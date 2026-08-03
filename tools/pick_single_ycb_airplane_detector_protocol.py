"""Pure scoring protocol for the controlled toy-airplane detector benchmark."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from libero_plus_failure_protocol import first_alert_index


BASE_METHODS = (
    "bridge_deep_knn", "bridge_llmd", "bridge_pca_residual", "final_llmd", "acc", "stac_single",
)
ALL_METHODS = BASE_METHODS + ("final_llmd_or_acc",)


def _balanced_accuracy(labels: Sequence[bool], alerts: Sequence[bool]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return None
    tp = sum(label and alert for label, alert in zip(labels, alerts))
    tn = sum((not label) and (not alert) for label, alert in zip(labels, alerts))
    return (tp / positive + tn / negative) / 2.0


def _roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positive, negative = sum(labels), len(labels) - sum(labels)
    if positive == 0 or negative == 0:
        return None
    wins = 0.0
    for score, label in zip(scores, labels):
        if not label:
            continue
        for other, other_label in zip(scores, labels):
            if other_label:
                continue
            wins += 1.0 if score > other else 0.5 if score == other else 0.0
    return wins / (positive * negative)


def _average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positive = sum(labels)
    if positive == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hits = 0; total = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            total += hits / rank
    return total / positive


def threshold_from_success_maxima(traces: Sequence[Sequence[float]], *, q: float = 0.95) -> dict[str, Any]:
    """Conformal q-th quantile of complete successful-policy trace maxima."""

    maxima = sorted(max(float(value) for value in trace) for trace in traces if trace)
    if not maxima or not 0 < q < 1:
        raise ValueError("need non-empty finite traces and 0 < q < 1")
    if not np.isfinite(maxima).all():
        raise ValueError("calibration scores must be finite")
    rank = min(len(maxima), math.ceil((len(maxima) + 1) * q))
    return {"threshold": float(maxima[rank - 1]), "q": q, "order_statistic_rank": rank,
            "trajectory_maxima": {"count": len(maxima), "min": maxima[0], "max": maxima[-1]}}


def union_trace(final_llmd: Sequence[float], acc: Sequence[float], *, final_threshold: float, acc_threshold: float) -> list[float]:
    """Normalized OR gate: an alert occurs exactly when either branch alerts."""

    if final_threshold <= 0 or acc_threshold <= 0:
        raise ValueError("OR-gate thresholds must be positive")
    result = []
    for index, value in enumerate(final_llmd):
        acc_value = float(acc[index - 1]) if index > 0 and index - 1 < len(acc) else 0.0
        result.append(max(float(value) / final_threshold, acc_value / acc_threshold))
    return result


def _episode_label(row: Mapping[str, Any]) -> bool:
    return not bool(row["success"])


def summary_for_method(episodes: Sequence[Mapping[str, Any]], method: str, threshold: float) -> dict[str, Any]:
    """Trajectory-level failure metrics, preserving decision-index lead time."""

    scored = [row for row in episodes if row.get("scores", {}).get(method)]
    if not scored:
        return {"episodes": 0}
    labels = [_episode_label(row) for row in scored]
    maxima = [max(float(value) for value in row["scores"][method]) for row in scored]
    alerted = [first_alert_index(row["scores"][method], threshold) is not None for row in scored]
    tp = sum(label and alarm for label, alarm in zip(labels, alerted))
    fp = sum((not label) and alarm for label, alarm in zip(labels, alerted))
    tn = sum((not label) and (not alarm) for label, alarm in zip(labels, alerted))
    fn = sum(label and (not alarm) for label, alarm in zip(labels, alerted))
    precision = None if tp + fp == 0 else tp / (tp + fp)
    recall = None if tp + fn == 0 else tp / (tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    lead = []
    for row, label in zip(scored, labels):
        if not label:
            continue
        index = first_alert_index(row["scores"][method], threshold)
        if index is not None:
            lead.append((len(row["scores"][method]) - 1 - index) * int(row.get("execute_horizon", 5)))
    return {
        "episodes": len(scored), "successes": labels.count(False), "failures": labels.count(True),
        "success_conditioned_false_alarm_rate": None if labels.count(False) == 0 else fp / labels.count(False),
        "failure_recall": recall, "precision": precision, "f1": f1,
        "balanced_accuracy": _balanced_accuracy(labels, alerted), "auroc": _roc_auc(labels, maxima),
        "auprc": _average_precision(labels, maxima), "first_alert_lead_low_level_steps": {
            "count": len(lead), "mean": None if not lead else float(np.mean(lead)),
            "median": None if not lead else float(np.median(lead)),
        },
    }
