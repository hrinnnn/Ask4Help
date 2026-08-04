#!/usr/bin/env python3
"""Pure, reproducible evaluation protocol for LIBERO-Plus failure detection.

The policy/environment adapter writes compact trajectory records.  This module
contains every data-only decision in the leaderboard: balanced expert-anchor
selection, trajectory-level scores, conformal operating points, and metrics.
Keeping it free of JAX, MuJoCo, and model code makes the published results
auditable and testable without a GPU.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _stable_rank(seed: int, task_id: str, episode_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task_id}:{episode_id}".encode("utf-8")).hexdigest()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    """Choose ``count`` distinct anchors including both trajectory endpoints."""

    if length < count or count < 1:
        raise ValueError(f"need at least {count} anchors, got {length}")
    if count == 1:
        return [0]
    # Integer arithmetic avoids platform-dependent floating rounding.
    return [(index * (length - 1)) // (count - 1) for index in range(count)]


def select_expert_anchors(
    experts: Sequence[Mapping[str, Any]], *, demos_per_task: int = 10, anchors_per_demo: int = 10, seed: int = 0
) -> list[dict[str, Any]]:
    """Deterministically select a task-balanced light expert feature bank.

    Every input expert record needs ``task_id``, ``episode_id``, and ordered
    ``anchor_ids``.  The latter are decision-anchor IDs rather than raw frames,
    so callers cannot accidentally fit statistics on a non-replanning state.
    """

    if demos_per_task < 1 or anchors_per_demo < 1:
        raise ValueError("demos_per_task and anchors_per_demo must be positive")
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for record in experts:
        task_id, episode_id = str(record["task_id"]), str(record["episode_id"])
        anchors = list(record["anchor_ids"])
        if len(anchors) < anchors_per_demo:
            raise ValueError(f"expert {task_id}/{episode_id} has fewer than {anchors_per_demo} anchors")
        by_task.setdefault(task_id, []).append(record)
    if not by_task:
        raise ValueError("expert selection requires at least one task")

    selected: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        candidates = by_task[task_id]
        unique = {str(record["episode_id"]) for record in candidates}
        if len(unique) < demos_per_task:
            raise ValueError(f"task {task_id} has fewer than {demos_per_task} distinct expert demos")
        chosen = sorted(
            candidates,
            key=lambda record: _stable_rank(seed, task_id, str(record["episode_id"])),
        )[:demos_per_task]
        if len({str(record["episode_id"]) for record in chosen}) != demos_per_task:
            raise ValueError(f"task {task_id} includes duplicate expert episode IDs")
        for record in sorted(chosen, key=lambda item: str(item["episode_id"])):
            anchor_ids = list(record["anchor_ids"])
            for position in evenly_spaced_indices(len(anchor_ids), anchors_per_demo):
                selected.append(
                    {
                        "task_id": task_id,
                        "episode_id": str(record["episode_id"]),
                        "anchor_id": anchor_ids[position],
                    }
                )
    return selected


def libero_plus_base_task_name(task_name: str) -> str:
    """Remove the official perturbation suffix without guessing task identity.

    LIBERO-Plus records Camera/Robot variants as ``..._view_*`` and layout
    variants as ``..._add_*``.  The stable prefix is the only legal key for
    pairing a Plus configuration with its clean LIBERO-10 task.
    """

    for marker in ("_view_", "_add_"):
        if marker in task_name:
            return task_name.split(marker, 1)[0]
    return task_name


def build_libero_plus_manifest(
    *,
    classifications: Sequence[Mapping[str, Any]],
    clean_tasks: Sequence[Mapping[str, Any]],
    categories: Sequence[str] = ("Camera Viewpoints", "Robot Initial States", "Objects Layout"),
    min_difficulty_level: int | None = None,
    max_difficulty_level: int | None = None,
) -> list[dict[str, Any]]:
    """Pair every official selected perturbation with exactly one clean task.

    The caller supplies clean task rows with a canonical ``name`` and a
    zero-based ``task_index``.  Ambiguous or unmatched official variants are
    rejected rather than silently paired to the wrong task.
    """

    if min_difficulty_level is not None and min_difficulty_level < 1:
        raise ValueError("min_difficulty_level must be positive")
    if max_difficulty_level is not None and max_difficulty_level < 1:
        raise ValueError("max_difficulty_level must be positive")
    if (
        min_difficulty_level is not None
        and max_difficulty_level is not None
        and min_difficulty_level > max_difficulty_level
    ):
        raise ValueError("min_difficulty_level cannot exceed max_difficulty_level")

    allowed = set(categories)
    by_name: dict[str, Mapping[str, Any]] = {}
    for clean in clean_tasks:
        name = libero_plus_base_task_name(str(clean["name"]))
        if name in by_name:
            raise ValueError(f"duplicate clean LIBERO task name: {name}")
        by_name[name] = clean
    manifest: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in classifications:
        if str(row.get("category")) not in allowed:
            continue
        difficulty_level = int(row["difficulty_level"])
        if min_difficulty_level is not None and difficulty_level < min_difficulty_level:
            continue
        if max_difficulty_level is not None and difficulty_level > max_difficulty_level:
            continue
        plus_id = int(row["id"])
        if plus_id in seen_ids:
            raise ValueError(f"duplicate official LIBERO-Plus task id: {plus_id}")
        seen_ids.add(plus_id)
        plus_name = str(row["name"])
        # The official layout variants use more than one suffix convention
        # (for example ``_add_*`` and ``_level*_sample*``).  Match the full
        # clean task prefix instead of maintaining an incomplete suffix list.
        matches = [
            (name, clean)
            for name, clean in by_name.items()
            if plus_name == name or plus_name.startswith(name + "_")
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely pair LIBERO-Plus task {plus_id}: {plus_name}")
        _base_name, clean = matches[0]
        manifest.append(
            {
                "plus_task_id": plus_id,
                "plus_task_index": plus_id - 1,
                "plus_task_name": str(row["name"]),
                "category": str(row["category"]),
                "difficulty_level": difficulty_level,
                "clean_task_index": int(clean["task_index"]),
                "clean_task_name": str(clean["name"]),
            }
        )
    if not manifest:
        raise ValueError("no official LIBERO-Plus tasks match the requested categories")
    return sorted(manifest, key=lambda row: (row["category"], row["plus_task_id"]))


def action_chunk_to_absolute_eef(action_chunk: np.ndarray, current_eef_position: np.ndarray) -> np.ndarray:
    """Convert LIBERO delta-position actions into chunk-wise absolute EEF points."""

    actions = np.asarray(action_chunk, dtype=np.float64)
    origin = np.asarray(current_eef_position, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 3 or origin.shape != (3,):
        raise ValueError("expected [horizon, >=3] actions and a three-dimensional EEF position")
    return origin[None, :] + np.cumsum(actions[:, :3], axis=0)


def single_sample_overlap_score(previous_points: np.ndarray, current_points: np.ndarray, *, execute_horizon: int) -> float:
    """STAC-Single: mean L2 mismatch of native single-sample chunk overlap."""

    previous, current = np.asarray(previous_points, dtype=np.float64), np.asarray(current_points, dtype=np.float64)
    if previous.ndim != 2 or previous.shape != current.shape or previous.shape[1] != 3:
        raise ValueError("overlap points must be equal [horizon, 3] arrays")
    horizon = previous.shape[0]
    if not 0 < execute_horizon < horizon:
        raise ValueError("execute_horizon must be in [1, horizon - 1]")
    return float(np.linalg.norm(previous[execute_horizon:] - current[: horizon - execute_horizon], axis=-1).mean())


def first_alert_index(scores: Sequence[float], threshold: float) -> int | None:
    for index, value in enumerate(scores):
        if not math.isfinite(float(value)):
            raise ValueError("detector score must be finite")
        if float(value) >= threshold:
            return index
    return None


def _validate_episodes(episodes: Sequence[Mapping[str, Any]]) -> None:
    if not episodes:
        raise ValueError("at least one episode is required")
    for episode in episodes:
        if not list(episode["scores"]):
            raise ValueError(f"episode {episode.get('episode_id', '<unknown>')} has no score")


def evaluate_fixed_threshold(episodes: Sequence[Mapping[str, Any]], *, threshold: float) -> list[dict[str, Any]]:
    _validate_episodes(episodes)
    result: list[dict[str, Any]] = []
    for episode in episodes:
        scores = [float(value) for value in episode["scores"]]
        result.append(
            {
                **dict(episode),
                "trajectory_score": max(scores),
                "first_alert_index": first_alert_index(scores, threshold),
            }
        )
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def fixed_threshold_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    _validate_episodes(records)
    tp = fp = tn = fn = 0
    normalized_times: list[float] = []
    twa_terms: list[float] = []
    for record in records:
        failed = not bool(record["success"])
        alert = record["first_alert_index"] is not None
        if failed and alert:
            tp += 1
            horizon = len(record["scores"])
            time = int(record["first_alert_index"]) / horizon
            normalized_times.append(time)
            twa_terms.append(1.0 - time)
        elif failed:
            fn += 1
            normalized_times.append(1.0)
            twa_terms.append(0.0)
        elif alert:
            fp += 1
        else:
            tn += 1
            twa_terms.append(1.0)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    tnr = _ratio(tn, tn + fp)
    tpr = recall
    bacc = None if tpr is None or tnr is None else (tpr + tnr) / 2.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": None if tnr is None else 1.0 - tnr,
        "f1": None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall),
        "balanced_accuracy": bacc,
        "weighted_accuracy": (tp + tn) / len(records),
        "mean_normalized_detection_time": None if not normalized_times else float(np.mean(normalized_times)),
        "twa": None if not twa_terms else float(np.mean(twa_terms)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def roc_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    values = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(labels, dtype=bool)
    pos, neg = int(positives.sum()), int((~positives).sum())
    if pos == 0 or neg == 0:
        return None
    rank_sum = _rankdata(values)[positives].sum()
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    positives = np.asarray(labels, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    total = int(positives.sum())
    if total == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    result = 0.0
    hits = 0
    for rank, index in enumerate(order, start=1):
        if positives[index]:
            hits += 1
            result += hits / rank
    return float(result / total)


def _threshold_points(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Evaluate every score threshold relevant to first-alert timing.

    Trajectory maxima suffice for ordinary trajectory classification, but not
    for AUCPDT: a non-maximum score may yield the same alerted trajectories as
    a larger threshold while triggering a failed rollout earlier.  VLA-FAIL's
    PDT curve therefore has to retain every observed timestep score.
    """

    values = sorted({float(score) for episode in episodes for score in episode["scores"]})
    sentinel = np.nextafter(values[-1], np.inf)
    return [_threshold_point(episodes, threshold) for threshold in [*values, float(sentinel)]]


def _threshold_point(episodes: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, float]:
    records = evaluate_fixed_threshold(episodes, threshold=threshold)
    metrics = fixed_threshold_metrics(records)
    failures = [record for record in records if not bool(record["success"])]
    predicted_failure = int(metrics["tp"]) + int(metrics["fp"])
    recall = 0.0 if not failures else int(metrics["tp"]) / len(failures)
    precision = 1.0 if predicted_failure == 0 else int(metrics["tp"]) / predicted_failure
    pdt = 1.0 if not failures else float(np.mean([
        1.0 if record["first_alert_index"] is None else int(record["first_alert_index"]) / len(record["scores"])
        for record in failures
    ]))
    return {"threshold": threshold, "precision": precision, "recall": recall, "pdt": pdt}


def penalized_detection_time_curve(episodes: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """VLA-FAIL's precision/PDT Pareto front, including the no-alert point."""

    _validate_episodes(episodes)
    points = _threshold_points(episodes)
    pareto: list[dict[str, float]] = []
    for point in points:
        dominated = any(
            other["precision"] >= point["precision"]
            and other["pdt"] <= point["pdt"]
            and (other["precision"] > point["precision"] or other["pdt"] < point["pdt"])
            for other in points
        )
        if not dominated:
            pareto.append(point)
    return sorted(pareto, key=lambda point: (point["precision"], point["pdt"], point["threshold"]))


def aucpdt(episodes: Sequence[Mapping[str, Any]]) -> float | None:
    """Right-Riemann AUCPDT from VLA-FAIL Eq. (9); lower is better."""

    _validate_episodes(episodes)
    failure_rate = sum(not bool(episode["success"]) for episode in episodes) / len(episodes)
    curve = penalized_detection_time_curve(episodes)
    if not curve:
        return None
    # The paper integrates only precision >= the failure-rate baseline.
    points = [point for point in curve if point["precision"] >= failure_rate]
    if not points:
        return 1.0 - failure_rate
    area = 0.0
    previous_precision = failure_rate
    for point in points:
        precision = max(previous_precision, point["precision"])
        area += (precision - previous_precision) * point["pdt"]
        previous_precision = precision
    return float(area)


def threshold_independent_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    _validate_episodes(episodes)
    labels = [not bool(episode["success"]) for episode in episodes]
    scores = [float(max(episode["scores"])) for episode in episodes]
    return {
        "roc_auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "aucpdt": aucpdt(episodes),
    }


def bootstrap_interval(
    episodes: Sequence[Mapping[str, Any]], metric: Callable[[Sequence[Mapping[str, Any]]], float], *, seed: int, samples: int = 1000
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    _validate_episodes(episodes)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(episodes), size=len(episodes))
        value = metric([episodes[int(index)] for index in indices])
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("bootstrap metric was non-finite for every resample")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
