from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure_protocol.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_failure_protocol", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROTOCOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTOCOL)


def _expert(task: str, episode: str, anchors: int) -> dict[str, object]:
    return {"task_id": task, "episode_id": episode, "anchor_ids": list(range(anchors))}


def test_expert_selection_is_task_balanced_reproducible_and_evenly_spaced() -> None:
    experts = [
        _expert(task, f"{task}-{episode}", 19)
        for task in ("task-0", "task-1")
        for episode in range(12)
    ]
    first = PROTOCOL.select_expert_anchors(experts, demos_per_task=10, anchors_per_demo=10, seed=17)
    second = PROTOCOL.select_expert_anchors(experts, demos_per_task=10, anchors_per_demo=10, seed=17)

    assert first == second
    assert len(first) == 200
    assert {entry["task_id"] for entry in first} == {"task-0", "task-1"}
    for task in ("task-0", "task-1"):
        selected_episodes = {entry["episode_id"] for entry in first if entry["task_id"] == task}
        assert len(selected_episodes) == 10
    assert [entry["anchor_id"] for entry in first if entry["episode_id"] == "task-0-0"] == sorted(
        entry["anchor_id"] for entry in first if entry["episode_id"] == "task-0-0"
    )


def test_expert_selection_rejects_short_or_unbalanced_inputs() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        PROTOCOL.select_expert_anchors([_expert("task", "only", 10)], demos_per_task=2, anchors_per_demo=1, seed=0)
    with pytest.raises(ValueError, match="fewer than"):
        PROTOCOL.select_expert_anchors([_expert("task", "only", 2)] * 2, demos_per_task=2, anchors_per_demo=3, seed=0)


def test_first_alert_and_fixed_threshold_metrics_are_trajectory_level() -> None:
    episodes = [
        {"episode_id": "success", "success": True, "scores": [0.1, 0.2, 0.3]},
        {"episode_id": "failure-early", "success": False, "scores": [0.8, 0.1, 0.1]},
        {"episode_id": "failure-missed", "success": False, "scores": [0.1, 0.2, 0.3]},
    ]
    records = PROTOCOL.evaluate_fixed_threshold(episodes, threshold=0.5)

    assert records[0]["first_alert_index"] is None
    assert records[1]["first_alert_index"] == 0
    assert records[2]["first_alert_index"] is None
    metrics = PROTOCOL.fixed_threshold_metrics(records)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 1
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["mean_normalized_detection_time"] == pytest.approx(0.5)


def test_auc_metrics_are_perfect_for_separated_trajectory_scores() -> None:
    episodes = [
        {"episode_id": "s0", "success": True, "scores": [0.1, 0.2]},
        {"episode_id": "s1", "success": True, "scores": [0.1, 0.3]},
        {"episode_id": "f0", "success": False, "scores": [0.9, 0.8]},
        {"episode_id": "f1", "success": False, "scores": [0.7, 0.6]},
    ]
    metrics = PROTOCOL.threshold_independent_metrics(episodes)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert 0.0 <= metrics["aucpdt"] <= 1.0


def test_aucpdt_penalizes_missed_failure_and_uses_episode_horizon() -> None:
    episodes = [
        {"episode_id": "success", "success": True, "scores": [0.1, 0.1, 0.1, 0.1]},
        {"episode_id": "late", "success": False, "scores": [0.0, 0.0, 0.0, 0.9]},
        {"episode_id": "missed", "success": False, "scores": [0.0, 0.0, 0.0, 0.2]},
    ]
    # The no-alert threshold can be Pareto dominated and therefore absent from
    # the plotted front, but it must still count as a full-horizon miss.
    missed = PROTOCOL.evaluate_fixed_threshold(episodes, threshold=1.0)
    missed_metrics = PROTOCOL.fixed_threshold_metrics(missed)
    assert missed_metrics["mean_normalized_detection_time"] == pytest.approx(1.0)

    late = PROTOCOL.evaluate_fixed_threshold(episodes, threshold=0.8)
    late_metrics = PROTOCOL.fixed_threshold_metrics(late)
    assert late_metrics["mean_normalized_detection_time"] == pytest.approx(0.875)


def test_bootstrap_confidence_interval_is_deterministic() -> None:
    values = [{"success": index % 2 == 0, "scores": [float(index)]} for index in range(12)]
    first = PROTOCOL.bootstrap_interval(values, lambda rows: float(sum(row["success"] for row in rows) / len(rows)), seed=3, samples=100)
    second = PROTOCOL.bootstrap_interval(values, lambda rows: float(sum(row["success"] for row in rows) / len(rows)), seed=3, samples=100)
    assert first == second
    assert first[0] <= 0.5 <= first[1]


def test_absolute_eef_and_single_sample_overlap_are_well_defined() -> None:
    actions = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    points = PROTOCOL.action_chunk_to_absolute_eef(actions, np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(points[-1], [11.0, 22.0, 33.0])
    # After executing one action, the following plan should begin at the
    # previous plan's second absolute point for a perfectly consistent overlap.
    next_chunk = np.vstack([points[1:], points[-1] + np.array([0.0, 0.0, 1.0])])
    assert PROTOCOL.single_sample_overlap_score(points, next_chunk, execute_horizon=1) == pytest.approx(0.0)
