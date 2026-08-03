from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
from pick_single_ycb_airplane_detector_protocol import summary_for_method, threshold_from_success_maxima, union_trace


def test_conformal_threshold_uses_trajectory_maxima():
    assert threshold_from_success_maxima([[1.0, 2.0], [3.0], [2.5]], q=0.95)["threshold"] == 3.0


def test_union_gate_preserves_either_branch_alert():
    assert union_trace([1.0, 5.0], [0.2], final_threshold=10.0, acc_threshold=0.1) == [0.1, 2.0]


def test_failure_metrics_use_final_task_outcome_not_split():
    rows = [
        {"success": True, "execute_horizon": 5, "scores": {"m": [0.1, 0.2]}},
        {"success": False, "execute_horizon": 5, "scores": {"m": [0.1, 0.8]}},
    ]
    summary = summary_for_method(rows, "m", 0.5)
    assert summary["success_conditioned_false_alarm_rate"] == 0.0
    assert summary["failure_recall"] == 1.0
