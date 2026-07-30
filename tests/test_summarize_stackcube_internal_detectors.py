from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "summarize_stackcube_internal_detectors.py"
SPEC = importlib.util.spec_from_file_location("internal_detector_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _result(metric: dict[str, object]) -> dict[str, object]:
    return {"format": "stackcube_internal_detector_rollout_v1", "checkpoint": "step7000", "detector_assets_sha256": "sha", "metrics": {"bridge__knn": metric}}


def test_summary_preserves_the_three_comparable_rates() -> None:
    id_metric = {
        "success_false_positive_rate": 0.0,
        "success_false_positive_rate_95_ci": [0.0, 0.1],
        "failure_recall": 1.0,
        "failure_recall_95_ci": [0.6, 1.0],
        "successes": 42,
        "failures": 8,
    }
    ood_metric = {**id_metric, "failure_recall": 0.98, "failure_recall_95_ci": [0.89, 1.0], "successes": 0, "failures": 50}
    summary = MODULE.build_summary(_result(id_metric), _result(ood_metric))
    assert summary["rows"] == [{
        "detector": "bridge__knn",
        "id_success_false_positive_rate": 0.0,
        "id_success_false_positive_rate_95_ci": [0.0, 0.1],
        "id_failure_recall": 1.0,
        "id_failure_recall_95_ci": [0.6, 1.0],
        "ood_failure_recall": 0.98,
        "ood_failure_recall_95_ci": [0.89, 1.0],
        "id_successes": 42,
        "id_failures": 8,
        "ood_failures": 50,
    }]
