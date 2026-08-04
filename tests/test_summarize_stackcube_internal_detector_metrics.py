from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_stackcube_internal_detector_metrics",
    ROOT / "tools" / "summarize_stackcube_internal_detector_metrics.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_summary_preserves_fixed_and_temporal_metrics() -> None:
    detector = "vlm_bridge_final_mean__knn_k10"
    id_payload = {
        "metrics": {detector: {}},
        "checkpoint": "checkpoint",
        "detector_assets_sha256": "assets",
    }
    ood_payload = dict(id_payload)
    thresholds = {
        "attempts": 2,
        "detector_assets_sha256": "assets",
        "detectors": {detector: {"threshold": 0.5}},
    }
    calibration = {"episodes": [{"success": True}, {"success": False}]}
    records = [
        {"episode_id": "id", "source": "id", "success": True, "scores": {detector: [0.1, 0.2]}},
        {"episode_id": "ood", "source": "ood", "success": False, "scores": {detector: [0.8, 0.7]}},
    ]

    summary = MODULE.build_summary(
        id_payload=id_payload,
        ood_payload=ood_payload,
        threshold_payload=thresholds,
        calibration_payload=calibration,
        id_records=records[:1],
        ood_records=records[1:],
        input_sha256={},
    )

    result = summary["overall"][detector]
    assert result["fixed_threshold"]["tp"] == 1
    assert result["fixed_threshold"]["tn"] == 1
    assert result["threshold_independent"]["roc_auc"] == 1.0
    assert result["threshold_independent"]["aucpr"] == 1.0
