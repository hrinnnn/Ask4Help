from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "RLinf"))
SPEC = importlib.util.spec_from_file_location("libero_plus_failure_scoring", ROOT / "tools" / "libero_plus_failure" / "score_passive_rollouts.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _assets() -> dict:
    cache = {
        "format": "libero_plus_expert_feature_cache_v1",
        "selection_sha256": "selected",
        "selected_anchors": [{"index": index} for index in range(12)],
        "features": {"bridge": torch.randn(12, 1, 4), "action_expert_final": torch.randn(12, 10, 4)},
    }
    return MODULE.ASSETS.fit_reference_assets(cache, knn_k=3)


def _record(index: int, success: bool) -> dict:
    return {
        "episode_id": str(index), "success": success, "episode_path": str(index), "suite": "libero_10", "source": "clean",
        "category": None, "configuration_id": None, "task_index": index % 10, "seed": index,
        "video_path": "video.mp4", "decision_steps": [0, 5], "runtime_ms": {"policy": [], "feature": []},
        "scores": {
            "bridge_llmd": [0.1 + index, 0.2 + index], "bridge_deep_knn": [0.1, 0.2],
            "bridge_pca_residual": [0.1, 0.2], "final_llmd": [0.1, 0.2], "acc": [0.1], "stac_single": [0.1],
        },
    }


def test_calibration_and_union_annotation_preserve_vla_fail_or_gate() -> None:
    assets = _assets()
    records = [_record(index, True) for index in range(5)]
    thresholds = MODULE.calibrate(records, "asset", required_successes=5)
    annotated = MODULE.annotate([_record(8, False), *records], thresholds)
    assert "vla_fail_final_or_acc" in annotated[0]["scores"]
    assert annotated[0]["first_alert"]["vla_fail_final_or_acc"] is not None
    summary = MODULE.summarize(annotated, thresholds)
    assert set(summary["all"]) == set(MODULE.ALL_METHODS)
    assert summary["all"]["bridge_llmd"]["aucpr"] == summary["all"]["bridge_llmd"]["average_precision"]
    assert summary["runtime_ms"]["decision_points"] == 0


def test_action_variance_is_only_calibrated_when_every_success_has_c10_trace() -> None:
    assets = _assets()
    records = [_record(index, True) for index in range(5)]
    records[0]["scores"]["action_total_variance"] = [0.2]
    thresholds = MODULE.calibrate(records, "asset", required_successes=5)
    assert "action_total_variance" not in thresholds["thresholds"]
    for record in records:
        record["scores"]["action_total_variance"] = [0.2]
    thresholds = MODULE.calibrate(records, "asset", required_successes=5)
    assert "action_total_variance" in thresholds["thresholds"]


def test_reference_asset_loading_falls_back_for_legacy_torch(monkeypatch) -> None:
    calls: list[dict] = []
    expected = {"asset": "loaded"}

    def legacy_load(*_args, **kwargs):
        calls.append(dict(kwargs))
        if "weights_only" in kwargs:
            raise TypeError("'weights_only' is an invalid keyword argument")
        return expected

    monkeypatch.setattr(MODULE.torch, "load", legacy_load)
    assert MODULE.load_reference_assets(Path("reference.pt")) == expected
    assert calls == [{"map_location": "cpu", "weights_only": False}, {"map_location": "cpu"}]


def test_joint_bootstrap_matches_individual_metric_bootstraps() -> None:
    rows = MODULE._protocol_records([_record(index, index % 2 == 0) for index in range(8)], "bridge_llmd", 3.5)
    threshold = 3.5
    intervals = MODULE.bootstrap_metric_intervals(rows, threshold=threshold, seed=9, samples=25)
    for key in MODULE.BOOTSTRAP_METRICS:
        def metric(sample, key=key):
            fixed = MODULE.PROTOCOL.fixed_threshold_metrics(
                MODULE.PROTOCOL.evaluate_fixed_threshold(sample, threshold=threshold)
            )
            labels = [not bool(record["success"]) for record in sample]
            scores = [float(max(record["scores"])) for record in sample]
            independent = {
                "roc_auc": MODULE.PROTOCOL.roc_auc(labels, scores),
                "average_precision": MODULE.PROTOCOL.average_precision(labels, scores),
            }
            value = ({**fixed, **independent})[key]
            return float("nan") if value is None else float(value)

        expected = MODULE.PROTOCOL.bootstrap_interval(rows, metric, seed=9, samples=25)
        assert intervals[key] == pytest.approx(expected)
