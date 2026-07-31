from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "RLinf"))
MODULE_PATH = ROOT / "tools" / "libero_plus_failure_assets.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_failure_assets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _cache() -> dict[str, object]:
    generator = torch.Generator().manual_seed(3)
    return {
        "format": "libero_plus_expert_feature_cache_v1",
        "selection_sha256": "selection",
        "selected_anchors": [{"index": index} for index in range(12)],
        "features": {
            "bridge": torch.randn(12, 1, 4, generator=generator),
            "action_expert_final": torch.randn(12, 10, 4, generator=generator),
        },
    }


def test_reference_assets_score_all_preregistered_methods() -> None:
    cache = _cache()
    assets = MODULE.fit_reference_assets(cache, knn_k=3)
    scores = MODULE.score_features(
        {
            "bridge": cache["features"]["bridge"][0],
            "action_expert_final": cache["features"]["action_expert_final"][0],
        },
        assets,
    )

    assert set(scores) == set(MODULE.DETECTOR_NAMES)
    assert all(value >= 0.0 for value in scores.values())
    assert assets["num_reference_anchors"] == 12


def test_feature_cache_rejects_unaligned_anchor_metadata() -> None:
    cache = _cache()
    cache["selected_anchors"] = [{"index": 0}]
    with pytest.raises(ValueError, match="selected_anchors"):
        MODULE.fit_reference_assets(cache, knn_k=3)


def test_conformal_thresholds_use_finite_sample_order_statistic() -> None:
    result = MODULE.conformal_thresholds({"bridge_llmd": [[0.1], [0.2], [0.3], [0.4], [0.5]]}, delta=0.05)
    value = result["thresholds"]["bridge_llmd"]
    assert value["threshold"] == pytest.approx(0.5)
    assert value["order_statistic_rank"] == 5
