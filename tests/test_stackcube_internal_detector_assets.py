from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "tools" / "build_stackcube_internal_detector_assets.py"
SPEC = importlib.util.spec_from_file_location("internal_detector_assets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_builder_uses_the_three_preregistered_representation_locations() -> None:
    cache = {
        "format": "stackcube_multilayer_llmd_feature_cache_v1",
        "statistics_sha256": "stats",
        "feature_cache_path": "cache.pt",
        "feature_cache_sha256": "cache-sha",
        "checkpoint": "checkpoint",
        "dataset_root": "dataset",
        "indices": [0, 1, 2, 3],
        "layers": {
            "vlm_block_08_mean": torch.randn(4, 1, 4),
            "vlm_bridge_final_mean": torch.randn(4, 1, 4),
            "action_expert_block_13": torch.randn(4, 2, 4),
        },
    }
    statistics = {
        "format": "stackcube_multilayer_llmd_statistics_v1",
        "statistics_sha256": "stats",
        "statistics_path": "multilayer.pt",
    }

    payload = MODULE.build_detector_payload(
        feature_cache=cache, multilayer_statistics=statistics, knn_k=2
    )

    assert payload["candidate_layers"] == list(MODULE.CANDIDATE_LAYERS)
    assert set(payload["detectors"]) == {
        "vlm_block_08_mean__knn_k2",
        "vlm_block_08_mean__pca_residual",
        "vlm_bridge_final_mean__knn_k2",
        "vlm_bridge_final_mean__pca_residual",
        "action_expert_block_13__knn_k2",
        "action_expert_block_13__pca_residual",
    }
