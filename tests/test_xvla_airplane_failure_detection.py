from __future__ import annotations

from pathlib import Path

import torch

from tools.xvla_airplane_failure_detection import (
    XVLAMultilayerScorer,
    first_tensor,
    fit_layer_asset,
    layer_names,
    trajectory_score_rows,
)


def test_first_tensor_accepts_transformer_tuple() -> None:
    value = torch.ones(2, 3, 4)
    assert first_tensor((value, None)) is value


def test_layer_names_cover_every_vlm_bridge_and_action_block() -> None:
    assert layer_names(2, 3) == [
        "vlm_encoder_01",
        "vlm_encoder_02",
        "vlm_action_bridge",
        "action_block_01",
        "action_block_02",
        "action_block_03",
    ]


def test_all_three_scores_rank_an_outlier_above_training_point(tmp_path: Path) -> None:
    torch.manual_seed(0)
    values = torch.randn(64, 8) * 0.1
    asset = fit_layer_asset(values, pca_dim=4)
    path = tmp_path / "assets.pt"
    torch.save(
        {
            "format": "xvla_airplane_multilayer_detector_assets_v1",
            "layers": {"vlm_action_bridge": asset},
        },
        path,
    )
    scorer = XVLAMultilayerScorer(path, device="cpu", knn_k=10)
    nominal = scorer.score({"vlm_action_bridge": values[:1]})
    outlier = scorer.score({"vlm_action_bridge": torch.full((1, 8), 10.0)})
    for suffix in ("pca", "llmd", "knn"):
        assert outlier[f"vlm_action_bridge_{suffix}"] > nominal[f"vlm_action_bridge_{suffix}"]


def test_timeline_conversion_drops_missing_initial_acc_only() -> None:
    rows = trajectory_score_rows(
        [
            {
                "timeline": [
                    {"scores": {"pca": 1.0, "acc": None}},
                    {"scores": {"pca": 2.0, "acc": 0.5}},
                ]
            }
        ]
    )
    assert rows[0]["scores"]["pca"] == [1.0, 2.0]
    assert rows[0]["scores"]["acc"] == [0.5]
