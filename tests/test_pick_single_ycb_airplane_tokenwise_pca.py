from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf"), str(ROOT / "tools")]
from rlinf.algorithms.vla_fail import (  # noqa: E402
    fit_tokenwise_pca_residual_statistics,
    tokenwise_pca_residual_scores,
    tokenwise_pca_z_scores,
    tokenwise_topk_mean,
)
from tools.pick_single_ycb_airplane_tokenwise_pca import (  # noqa: E402
    lerobot_sample_to_policy_observation,
    token_source_masks,
)


def test_independent_token_bases_keep_padding_out_and_standardize_per_position() -> None:
    values = torch.tensor(
        [
            [[-1.0, 0.0], [0.0, -1.0], [5.0, 5.0]],
            [[0.0, 0.0], [0.0, 0.0], [5.0, 5.0]],
            [[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]],
        ]
    )
    valid = torch.tensor([[True, True, False], [True, True, False], [True, True, False]])
    statistics = fit_tokenwise_pca_residual_statistics(values, valid, principal_dim=1, min_observations=2)
    assert statistics.eligible_tokens.tolist() == [True, True, False]
    raw = tokenwise_pca_residual_scores(values[:1], statistics)
    z = tokenwise_pca_z_scores(values[:1], valid[:1], statistics)
    assert raw.shape == z.shape == (1, 3)
    assert torch.isneginf(z[0, 2])
    score, indices = tokenwise_topk_mean(z, k=2)
    assert torch.isfinite(score).all()
    assert set(indices[0].tolist()) == {0, 1}


def test_tokenwise_pca_detects_a_direction_outside_its_own_subspace() -> None:
    train = torch.tensor([[[float(index), 0.0]] for index in range(-3, 4)])
    valid = torch.ones((7, 1), dtype=torch.bool)
    statistics = fit_tokenwise_pca_residual_statistics(train, valid, principal_dim=1, min_observations=2)
    in_space = tokenwise_pca_residual_scores(torch.tensor([[[8.0, 0.0]]]), statistics)
    out_of_space = tokenwise_pca_residual_scores(torch.tensor([[[0.0, 8.0]]]), statistics)
    assert in_space.item() < 1e-4
    assert out_of_space.item() > 1.0
    components = statistics.principal_components[0].to(torch.float64)
    assert torch.allclose(components.transpose(0, 1) @ components, torch.eye(1, dtype=torch.float64), atol=1e-5)


def test_topk_and_modality_masks_do_not_admit_padding_or_wrong_sources() -> None:
    scores = torch.tensor([[0.0, 9.0, float("-inf"), 3.0, 2.0]])
    score, indices = tokenwise_topk_mean(scores, k=3)
    assert torch.isclose(score, torch.tensor([14.0 / 3.0])).all()
    assert indices.tolist() == [[1, 3, 4]]
    masks = token_source_masks(torch.tensor([0, 0, 1, 1, 2]), tokens=5)
    assert masks["base_camera"].tolist() == [True, True, False, False, False]
    assert masks["wrist_camera"].tolist() == [False, False, True, True, False]
    assert masks["language_state"].tolist() == [False, False, False, False, True]


def test_id_lerobot_row_uses_the_same_two_view_policy_contract_as_rollouts() -> None:
    sample = {
        "image": torch.zeros((16, 20, 3), dtype=torch.uint8),
        "wrist_image": torch.zeros((3, 16, 20), dtype=torch.uint8),
        "state": torch.arange(9, dtype=torch.float32),
    }
    observation = lerobot_sample_to_policy_observation(sample, task_description="pick up the toy airplane")
    assert observation["main_images"].shape == (1, 16, 20, 3)
    assert observation["wrist_images"].shape == (1, 16, 20, 3)
    assert observation["states"].shape == (1, 9)
    assert observation["task_ids"].shape == (1,)


def test_posthoc_scanner_uses_ever_grasped_not_distribution_split(tmp_path: Path) -> None:
    module_path = ROOT / "tools" / "sweep_pick_single_ycb_airplane_tokenwise_pca.py"
    spec = importlib.util.spec_from_file_location("airplane_tokenwise_scan", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    episodes = [
        {"episode_index": 0, "seed": 1, "split": "id", "ever_grasped": True, "timeline": [{"scores": {"bridge_pooled_pca": 0.1}}]},
        {"episode_index": 1, "seed": 2, "split": "ood", "ever_grasped": False, "timeline": [{"scores": {"bridge_pooled_pca": 0.9}}]},
    ]
    records = module._records(episodes, "bridge_pooled_pca")
    scan = module._scan(records, "bridge_pooled_pca")
    assert scan["best_balanced_accuracy"]["balanced_accuracy"] == 1.0
    assert scan["pdt_curve"]
    assert scan["aucpdt"] is not None
    assert module._representatives(episodes)["ood_failure"] == 1


def test_asset_manifest_records_all_four_registered_methods(tmp_path: Path) -> None:
    manifest = {
        "format": "pick_single_ycb_airplane_tokenwise_pca_topk16_v1",
        "locations": {"vlm_input": {}, "bridge": {}},
        "source_ids": [0, 1, 2],
    }
    path = tmp_path / "assets_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest["format"].endswith("v1")
