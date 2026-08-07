from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from pick_single_ycb_airplane_external_detectors import (  # noqa: E402
    crsail_score,
    fidel_ot_score,
    fit_crsail_bank,
    fit_fidel_memory,
    fit_official_fidel_memory,
    official_fidel_euclidean_score,
)


def test_fidel_ot_prefers_matching_nominal_time() -> None:
    episodes = torch.tensor(
        [
            [[[0.0], [2.0]], [[10.0], [12.0]]],
            [[[0.2], [2.2]], [[10.2], [12.2]]],
        ]
    )
    memory = fit_fidel_memory(episodes, regularization=0.1, sinkhorn_iterations=50)
    score, match = fidel_ot_score(torch.tensor([[[10.1], [12.1]]]), memory)
    far, _ = fidel_ot_score(torch.tensor([[[30.0], [32.0]]]), memory)
    assert match.tolist() == [1]
    assert score.item() < far.item()


def test_fidel_ot_is_patch_permutation_invariant() -> None:
    episodes = torch.tensor(
        [
            [[[0.0], [2.0]]],
            [[[0.1], [2.1]]],
        ]
    )
    memory = fit_fidel_memory(episodes, regularization=0.05, sinkhorn_iterations=60)
    ordered, _ = fidel_ot_score(torch.tensor([[[0.05], [2.05]]]), memory)
    permuted, _ = fidel_ot_score(torch.tensor([[[2.05], [0.05]]]), memory)
    assert torch.allclose(ordered, permuted, atol=1e-4)


def test_official_fidel_euclidean_matches_released_patch_minimum_rule() -> None:
    episodes = torch.tensor(
        [
            [[[0.0], [10.0]], [[30.0], [40.0]]],
            [[[2.0], [12.0]], [[32.0], [42.0]]],
        ]
    )
    memory = fit_official_fidel_memory(episodes)
    score, match = official_fidel_euclidean_score(torch.tensor([[[11.0], [1.0]]]), memory)
    assert match.tolist() == [0]
    assert score.item() == 0.0


def test_official_fidel_adapter_has_parity_with_pinned_repository() -> None:
    official_file = Path(__file__).parents[1] / "external" / "FIDeL" / "src" / "anomaly_detection" / "Representation.py"
    fake_ot = types.ModuleType("ot")
    sys.modules.setdefault("ot", fake_ot)
    spec = importlib.util.spec_from_file_location("fidel_official_representation", official_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = types.SimpleNamespace(
        proj_net=False,
        anomaly_detection=types.SimpleNamespace(distance_type="euclidean"),
    )
    official = module.Memory(2, 2, 2, 1, cfg)
    episodes = torch.tensor(
        [
            [[[0.0], [10.0]], [[30.0], [40.0]]],
            [[[2.0], [12.0]], [[32.0], [42.0]]],
        ]
    )
    adapter = fit_official_fidel_memory(episodes)
    official.memory_avg = adapter.mean
    official.memory_std = torch.ones_like(adapter.mean)
    query = torch.tensor([[11.0], [1.0]])
    official_score, official_index, _ = official.compute_distance(query)
    adapter_score, adapter_index = official_fidel_euclidean_score(query, adapter)
    assert torch.allclose(adapter_score, official_score.reshape(1))
    assert adapter_index.tolist() == [int(official_index)]


def test_crsail_returns_kth_neighbor_distance() -> None:
    bank = fit_crsail_bank(torch.tensor([[0.0], [1.0], [3.0], [8.0]]), k=3)
    assert crsail_score(torch.tensor([[0.0]]), bank).item() == 3.0


def test_crsail_standardization_preserves_training_contract() -> None:
    bank = fit_crsail_bank(torch.tensor([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]]), k=1, standardize=True)
    assert crsail_score(torch.tensor([[1.0, 20.0]]), bank).item() < 1e-6
