from __future__ import annotations

import torch

from tools.xvla_tokenwise_ot import (
    fit_tokenwise_pca_ot,
    select_monotonic_multiview_phase,
    select_monotonic_phase,
    token_ot_score,
)


def _phase_features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    phase0 = torch.randn(8, 4, 4, generator=generator) * 0.02
    phase1 = torch.randn(8, 4, 4, generator=generator) * 0.02 + 4.0
    features = torch.cat([phase0, phase1], dim=0)
    mask = torch.ones(16, 4, dtype=torch.bool)
    phases = torch.tensor([0] * 8 + [1] * 8)
    return features, mask, phases


def test_tokenwise_pca_ot_fits_phases_and_scores_finitely() -> None:
    features, mask, phases = _phase_features()
    assets = fit_tokenwise_pca_ot(
        features,
        mask,
        phase_ids=phases,
        principal_dim=2,
        min_observations=4,
    )
    result = token_ot_score(features[:1], mask[:1], assets[0], topk=2)
    assert len(assets) == 2
    assert torch.isfinite(result["ot_cost"]).all()
    assert torch.isfinite(result["aligned_topk_cost"]).all()


def test_ot_ignores_invalid_query_tokens() -> None:
    features, mask, phases = _phase_features()
    assets = fit_tokenwise_pca_ot(features, mask, phase_ids=phases, principal_dim=2, min_observations=4)
    query = features[:1].clone()
    valid = mask[:1].clone()
    valid[:, -1] = False
    query[:, -1] = 10_000.0
    result = token_ot_score(query, valid, assets[0], topk=2)
    reference_query = features[:1].clone()
    reference_query[:, -1] = -10_000.0
    reference = token_ot_score(reference_query, valid, assets[0], topk=2)
    torch.testing.assert_close(result["ot_cost"], reference["ot_cost"], atol=1e-4, rtol=1e-4)


def test_ot_is_permutation_tolerant() -> None:
    features, mask, phases = _phase_features()
    assets = fit_tokenwise_pca_ot(features, mask, phase_ids=phases, principal_dim=2, min_observations=4)
    permutation = torch.tensor([2, 0, 3, 1])
    result = token_ot_score(features[:1], mask[:1], assets[0], topk=2)
    permuted = token_ot_score(features[:1, permutation], mask[:1, permutation], assets[0], topk=2)
    torch.testing.assert_close(result["ot_cost"], permuted["ot_cost"], atol=2e-3, rtol=2e-3)


def test_phase_selection_can_move_forward_but_not_jump_back() -> None:
    features, mask, phases = _phase_features()
    assets = fit_tokenwise_pca_ot(features, mask, phase_ids=phases, principal_dim=2, min_observations=4)
    phase, _scores = select_monotonic_phase(features[9:10], mask[9:10], assets, previous_phase=0)
    assert phase == 1
    phase, _scores = select_monotonic_phase(features[0:1], mask[0:1], assets, previous_phase=1, backtrack=0)
    assert phase == 1


def test_multiview_phase_selection_aggregates_views() -> None:
    features, mask, phases = _phase_features()
    assets = fit_tokenwise_pca_ot(features, mask, phase_ids=phases, principal_dim=2, min_observations=4)
    phase, scores = select_monotonic_multiview_phase(
        [features[9:10], features[9:10] + 0.01],
        [mask[9:10], mask[9:10]],
        [assets, assets],
        topk=2,
    )
    assert phase == 1
    assert torch.isfinite(scores["ot_cost"]).all()
