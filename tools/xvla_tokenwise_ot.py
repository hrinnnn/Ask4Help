"""Token-wise PCA and phase-aware Sinkhorn OT for VLA visual tokens.

The module is deliberately model-agnostic. A caller supplies the visual token
tensor captured before pooling, and may fit one asset per coarse execution
phase. The scorer keeps the pooled/global score available as a baseline while
adding local token residuals and a monotonic phase-selection primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _check_features(features: torch.Tensor) -> tuple[int, int, int]:
    if features.ndim != 3:
        raise ValueError("features must have shape [observations, tokens, hidden_dim]")
    if min(features.shape) < 1:
        raise ValueError("features must be non-empty")
    if not torch.isfinite(features).all():
        raise ValueError("features must be finite")
    return tuple(int(value) for value in features.shape)


@dataclass(frozen=True)
class TokenwisePCAOTAsset:
    """Nominal token statistics and a prototype token set for one phase."""

    mean: torch.Tensor
    components: torch.Tensor
    residual_mean: torch.Tensor
    residual_std: torch.Tensor
    feature_scale: torch.Tensor
    prototype: torch.Tensor
    observation_counts: torch.Tensor
    principal_dim: int

    def validate(self) -> None:
        if self.mean.ndim != 2:
            raise ValueError("mean must have shape [tokens, hidden_dim]")
        tokens, hidden_dim = self.mean.shape
        if self.components.shape != (tokens, hidden_dim, self.principal_dim):
            raise ValueError("components shape does not match mean/principal_dim")
        if self.residual_mean.shape != (tokens,) or self.residual_std.shape != (tokens,):
            raise ValueError("residual statistics must have one value per token")
        if self.feature_scale.shape != (hidden_dim,):
            raise ValueError("feature_scale must have shape [hidden_dim]")
        if self.prototype.shape != (tokens, hidden_dim):
            raise ValueError("prototype shape does not match mean")
        if self.observation_counts.shape != (tokens,):
            raise ValueError("observation_counts must have one value per token")
        tensors = (
            self.mean,
            self.components,
            self.residual_mean,
            self.residual_std,
            self.feature_scale,
            self.prototype,
        )
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("tokenwise PCA/OT asset contains non-finite values")
        if self.principal_dim < 1 or self.principal_dim > hidden_dim:
            raise ValueError("principal_dim must lie within hidden_dim")

    def to(self, device: torch.device | str) -> "TokenwisePCAOTAsset":
        return TokenwisePCAOTAsset(
            mean=self.mean.to(device),
            components=self.components.to(device),
            residual_mean=self.residual_mean.to(device),
            residual_std=self.residual_std.to(device),
            feature_scale=self.feature_scale.to(device),
            prototype=self.prototype.to(device),
            observation_counts=self.observation_counts.to(device),
            principal_dim=self.principal_dim,
        )


def _fit_one_phase(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    principal_dim: int,
    min_observations: int,
    epsilon: float,
) -> TokenwisePCAOTAsset:
    observations, tokens, hidden_dim = _check_features(values)
    if mask.shape != (observations, tokens):
        raise ValueError("valid_mask must have shape [observations, tokens]")
    if principal_dim < 1 or principal_dim > hidden_dim:
        raise ValueError("principal_dim must lie within hidden_dim")
    if min_observations < principal_dim + 1:
        raise ValueError("min_observations must exceed principal_dim")
    values = values.float()
    mask = mask.bool()
    counts = mask.sum(dim=0, dtype=torch.int64)
    if torch.any(counts < min_observations):
        raise ValueError("every token needs enough valid observations for its PCA basis")

    mean = torch.zeros(tokens, hidden_dim, dtype=values.dtype, device=values.device)
    components = torch.zeros(tokens, hidden_dim, principal_dim, dtype=values.dtype, device=values.device)
    residual_mean = torch.zeros(tokens, dtype=values.dtype, device=values.device)
    residual_std = torch.zeros(tokens, dtype=values.dtype, device=values.device)

    valid_values = values[mask]
    feature_scale = valid_values.std(dim=0, unbiased=False).clamp_min(epsilon)
    for token in range(tokens):
        token_values = values[mask[:, token], token]
        token_mean = token_values.mean(dim=0)
        centered = token_values - token_mean
        _, eigenvectors = torch.linalg.eigh(
            centered.transpose(0, 1) @ centered / token_values.shape[0]
        )
        basis = eigenvectors[:, -principal_dim:]
        residual = centered - (centered @ basis) @ basis.transpose(0, 1)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        mean[token] = token_mean
        components[token] = basis
        residual_mean[token] = residual_norm.mean()
        residual_std[token] = residual_norm.std(unbiased=False).clamp_min(epsilon)

    asset = TokenwisePCAOTAsset(
        mean=mean,
        components=components,
        residual_mean=residual_mean,
        residual_std=residual_std,
        feature_scale=feature_scale,
        prototype=mean.clone(),
        observation_counts=counts,
        principal_dim=principal_dim,
    )
    asset.validate()
    return asset


def fit_tokenwise_pca_ot(
    features: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    phase_ids: torch.Tensor | None = None,
    principal_dim: int = 8,
    min_observations: int = 2,
    epsilon: float = 1e-6,
) -> list[TokenwisePCAOTAsset]:
    """Fit one token-wise PCA/OT asset per phase, or one global asset.

    ``features`` contains ID expert anchors only. ``phase_ids`` is an integer
    label for each observation; labels are intentionally supplied by the
    caller so this module never invents a task success definition.
    """

    observations, tokens, _hidden_dim = _check_features(features)
    if valid_mask is None:
        valid_mask = torch.ones(observations, tokens, dtype=torch.bool, device=features.device)
    if valid_mask.shape != (observations, tokens):
        raise ValueError("valid_mask must have shape [observations, tokens]")
    if phase_ids is None:
        phase_ids = torch.zeros(observations, dtype=torch.long, device=features.device)
    phase_ids = torch.as_tensor(phase_ids, dtype=torch.long, device=features.device).reshape(-1)
    if phase_ids.shape != (observations,):
        raise ValueError("phase_ids must have one entry per observation")
    phases = torch.unique(phase_ids, sorted=True).tolist()
    assets: list[TokenwisePCAOTAsset] = []
    for phase in phases:
        selected = phase_ids == int(phase)
        assets.append(
            _fit_one_phase(
                features[selected],
                valid_mask[selected],
                principal_dim=principal_dim,
                min_observations=min_observations,
                epsilon=epsilon,
            )
        )
    return assets


def sinkhorn_transport(
    cost: torch.Tensor,
    *,
    source_weights: torch.Tensor | None = None,
    epsilon: float = 0.05,
    iterations: int = 50,
    stabilization: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a uniform-marginal entropic transport plan and its cost.

    ``cost`` has shape ``[batch, query_tokens, key_tokens]``. Both token sets
    receive uniform mass. The implementation is intentionally small and
    deterministic for feature-space smoke tests.
    """

    if cost.ndim != 3:
        raise ValueError("cost must have shape [batch, query_tokens, key_tokens]")
    if not torch.isfinite(cost).all():
        raise ValueError("cost must be finite")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon must be positive and iterations must be positive")
    batch, query_tokens, key_tokens = cost.shape
    kernel = torch.exp(-cost.float() / float(epsilon)).clamp_min(stabilization)
    if source_weights is None:
        source = torch.full(
            (batch, query_tokens), 1.0 / query_tokens, dtype=kernel.dtype, device=kernel.device
        )
    else:
        if source_weights.shape != (batch, query_tokens):
            raise ValueError("source_weights must have shape [batch, query_tokens]")
        source = source_weights.to(device=kernel.device, dtype=kernel.dtype)
        source = source / source.sum(dim=-1, keepdim=True).clamp_min(stabilization)
    target = torch.full(
        (batch, key_tokens), 1.0 / key_tokens, dtype=kernel.dtype, device=kernel.device
    )
    left = torch.ones_like(source)
    right = torch.ones_like(target)
    for _ in range(iterations):
        left = source / (kernel @ right.unsqueeze(-1)).squeeze(-1).clamp_min(stabilization)
        right = target / (kernel.transpose(1, 2) @ left.unsqueeze(-1)).squeeze(-1).clamp_min(stabilization)
    plan = left.unsqueeze(-1) * kernel * right.unsqueeze(1)
    transport_cost = (plan * cost.float()).sum(dim=(1, 2))
    return plan, transport_cost


def tokenwise_pca_z_scores(
    query: torch.Tensor,
    valid_mask: torch.Tensor,
    asset: TokenwisePCAOTAsset,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute position-wise PCA residual z-scores, masking invalid tokens."""

    if query.ndim != 3:
        raise ValueError("query must have shape [batch, tokens, hidden_dim]")
    asset.validate()
    if query.shape[1:] != asset.mean.shape:
        raise ValueError("query token shape does not match asset")
    if valid_mask.shape != query.shape[:2]:
        raise ValueError("valid_mask shape does not match query")
    centered = query.float() - asset.mean.float().unsqueeze(0)
    projected = torch.einsum("btd,tdk->btk", centered, asset.components.float())
    reconstruction = torch.einsum("btk,tdk->btd", projected, asset.components.float())
    residual = torch.linalg.vector_norm(centered - reconstruction, dim=-1)
    z = (residual - asset.residual_mean.float().unsqueeze(0)) / asset.residual_std.float().clamp_min(epsilon).unsqueeze(0)
    return torch.where(valid_mask.bool(), z, torch.full_like(z, -torch.inf))


def token_ot_score(
    query: torch.Tensor,
    valid_mask: torch.Tensor,
    asset: TokenwisePCAOTAsset,
    *,
    epsilon: float = 0.05,
    iterations: int = 50,
    topk: int = 1,
) -> dict[str, torch.Tensor]:
    """Score a query token set against one nominal phase asset."""

    if query.ndim != 3 or query.shape[1:] != asset.prototype.shape:
        raise ValueError("query shape does not match asset token set")
    if valid_mask.shape != query.shape[:2]:
        raise ValueError("valid_mask shape does not match query")
    scale = asset.feature_scale.float().clamp_min(1e-6)
    normalized_query = query.float() / scale
    normalized_key = asset.prototype.float().unsqueeze(0) / scale
    cost = (normalized_query.unsqueeze(2) - normalized_key.unsqueeze(1)).square().mean(dim=-1)
    invalid = ~valid_mask.bool()
    if torch.any(invalid):
        if torch.any(valid_mask.sum(dim=-1) == 0):
            raise ValueError("a query has no valid tokens for OT")
        cost = torch.where(torch.isfinite(cost), cost, torch.zeros_like(cost))
    plan, ot_cost = sinkhorn_transport(
        cost,
        source_weights=valid_mask.to(dtype=cost.dtype),
        epsilon=epsilon,
        iterations=iterations,
    )
    z_scores = tokenwise_pca_z_scores(query, valid_mask, asset)
    valid_count = valid_mask.sum(dim=-1).clamp_min(1)
    requested_topk = min(int(topk), query.shape[1])
    usable_z = torch.where(torch.isfinite(z_scores), z_scores, torch.full_like(z_scores, -1e9))
    pca_topk = usable_z.topk(requested_topk, dim=-1).values.mean(dim=-1)
    row_mass = plan.sum(dim=-1).clamp_min(1e-8)
    aligned_cost = (plan * cost).sum(dim=-1) / row_mass
    aligned_cost = torch.where(valid_mask.bool(), aligned_cost, torch.full_like(aligned_cost, -torch.inf))
    aligned_topk = aligned_cost.topk(requested_topk, dim=-1).values.mean(dim=-1)
    return {
        "ot_cost": ot_cost,
        "pca_topk_z": pca_topk,
        "aligned_topk_cost": aligned_topk,
        "token_pca_z": z_scores,
        "token_ot_cost": aligned_cost,
        "valid_count": valid_count,
    }


def select_monotonic_phase(
    query: torch.Tensor,
    valid_mask: torch.Tensor,
    assets: Sequence[TokenwisePCAOTAsset],
    *,
    previous_phase: int | None = None,
    backtrack: int = 0,
    lookahead: int | None = None,
    epsilon: float = 0.05,
    iterations: int = 50,
) -> tuple[int, dict[str, torch.Tensor]]:
    """Select the lowest-cost phase within a monotonic candidate window."""

    if not assets:
        raise ValueError("at least one phase asset is required")
    if previous_phase is None:
        start = 0
    else:
        start = max(0, int(previous_phase) - int(backtrack))
    stop = len(assets) if lookahead is None else min(len(assets), start + int(lookahead) + 1)
    if start >= stop:
        raise ValueError("phase candidate window is empty")
    candidates = [
        (phase, token_ot_score(query, valid_mask, assets[phase], epsilon=epsilon, iterations=iterations))
        for phase in range(start, stop)
    ]
    phase = min(candidates, key=lambda item: float(item[1]["ot_cost"].mean().item()))[0]
    return phase, dict(candidates[phase][1])
