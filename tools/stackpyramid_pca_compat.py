"""Self-contained PCA residual statistics for the StackPyramid detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class PCAResidualStatistics:
    mean: torch.Tensor
    principal_components: torch.Tensor
    principal_dim: int
    num_observations: int

    def validate(self) -> None:
        if self.mean.ndim != 2 or self.principal_components.ndim != 3:
            raise ValueError("invalid PCA residual rank")
        tokens, features = self.mean.shape
        if self.principal_components.shape[:2] != (tokens, features):
            raise ValueError("incompatible PCA components")
        if self.principal_components.shape[2] != self.principal_dim:
            raise ValueError("incompatible PCA dimension")
        if not 0 < self.principal_dim <= features or self.num_observations < 2:
            raise ValueError("invalid PCA dimensions")

    def state_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "mean": self.mean.cpu(),
            "principal_components": self.principal_components.cpu(),
            "principal_dim": self.principal_dim,
            "num_observations": self.num_observations,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> "PCAResidualStatistics":
        result = cls(
            mean=torch.as_tensor(payload["mean"], dtype=torch.float32),
            principal_components=torch.as_tensor(payload["principal_components"], dtype=torch.float32),
            principal_dim=int(payload["principal_dim"]),
            num_observations=int(payload["num_observations"]),
        )
        result.validate()
        return result


def vim_default_principal_dim(hidden_dim: int) -> int:
    if hidden_dim < 2:
        raise ValueError("PCA residual needs at least two hidden dimensions")
    if hidden_dim >= 2048:
        return 1000
    if hidden_dim >= 768:
        return 512
    return hidden_dim // 2


def fit_pca_residual_statistics(features: torch.Tensor, *, principal_dim: int | None = None) -> PCAResidualStatistics:
    if features.ndim != 3:
        raise ValueError("features must have shape [observations, tokens, features]")
    observations, tokens, hidden_dim = features.shape
    if observations < 2 or tokens < 1 or hidden_dim < 2 or not torch.isfinite(features).all():
        raise ValueError("invalid PCA feature bank")
    dimension = vim_default_principal_dim(hidden_dim) if principal_dim is None else principal_dim
    values = features.detach().to(device="cpu", dtype=torch.float64)
    mean = values.mean(dim=0)
    centered = values - mean.unsqueeze(0)
    covariance = torch.einsum("nth,ntk->thk", centered, centered) / observations
    components = []
    for token in range(tokens):
        _eigenvalues, eigenvectors = torch.linalg.eigh(covariance[token])
        components.append(eigenvectors[:, -dimension:])
    result = PCAResidualStatistics(mean.float(), torch.stack(components).float(), dimension, observations)
    result.validate()
    return result


def pca_residual_score(features: torch.Tensor, statistics: PCAResidualStatistics) -> torch.Tensor:
    statistics.validate()
    if features.ndim != 3 or tuple(features.shape[1:]) != tuple(statistics.mean.shape):
        raise ValueError("feature shape does not match PCA statistics")
    values = features.float()
    centered = values - statistics.mean.to(values.device).unsqueeze(0)
    components = statistics.principal_components.to(values.device)
    coordinates = torch.einsum("bth,thr->btr", centered, components)
    reconstruction = torch.einsum("btr,thr->bth", coordinates, components)
    scores = torch.linalg.vector_norm(centered - reconstruction, dim=-1)
    if not torch.isfinite(scores).all():
        raise RuntimeError("non-finite PCA residual")
    return scores.amax(dim=-1)
