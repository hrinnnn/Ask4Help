"""External visual/state baselines for the airplane failure benchmark.

The module deliberately contains no simulator or Hugging Face model loading.
It implements the score definitions used by the asset builder and rollout
replayer so the numerical core can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


FIDEL_FORMAT = "pick_single_ycb_airplane_fidel_dinov2_ot_v1"
CRSAIL_FORMAT = "pick_single_ycb_airplane_crsail_k5_v1"


@dataclass(frozen=True)
class OfficialFIDeLMemory:
    """Released FIDeL ``Representation`` memory and distance settings.

    The official repository defaults to ResNet-18 features and its
    ``euclidean`` distance branch. That branch compares the query embedding to
    each time-indexed expert mean and takes the minimum. Keeping this baseline
    separate avoids silently mixing it with the paper's DINOv2/Sinkhorn form.
    """

    mean: torch.Tensor

    def validate(self) -> None:
        if self.mean.ndim != 3 or min(self.mean.shape) < 1:
            raise ValueError("official FIDeL memory must have [time,patch,feature] shape")
        if not torch.isfinite(self.mean).all():
            raise ValueError("official FIDeL memory must be finite")

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {"mean": self.mean}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "OfficialFIDeLMemory":
        result = cls(mean=torch.as_tensor(payload["mean"]))
        result.validate()
        return result


def fit_official_fidel_memory(features: torch.Tensor) -> OfficialFIDeLMemory:
    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 4 or values.shape[0] < 1:
        raise ValueError("official FIDeL fitting needs [episode,time,patch,feature]")
    return OfficialFIDeLMemory(mean=values.mean(dim=0))


def official_fidel_euclidean_score(
    query: torch.Tensor, memory: OfficialFIDeLMemory
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact released ``distance_type=euclidean`` Representation score."""

    memory.validate()
    values = torch.as_tensor(query, device=memory.mean.device, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[-1] != memory.mean.shape[-1]:
        raise ValueError("official FIDeL query must have [batch,patch,feature]")
    scores, matches = [], []
    for item in values:
        per_time = []
        for key in memory.mean:
            matrix = torch.cdist(item, key)
            per_time.append(matrix.min(dim=1, keepdim=True).values.sum())
        distances = torch.stack(per_time)
        score, index = distances.min(dim=0)
        scores.append(score)
        matches.append(index)
    return torch.stack(scores), torch.stack(matches)


@dataclass(frozen=True)
class FIDeLMemory:
    """Time-indexed patch Gaussian memory from successful demonstrations."""

    mean: torch.Tensor
    std: torch.Tensor
    regularization: float = 0.05
    sinkhorn_iterations: int = 30
    epsilon: float = 1e-5

    def validate(self) -> None:
        if self.mean.ndim != 3 or self.std.shape != self.mean.shape:
            raise ValueError("FIDeL mean/std must have [time, patch, feature] shape")
        if self.mean.shape[0] < 1 or self.mean.shape[1] < 1 or self.mean.shape[2] < 1:
            raise ValueError("FIDeL memory cannot be empty")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("FIDeL memory must be finite")
        if self.regularization <= 0 or self.sinkhorn_iterations < 1 or self.epsilon <= 0:
            raise ValueError("invalid FIDeL Sinkhorn configuration")

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "mean": self.mean,
            "std": self.std,
            "regularization": self.regularization,
            "sinkhorn_iterations": self.sinkhorn_iterations,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "FIDeLMemory":
        result = cls(
            mean=torch.as_tensor(payload["mean"]),
            std=torch.as_tensor(payload["std"]),
            regularization=float(payload.get("regularization", 0.05)),
            sinkhorn_iterations=int(payload.get("sinkhorn_iterations", 30)),
            epsilon=float(payload.get("epsilon", 1e-5)),
        )
        result.validate()
        return result


def fit_fidel_memory(features: torch.Tensor, *, regularization: float = 0.05,
                     sinkhorn_iterations: int = 30, epsilon: float = 1e-5) -> FIDeLMemory:
    """Fit the paper's per-time/per-patch Gaussian memory.

    ``features`` is a dense tensor of shape [episode, time, patch, feature].
    Callers are responsible for applying FIDeL's common-horizon truncation.
    """

    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 4 or values.shape[0] < 2:
        raise ValueError("FIDeL fitting needs at least two [episode,time,patch,feature] episodes")
    return FIDeLMemory(
        mean=values.mean(dim=0),
        std=values.std(dim=0, unbiased=False),
        regularization=regularization,
        sinkhorn_iterations=sinkhorn_iterations,
        epsilon=epsilon,
    )


def _sinkhorn_cost(cost: torch.Tensor, *, regularization: float, iterations: int) -> torch.Tensor:
    """Entropic OT cost for a batch of uniform patch distributions."""

    if cost.ndim != 3 or cost.shape[-2] < 1 or cost.shape[-1] < 1:
        raise ValueError("cost must have [batch, query_patch, memory_patch] shape")
    batch, query_patches, memory_patches = cost.shape
    dtype, device = cost.dtype, cost.device
    log_a = torch.full((batch, query_patches), -torch.log(torch.tensor(float(query_patches), device=device, dtype=dtype)), device=device, dtype=dtype)
    log_b = torch.full((batch, memory_patches), -torch.log(torch.tensor(float(memory_patches), device=device, dtype=dtype)), device=device, dtype=dtype)
    log_kernel = -cost / regularization
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
    plan = torch.exp(log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2))
    return (plan * cost).sum(dim=(-2, -1))


def fidel_ot_score(query: torch.Tensor, memory: FIDeLMemory) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FIDeL's minimum normalized patch OT score and matching time."""

    memory.validate()
    values = torch.as_tensor(query, device=memory.mean.device, dtype=torch.float32)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or tuple(values.shape[1:]) != tuple(memory.mean.shape[1:]):
        raise ValueError("FIDeL query must have [batch, patch, feature] matching the memory")
    results = []
    matches = []
    scale = memory.std.clamp_min(memory.epsilon)
    for item in values:
        # [time, query_patch, memory_patch, feature]. The cost follows the
        # paper's standard-deviation-normalized Euclidean ground metric.
        delta = item.unsqueeze(0).unsqueeze(2) - memory.mean.unsqueeze(1)
        normalized = delta / scale.unsqueeze(1)
        cost = torch.linalg.vector_norm(normalized, dim=-1)
        per_time = _sinkhorn_cost(
            cost,
            regularization=memory.regularization,
            iterations=memory.sinkhorn_iterations,
        )
        score, index = per_time.min(dim=0)
        results.append(score)
        matches.append(index)
    return torch.stack(results), torch.stack(matches)


@dataclass(frozen=True)
class CRSAILBank:
    """Reference bank for the CRSAIL K-th-nearest-state novelty rule."""

    values: torch.Tensor
    k: int = 5
    center: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    def validate(self) -> None:
        if self.values.ndim != 2 or self.values.shape[0] < self.k or self.k < 1:
            raise ValueError("CRSAIL bank needs [reference,feature] with reference >= k")
        if not torch.isfinite(self.values).all():
            raise ValueError("CRSAIL bank must be finite")
        if (self.center is None) != (self.scale is None):
            raise ValueError("CRSAIL center and scale must be supplied together")
        if self.center is not None:
            if self.center.shape != (self.values.shape[1],) or self.scale.shape != self.center.shape:
                raise ValueError("CRSAIL normalization shape mismatch")
            if torch.any(self.scale <= 0):
                raise ValueError("CRSAIL scale must be positive")

    def state_dict(self) -> dict[str, Any]:
        self.validate()
        return {"values": self.values, "k": self.k, "center": self.center, "scale": self.scale}

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> "CRSAILBank":
        center = payload.get("center")
        scale = payload.get("scale")
        result = cls(
            values=torch.as_tensor(payload["values"]),
            k=int(payload.get("k", 5)),
            center=None if center is None else torch.as_tensor(center),
            scale=None if scale is None else torch.as_tensor(scale),
        )
        result.validate()
        return result


def fit_crsail_bank(values: torch.Tensor, *, k: int = 5, standardize: bool = False) -> CRSAILBank:
    rows = torch.as_tensor(values, dtype=torch.float32)
    if rows.ndim != 2 or rows.shape[0] < k:
        raise ValueError("CRSAIL fitting needs a non-empty two-dimensional bank with rows >= k")
    if not standardize:
        return CRSAILBank(values=rows, k=k)
    center = rows.mean(dim=0)
    scale = rows.std(dim=0, unbiased=False).clamp_min(1e-6)
    return CRSAILBank(values=(rows - center) / scale, k=k, center=center, scale=scale)


def crsail_score(query: torch.Tensor, bank: CRSAILBank, *, cosine: bool = False) -> torch.Tensor:
    """Distance to the K-th nearest reference, as defined by CRSAIL."""

    bank.validate()
    values = torch.as_tensor(query, device=bank.values.device, dtype=torch.float32)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != bank.values.shape[1]:
        raise ValueError("CRSAIL query shape does not match its reference bank")
    if bank.center is not None:
        values = (values - bank.center) / bank.scale
    if cosine:
        distances = 1.0 - F.normalize(values, dim=-1) @ F.normalize(bank.values, dim=-1).T
    else:
        distances = torch.cdist(values, bank.values)
    return distances.kthvalue(bank.k, dim=-1).values
