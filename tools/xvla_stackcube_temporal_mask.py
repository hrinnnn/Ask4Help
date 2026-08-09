"""Temporal padding and masked loss shared by X-VLA StackCube training."""

from __future__ import annotations

import numpy as np
import torch


def padded_action_chunk(
    actions: np.ndarray, anchor: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed-horizon chunk and validity mask for every real anchor."""
    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or len(array) == 0:
        raise ValueError("actions must be a non-empty [T, D] array")
    if not 0 <= anchor < len(array):
        raise IndexError(anchor)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    real = array[anchor : anchor + horizon]
    valid = len(real)
    chunk = np.repeat(array[-1][None], horizon, axis=0)
    chunk[:valid] = real
    mask = np.arange(horizon) < valid
    return chunk, mask


def masked_action_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    temporal_mask: torch.Tensor | None,
    *,
    real_dim: int,
) -> torch.Tensor:
    """Mean squared error over real dimensions and valid temporal positions."""
    element_loss = (pred[..., :real_dim] - target[..., :real_dim]).square()
    if temporal_mask is None:
        return element_loss.mean()
    mask = temporal_mask.to(device=element_loss.device, dtype=torch.bool)
    if mask.shape != element_loss.shape[:2]:
        raise ValueError(
            f"temporal mask {tuple(mask.shape)} does not match actions "
            f"{tuple(element_loss.shape[:2])}"
        )
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every anchor must contain at least one real action target")
    weights = mask.unsqueeze(-1).to(element_loss.dtype)
    return (element_loss * weights).sum() / (weights.sum() * real_dim)
