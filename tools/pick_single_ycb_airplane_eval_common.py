"""Small dependency-free helpers shared by PickSingleYCB airplane evaluators."""

from __future__ import annotations

from typing import Any

import numpy as np


def clip_action_chunk(actions: Any, low: np.ndarray, high: np.ndarray, horizon: int) -> np.ndarray:
    """Validate and bound one pi0.5 action chunk before ManiSkill execution."""

    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape[0] < horizon:
        raise ValueError(f"Expected action chunk [>= {horizon}, action_dim], got {chunk.shape}")
    if chunk.shape[1] != low.size or high.shape != low.shape:
        raise ValueError(f"Action bounds {low.shape} do not match chunk {chunk.shape}")
    return np.clip(chunk[:horizon], low, high).astype(np.float32, copy=False)
