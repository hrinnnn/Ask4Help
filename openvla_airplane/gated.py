"""Pure helpers for OpenVLA airplane robot-gated DAgger."""

from __future__ import annotations

import numpy as np


METHODS = ("siglip_pca", "offline_oracle", "failure_recovery", "diffdagger")


def alternating_split(attempt_index: int) -> str:
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    return "id" if attempt_index % 2 == 0 else "ood"


def update_patience_gate(score: float, threshold: float, count: int, patience: int) -> tuple[int, bool]:
    if patience < 1:
        raise ValueError("patience must be positive")
    count = count + 1 if score > threshold else 0
    return count, count >= patience


def patience_episode_score(scores: list[float], patience: int) -> float:
    """Largest threshold for which one complete patience window fires."""
    values = np.asarray(scores, dtype=np.float64)
    if patience < 1:
        raise ValueError("patience must be positive")
    if values.size < patience:
        return float("-inf")
    if not np.isfinite(values).all():
        raise ValueError("gate scores must be finite")
    return float(max(np.min(values[index : index + patience]) for index in range(values.size - patience + 1)))


def calibrate_gate(score_sequences: list[list[float]], quantile: float = 0.95, patience: int = 2) -> dict:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    episode_scores = [patience_episode_score(scores, patience) for scores in score_sequences]
    finite = np.asarray([score for score in episode_scores if np.isfinite(score)], dtype=np.float64)
    if finite.size == 0:
        raise ValueError("no trajectory is long enough to calibrate the gate")
    return {
        "threshold": float(np.quantile(finite, quantile)),
        "quantile": float(quantile),
        "patience": int(patience),
        "episode_scores": episode_scores,
    }


def admitted_expert_suffix(strict_success: bool, expert_start: int | None, action_count: int) -> tuple[int, int] | None:
    if not strict_success or expert_start is None or expert_start >= action_count:
        return None
    return expert_start, action_count
