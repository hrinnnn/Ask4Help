"""Portable trace IO and temporal detector signals for LIBERO-Plus.

No simulator or model imports live here.  The client stores raw model
features once; all detectors, thresholds, plots, and bootstrap summaries are
subsequently reproducible from these small trace files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FORMAT = "libero_plus_failure_raw_rollout_v1"


def absolute_eef_points(action_chunk: np.ndarray, current_eef_position: np.ndarray) -> np.ndarray:
    actions = np.asarray(action_chunk, dtype=np.float64)
    origin = np.asarray(current_eef_position, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 3 or origin.shape != (3,):
        raise ValueError("expected [horizon, >=3] actions and current [3] EEF position")
    return origin[None, :] + np.cumsum(actions[:, :3], axis=0)


def velocity_normalized_acc(
    previous_points: np.ndarray,
    current_points: np.ndarray,
    *,
    execute_horizon: int,
    min_velocity: float = 1e-6,
    previous_ema: float | None = None,
    ema_alpha: float = 0.9,
) -> tuple[float, float]:
    """Numerically mirrors VLA-FAIL Eq. 7 without a Torch dependency."""
    previous, current = np.asarray(previous_points, dtype=np.float64), np.asarray(current_points, dtype=np.float64)
    if previous.ndim != 2 or previous.shape != current.shape or previous.shape[1] != 3:
        raise ValueError("ACC requires equal [horizon, 3] predictions")
    horizon = previous.shape[0]
    if not 0 < execute_horizon < horizon:
        raise ValueError("execute_horizon must be in [1, horizon - 1]")
    old_suffix = previous[execute_horizon:]
    new_prefix = current[: horizon - execute_horizon]
    velocity = np.maximum(new_prefix.max(axis=0) - new_prefix.min(axis=0), min_velocity)
    raw = float((np.abs(old_suffix - new_prefix) / velocity).mean())
    ema = raw if previous_ema is None else ema_alpha * previous_ema + (1.0 - ema_alpha) * raw
    return raw, float(ema)


def single_sample_overlap(previous_points: np.ndarray, current_points: np.ndarray, *, execute_horizon: int) -> float:
    previous, current = np.asarray(previous_points, dtype=np.float64), np.asarray(current_points, dtype=np.float64)
    if previous.ndim != 2 or previous.shape != current.shape or previous.shape[1] != 3:
        raise ValueError("STAC-Single requires equal [horizon, 3] predictions")
    return float(np.linalg.norm(previous[execute_horizon:] - current[: -execute_horizon], axis=-1).mean())


def write_rollout(output_dir: Path, *, episode: Mapping[str, Any], features: Mapping[str, Sequence[np.ndarray]]) -> None:
    """Atomically persist one JSON timeline and aligned compressed features."""
    output_dir.mkdir(parents=True, exist_ok=False)
    timeline = list(episode.get("timeline", []))
    if not timeline:
        raise ValueError("a raw rollout needs at least one decision point")
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in features.items()}
    expected = len(timeline)
    for key, values in arrays.items():
        if values.shape[0] != expected:
            raise ValueError(f"feature {key} has {values.shape[0]} decisions, expected {expected}")
        if not np.isfinite(values).all():
            raise ValueError(f"feature {key} is non-finite")
    payload = {**dict(episode), "format": FORMAT, "feature_file": "features.npz"}
    temporary = output_dir / "features.partial.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output_dir / "features.npz")
    (output_dir / "episode.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_rollout(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads((path / "episode.json").read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} record: {path}")
    with np.load(path / str(payload["feature_file"])) as handle:
        features = {name: np.asarray(handle[name], dtype=np.float32) for name in handle.files}
    decisions = len(payload.get("timeline", []))
    if decisions < 1 or any(values.shape[0] != decisions for values in features.values()):
        raise ValueError(f"raw feature/timeline alignment failure in {path}")
    return payload, features
