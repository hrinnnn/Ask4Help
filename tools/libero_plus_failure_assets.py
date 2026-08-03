#!/usr/bin/env python3
"""Persistent no-training detector assets for the LIBERO-Plus main table.

The input cache is deliberately small and fixed by the protocol: 1,000
task-balanced successful expert anchors.  This module never looks at rollout
success or failure labels; it only fits the ID reference distributions that
the passive evaluator later queries.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
_VLA_FAIL_PATH = ROOT / "RLinf" / "rlinf" / "algorithms" / "vla_fail.py"
_VLA_FAIL_SPEC = importlib.util.spec_from_file_location("ask4help_vla_fail_core", _VLA_FAIL_PATH)
if _VLA_FAIL_SPEC is None or _VLA_FAIL_SPEC.loader is None:
    raise ImportError(f"cannot load standalone VLA-FAIL core from {_VLA_FAIL_PATH}")
_VLA_FAIL = importlib.util.module_from_spec(_VLA_FAIL_SPEC)
sys.modules[_VLA_FAIL_SPEC.name] = _VLA_FAIL
_VLA_FAIL_SPEC.loader.exec_module(_VLA_FAIL)

KNNStatistics = _VLA_FAIL.KNNStatistics
LLMDStatistics = _VLA_FAIL.LLMDStatistics
PCAResidualStatistics = _VLA_FAIL.PCAResidualStatistics
constant_split_conformal_threshold = _VLA_FAIL.constant_split_conformal_threshold
fit_knn_statistics = _VLA_FAIL.fit_knn_statistics
fit_llmd_statistics = _VLA_FAIL.fit_llmd_statistics
fit_pca_residual_statistics = _VLA_FAIL.fit_pca_residual_statistics
knn_score = _VLA_FAIL.knn_score
llmd_score = _VLA_FAIL.llmd_score
pca_residual_score = _VLA_FAIL.pca_residual_score
vim_default_principal_dim = _VLA_FAIL.vim_default_principal_dim


FEATURE_KEYS = ("bridge", "action_expert_final")
DETECTOR_NAMES = (
    "bridge_llmd",
    "bridge_deep_knn",
    "bridge_pca_residual",
    "final_llmd",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_feature_cache(feature_cache: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    layers = feature_cache.get("features")
    if not isinstance(layers, Mapping):
        raise ValueError("feature cache needs a features mapping")
    result: dict[str, torch.Tensor] = {}
    count: int | None = None
    for key in FEATURE_KEYS:
        if key not in layers:
            raise ValueError(f"feature cache is missing {key}")
        values = torch.as_tensor(layers[key], dtype=torch.float32, device="cpu")
        if values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1 or values.shape[2] < 2:
            raise ValueError(f"{key} must have shape [anchors, tokens, dim] with at least two anchors")
        if not torch.isfinite(values).all():
            raise ValueError(f"{key} feature cache is non-finite")
        if count is None:
            count = int(values.shape[0])
        elif count != int(values.shape[0]):
            raise ValueError("all feature arrays must contain the same selected anchors")
        result[key] = values
    selected = feature_cache.get("selected_anchors")
    if not isinstance(selected, list) or len(selected) != count:
        raise ValueError("feature cache selected_anchors length does not match features")
    return result


def fit_reference_assets(feature_cache: Mapping[str, Any], *, knn_k: int = 10) -> dict[str, Any]:
    """Fit exact LLMD and frozen-implementation kNN/PCA reference assets."""
    if knn_k < 1:
        raise ValueError("knn_k must be positive")
    features = _validate_feature_cache(feature_cache)
    bridge, final = features["bridge"], features["action_expert_final"]
    if bridge.shape[1] != 1:
        raise ValueError("bridge feature must be a single already-pooled token")
    if bridge.shape[0] < knn_k:
        raise ValueError("feature cache has too few anchors for requested kNN k")
    assets = {
        "bridge_llmd": {
            "kind": "llmd",
            "feature_key": "bridge",
            "statistics": fit_llmd_statistics(bridge).state_dict(),
            "paper_mapping": "LLMD on final VLM-to-Action bridge feature",
        },
        "bridge_deep_knn": {
            "kind": "knn",
            "feature_key": "bridge",
            "statistics": fit_knn_statistics(bridge, k=knn_k).state_dict(),
            "paper_mapping": "Deep kNN: normalized k-th squared L2 distance",
        },
        "bridge_pca_residual": {
            "kind": "pca_residual",
            "feature_key": "bridge",
            "statistics": fit_pca_residual_statistics(bridge).state_dict(),
            "paper_mapping": "ViM principal-subspace residual only; no unavailable classifier logits",
        },
        "final_llmd": {
            "kind": "llmd",
            "feature_key": "action_expert_final",
            "statistics": fit_llmd_statistics(final).state_dict(),
            "paper_mapping": "VLA-FAIL LLMD: max over final Action Expert action-token scores",
        },
    }
    return {
        "format": "libero_plus_failure_reference_assets_v1",
        "reference_protocol": {
            "demos_per_task": 10,
            "anchors_per_demo": 10,
            "expected_total_anchors": 1000,
            "successful_expert_only": True,
            "feature_forward_per_anchor": 1,
        },
        "feature_cache_format": str(feature_cache.get("format", "")),
        "feature_cache_selection_sha256": str(feature_cache.get("selection_sha256", "")),
        "num_reference_anchors": int(bridge.shape[0]),
        "shapes": {key: list(value.shape) for key, value in features.items()},
        "knn_k": knn_k,
        "detectors": assets,
    }


def _load_statistics(spec: Mapping[str, Any]) -> LLMDStatistics | KNNStatistics | PCAResidualStatistics:
    kind = str(spec["kind"])
    if kind == "llmd":
        return LLMDStatistics.from_state_dict(spec["statistics"])
    if kind == "knn":
        return KNNStatistics.from_state_dict(spec["statistics"])
    if kind == "pca_residual":
        return PCAResidualStatistics.from_state_dict(spec["statistics"])
    raise ValueError(f"unsupported detector type {kind}")


def score_features(features: Mapping[str, Any], assets: Mapping[str, Any]) -> dict[str, float]:
    """Score one stored policy decision without re-running the model."""
    if assets.get("format") not in {
        "libero_plus_failure_reference_assets_v1",
        "libero10_all_observation_reference_assets_v1",
    }:
        raise ValueError("not a LIBERO-Plus reference asset")
    scores: dict[str, float] = {}
    for name in DETECTOR_NAMES:
        spec = assets["detectors"][name]
        values = torch.as_tensor(features[spec["feature_key"]], dtype=torch.float32)
        if values.ndim == 2:
            values = values.unsqueeze(0)
        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError(f"feature {spec['feature_key']} must represent one [tokens, dim] decision")
        statistics = _load_statistics(spec)
        if spec["kind"] == "llmd":
            score = llmd_score(values, statistics)  # type: ignore[arg-type]
        elif spec["kind"] == "knn":
            score = knn_score(values, statistics)  # type: ignore[arg-type]
        else:
            score = pca_residual_score(values, statistics)  # type: ignore[arg-type]
        scores[name] = float(score.item())
    return scores


class ReferenceScorer:
    """Keep immutable detector statistics, especially the kNN bank, resident.

    The small exploratory bank was inexpensive to reload for every score.  A
    full 100k-observation bridge bank is not.  This wrapper constructs each
    statistic once and keeps its tensors on the requested device for a whole
    scoring run.
    """

    def __init__(self, assets: Mapping[str, Any], *, device: torch.device | str = "cpu") -> None:
        if assets.get("format") not in {
            "libero_plus_failure_reference_assets_v1",
            "libero10_all_observation_reference_assets_v1",
        }:
            raise ValueError("not a supported LIBERO reference asset")
        self._assets = assets
        self._device = torch.device(device)
        self._statistics = {name: _load_statistics(assets["detectors"][name]) for name in DETECTOR_NAMES}
        for statistics in self._statistics.values():
            for value in vars(statistics).values():
                if isinstance(value, torch.Tensor):
                    value.data = value.to(self._device)

    def score_features(self, features: Mapping[str, Any]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name in DETECTOR_NAMES:
            spec = self._assets["detectors"][name]
            values = torch.as_tensor(features[spec["feature_key"]], dtype=torch.float32, device=self._device)
            if values.ndim == 2:
                values = values.unsqueeze(0)
            if values.ndim != 3 or values.shape[0] != 1:
                raise ValueError("feature %s must represent one [tokens, dim] decision" % spec["feature_key"])
            statistics = self._statistics[name]
            if spec["kind"] == "llmd":
                score = llmd_score(values, statistics)  # type: ignore[arg-type]
            elif spec["kind"] == "knn":
                score = knn_score(values, statistics)  # type: ignore[arg-type]
            else:
                score = pca_residual_score(values, statistics)  # type: ignore[arg-type]
            scores[name] = float(score.item())
        return scores


def conformal_thresholds(
    successful_rollout_scores: Mapping[str, list[list[float]]], *, delta: float = 0.05
) -> dict[str, Any]:
    """Fit one fixed q=.95 threshold per score trace, including ACC separately."""
    result: dict[str, Any] = {}
    for name, traces in successful_rollout_scores.items():
        result[name] = {
            "threshold": constant_split_conformal_threshold(traces, delta=delta),
            "calibration_rollouts": len(traces),
            "delta": delta,
            "order_statistic_rank": min(len(traces), int(__import__("math").ceil((len(traces) + 1) * (1.0 - delta)))),
        }
    return {"format": "libero_plus_failure_thresholds_v1", "thresholds": result}


def save_assets(*, feature_cache: Mapping[str, Any], output_dir: Path, knn_k: int = 10) -> dict[str, Any]:
    """Write immutable binary assets plus a small JSON manifest."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    assets = fit_reference_assets(feature_cache, knn_k=knn_k)
    output_dir.mkdir(parents=True, exist_ok=False)
    asset_path = output_dir / "reference_assets.pt"
    torch.save(assets, asset_path)
    manifest = {key: value for key, value in assets.items() if key != "detectors"}
    manifest.update(
        {
            "reference_assets_path": str(asset_path),
            "reference_assets_sha256": sha256(asset_path),
            "detectors": {
                name: {key: value for key, value in spec.items() if key != "statistics"}
                for name, spec in assets["detectors"].items()
            },
        }
    )
    (output_dir / "reference_assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
