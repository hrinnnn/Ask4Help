#!/usr/bin/env python3
"""Fit persistent kNN and PCA-residual assets for internal StackCube probes.

The frozen multi-layer LLMD asset remains the source of the three LLMD
baselines. This script consumes its exact raw feature cache and adds two
training-free alternatives at the same three representation locations:
Deep kNN and a ViM-inspired PCA residual without classifier logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    fit_knn_statistics,
    fit_pca_residual_statistics,
    knn_score,
    pca_residual_score,
    vim_default_principal_dim,
)


CANDIDATE_LAYERS = (
    "vlm_block_08_mean",
    "vlm_bridge_final_mean",
    "action_expert_block_13",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_detector_payload(
    *, feature_cache: dict[str, Any], multilayer_statistics: dict[str, Any], knn_k: int
) -> dict[str, Any]:
    """Fit all non-LLMD comparison assets from the immutable ID cache."""
    if feature_cache.get("format") != "stackcube_multilayer_llmd_feature_cache_v1":
        raise ValueError("not a StackCube multi-layer feature cache")
    if multilayer_statistics.get("format") != "stackcube_multilayer_llmd_statistics_v1":
        raise ValueError("not a StackCube multi-layer statistics asset")
    if feature_cache.get("statistics_sha256") != multilayer_statistics.get("statistics_sha256"):
        raise ValueError("feature cache and multi-layer statistics have different provenance")
    available = feature_cache.get("layers", {})
    missing = [name for name in CANDIDATE_LAYERS if name not in available]
    if missing:
        raise ValueError(f"feature cache is missing candidate layers: {missing}")

    detectors: dict[str, dict[str, Any]] = {}
    for layer in CANDIDATE_LAYERS:
        features = torch.as_tensor(available[layer])
        hidden_dim = int(features.shape[-1])
        knn = fit_knn_statistics(features, k=knn_k)
        pca = fit_pca_residual_statistics(features)
        # A deterministic in-cache query catches serialization/shape mistakes
        # before a costly simulator evaluation starts.
        query = features[:1].to(dtype=torch.float32)
        knn_value = float(knn_score(query, knn)[0].item())
        residual_value = float(pca_residual_score(query, pca)[0].item())
        detectors[f"{layer}__knn_k{knn_k}"] = {
            "kind": "knn",
            "layer": layer,
            "statistics": knn.state_dict(),
            "feature_shape": list(features.shape[1:]),
            "official_mapping": "Deep kNN: L2-normalized feature and k-th squared L2 distance",
            "smoke_score": knn_value,
        }
        detectors[f"{layer}__pca_residual"] = {
            "kind": "pca_residual",
            "layer": layer,
            "statistics": pca.state_dict(),
            "feature_shape": list(features.shape[1:]),
            "principal_dim": vim_default_principal_dim(hidden_dim),
            "official_mapping": "ViM principal-subspace residual only; no classifier logits or virtual-logit term",
            "smoke_score": residual_value,
        }
    return {
        "format": "stackcube_internal_detector_assets_v1",
        "candidate_layers": list(CANDIDATE_LAYERS),
        "knn_k": knn_k,
        "detectors": detectors,
        "llmd_source_statistics": str(multilayer_statistics.get("statistics_path", "")),
        "llmd_source_statistics_sha256": str(multilayer_statistics.get("statistics_sha256", "")),
        "feature_cache_source": str(feature_cache.get("feature_cache_path", "")),
        "feature_cache_sha256": str(feature_cache.get("feature_cache_sha256", "")),
        "checkpoint": str(feature_cache["checkpoint"]),
        "dataset_root": str(feature_cache["dataset_root"]),
        "num_id_observations": len(feature_cache["indices"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--multilayer-statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite detector asset directory: {args.output_dir}")

    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    statistics = torch.load(args.multilayer_statistics, map_location="cpu", weights_only=False)
    statistics = {**statistics, "statistics_sha256": sha256(args.multilayer_statistics)}
    payload = build_detector_payload(
        feature_cache=cache, multilayer_statistics=statistics, knn_k=args.knn_k
    )
    # The original immutable payloads intentionally do not self-reference their
    # filesystem path. Record the exact paths and digests in this derived asset.
    payload["llmd_source_statistics"] = str(args.multilayer_statistics)
    payload["llmd_source_statistics_sha256"] = sha256(args.multilayer_statistics)
    payload["feature_cache_source"] = str(args.feature_cache)
    payload["feature_cache_sha256"] = sha256(args.feature_cache)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    asset_path = args.output_dir / "internal_detector_assets.pt"
    torch.save(payload, asset_path)
    manifest = {
        **{key: value for key, value in payload.items() if key != "detectors"},
        "asset_path": str(asset_path),
        "asset_sha256": sha256(asset_path),
        "detectors": {
            name: {
                key: value
                for key, value in spec.items()
                if key != "statistics"
            }
            for name, spec in payload["detectors"].items()
        },
    }
    (args.output_dir / "internal_detector_assets.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
