"""Faiss HNSW bridge-kNN index for low-latency deployment.

The strict leaderboard keeps its exact, exhaustive Deep kNN implementation.
This module builds a separately versioned approximate index from the identical
frozen bridge bank.  It never mutates the reference asset or reuses an exact
threshold: an HNSW index must receive its own clean-success calibration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


FORMAT = "ask4help_bridge_hnsw_faiss_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "Faiss is required for the HNSW deployment index. Install faiss-cpu "
            "into the isolated experiment dependency directory, not the system environment."
        ) from error
    return faiss


def _normalized(vector: np.ndarray, *, epsilon: float = 1e-10) -> np.ndarray:
    values = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("vectors must have shape [count, dim]")
    if not np.isfinite(values).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values / np.maximum(norms, epsilon), dtype=np.float32)


def bridge_bank_from_assets(assets: Mapping[str, Any]) -> tuple[np.ndarray, int, float]:
    """Extract the exact normalized bridge bank used by the strict baseline."""

    try:
        detector = assets["detectors"]["bridge_deep_knn"]
        stats = detector["statistics"]
        bank = torch.as_tensor(stats["bank"], dtype=torch.float32, device="cpu")
        k = int(stats["k"])
        epsilon = float(stats["normalization_epsilon"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("asset does not contain bridge Deep kNN statistics") from error
    if bank.ndim != 3 or bank.shape[0] != 1:
        raise ValueError("bridge bank must have shape [1, observations, dim]")
    if k < 1 or bank.shape[1] < k or epsilon <= 0:
        raise ValueError("bridge Deep kNN statistics are invalid")
    values = np.ascontiguousarray(bank[0].numpy(), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("bridge bank is non-finite")
    # The source asset stores unit-normalized values. Verify rather than silently
    # normalizing a mismatched asset, because that would no longer match LLMD's
    # paired reference protocol.
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError("bridge bank is expected to be pre-normalized")
    return values, k, epsilon


def build_index(
    bank: np.ndarray,
    *,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
    faiss_module: Any | None = None,
) -> Any:
    """Build a deterministic CPU HNSW-L2 index over normalized bridge vectors."""

    if m < 2 or ef_construction < m or ef_search < 1:
        raise ValueError("HNSW parameters require m>=2, ef_construction>=m, ef_search>=1")
    values = _normalized(bank)
    faiss = require_faiss() if faiss_module is None else faiss_module
    metric_l2 = getattr(faiss, "METRIC_L2", 1)
    index = faiss.IndexHNSWFlat(int(values.shape[1]), int(m), metric_l2)
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)
    index.add(values)
    if int(index.ntotal) != int(values.shape[0]):
        raise RuntimeError("HNSW index dropped reference vectors")
    return index


def query_kth_squared_l2(index: Any, query: np.ndarray, *, k: int, epsilon: float = 1e-10) -> np.ndarray:
    """Return HNSW's k-th normalized squared-L2 distance per bridge feature."""

    if k < 1:
        raise ValueError("k must be positive")
    values = _normalized(query, epsilon=epsilon)
    if int(index.ntotal) < k:
        raise ValueError("index has fewer than k reference vectors")
    distances, ids = index.search(values, int(k))
    distances = np.asarray(distances, dtype=np.float32)
    ids = np.asarray(ids)
    if distances.shape != (len(values), k) or ids.shape != (len(values), k) or np.any(ids < 0):
        raise RuntimeError("HNSW search returned incomplete neighbors")
    return np.maximum(distances[:, k - 1], 0.0)


def build_and_save(
    *,
    assets_path: Path,
    output_dir: Path,
    m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 128,
) -> dict[str, Any]:
    """Build an immutable index plus a manifest bound to one reference asset."""

    if output_dir.exists():
        raise FileExistsError("refusing to overwrite " + str(output_dir))
    try:
        assets = torch.load(assets_path, map_location="cpu", weights_only=False)
    except TypeError:
        assets = torch.load(assets_path, map_location="cpu")
    bank, k, epsilon = bridge_bank_from_assets(assets)
    index = build_index(bank, m=m, ef_construction=ef_construction, ef_search=ef_search)
    faiss = require_faiss()
    output_dir.mkdir(parents=True, exist_ok=False)
    index_path = output_dir / "bridge_hnsw.faiss"
    faiss.write_index(index, str(index_path))
    manifest = {
        "format": FORMAT,
        "reference_assets_path": str(assets_path),
        "reference_assets_sha256": sha256(assets_path),
        "index_path": str(index_path),
        "index_sha256": sha256(index_path),
        "metric": "squared_l2_on_unit_normalized_bridge",
        "num_reference_anchors": int(bank.shape[0]),
        "feature_dim": int(bank.shape[1]),
        "k": k,
        "normalization_epsilon": epsilon,
        "m": int(m),
        "ef_construction": int(ef_construction),
        "ef_search": int(ef_search),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

