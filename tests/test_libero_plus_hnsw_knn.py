from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "tools" / "libero_plus_failure" / "hnsw_knn.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_hnsw_knn", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _assets() -> dict[str, object]:
    bank = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    return {
        "detectors": {
            "bridge_deep_knn": {
                "statistics": {"bank": bank, "k": 2, "normalization_epsilon": 1e-10}
            }
        }
    }


def test_bridge_bank_reads_frozen_normalized_statistics() -> None:
    bank, k, epsilon = MODULE.bridge_bank_from_assets(_assets())
    assert bank.shape == (3, 2)
    assert k == 2
    assert epsilon == pytest.approx(1e-10)
    assert np.allclose(np.linalg.norm(bank, axis=1), 1.0)


def test_bridge_bank_rejects_non_normalized_asset() -> None:
    assets = _assets()
    assets["detectors"]["bridge_deep_knn"]["statistics"]["bank"] *= 2
    with pytest.raises(ValueError, match="pre-normalized"):
        MODULE.bridge_bank_from_assets(assets)


class _FakeHNSW:
    def __init__(self) -> None:
        self.efConstruction = 0
        self.efSearch = 0


class _FakeIndex:
    def __init__(self, dimension: int, m: int, metric: int) -> None:
        self.dimension, self.m, self.metric = dimension, m, metric
        self.hnsw = _FakeHNSW()
        self.values: np.ndarray | None = None
        self.ntotal = 0

    def add(self, values: np.ndarray) -> None:
        self.values = values
        self.ntotal = len(values)

    def search(self, values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.values is not None
        distances = ((values[:, None, :] - self.values[None, :, :]) ** 2).sum(axis=-1)
        order = np.argsort(distances, axis=1)[:, :k]
        return np.take_along_axis(distances, order, axis=1), order


class _FakeFaiss:
    METRIC_L2 = 1
    IndexHNSWFlat = _FakeIndex


def test_hnsw_query_preserves_kth_l2_semantics() -> None:
    bank = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    index = MODULE.build_index(bank, m=2, ef_construction=4, ef_search=8, faiss_module=_FakeFaiss)
    result = MODULE.query_kth_squared_l2(index, np.asarray([[1.0, 0.0]], dtype=np.float32), k=2)
    assert result.tolist() == pytest.approx([2.0])


def test_hnsw_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="HNSW parameters"):
        MODULE.build_index(np.eye(2, dtype=np.float32), m=1, faiss_module=_FakeFaiss)
