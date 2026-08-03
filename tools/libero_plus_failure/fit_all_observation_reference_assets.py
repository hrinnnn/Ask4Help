#!/usr/bin/env python3
"""Fit full-bank LLMD, Deep kNN, and PCA-residual assets without a monolith.

The final Action Expert features are consumed in streaming form.  Only the
Bridge representation is materialized as a kNN bank because exact kNN needs
the original normalized ID vectors at runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
from libero_plus_failure.full_reference_bank import (  # noqa: E402
    BRIDGE_SHAPE,
    FINAL_SHAPE,
    all_records,
    read_episode_metadata,
    sha256,
    shard_paths,
    validate_record_sequence,
)

_ASSET_PATH = TOOLS / "libero_plus_failure_assets.py"
_SPEC = importlib.util.spec_from_file_location("full_bank_assets", _ASSET_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ASSETS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = ASSETS
_SPEC.loader.exec_module(ASSETS)


@dataclass
class StreamingMoments:
    count: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def empty(cls, shape: tuple[int, int]) -> "StreamingMoments":
        return cls(0, torch.zeros(shape, dtype=torch.float64), torch.zeros((shape[0], shape[1], shape[1]), dtype=torch.float64))

    def update(self, values: np.ndarray) -> None:
        tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu")
        if tensor.ndim != 3 or tuple(tensor.shape[1:]) != tuple(self.mean.shape):
            raise ValueError("streamed features have an incompatible shape")
        if not torch.isfinite(tensor).all():
            raise ValueError("streamed features are non-finite")
        batch = int(tensor.shape[0])
        if batch == 0:
            return
        batch_mean = tensor.mean(dim=0)
        centered = tensor - batch_mean.unsqueeze(0)
        batch_m2 = torch.einsum("nth,ntk->thk", centered, centered)
        if self.count == 0:
            self.count, self.mean, self.m2 = batch, batch_mean, batch_m2
            return
        total = self.count + batch
        delta = batch_mean - self.mean
        self.m2 = self.m2 + batch_m2 + torch.einsum("th,tk->thk", delta, delta) * (self.count * batch / total)
        self.mean = self.mean + delta * (batch / total)
        self.count = total

    def covariance(self, ridge: float) -> torch.Tensor:
        if self.count < 2:
            raise ValueError("need at least two reference observations")
        dim = self.mean.shape[-1]
        return self.m2 / self.count + ridge * torch.eye(dim, dtype=torch.float64).expand(self.mean.shape[0], -1, -1)


def _llmd_from_moments(moments: StreamingMoments, *, ridge: float, device: str) -> Any:
    covariance = moments.covariance(ridge).to(device)
    precision = torch.linalg.inv(covariance).cpu()
    result = ASSETS.LLMDStatistics(moments.mean.cpu(), precision, ridge, moments.count)
    result.validate()
    return result


def _pca_from_moments(moments: StreamingMoments, *, principal_dim: int, device: str) -> Any:
    covariance = moments.covariance(0.0).to(device)
    _values, vectors = torch.linalg.eigh(covariance)
    components = vectors[:, :, -principal_dim:].cpu().to(dtype=torch.float32)
    result = ASSETS.PCAResidualStatistics(
        moments.mean.cpu().to(dtype=torch.float32), components, principal_dim, moments.count
    )
    result.validate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite full reference assets")
    validation_path = args.bank_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    paths = shard_paths(args.bank_root)
    metadata = [read_episode_metadata(path.with_suffix(".json")) for path in paths]
    records = all_records(metadata)
    summary = validate_record_sequence(records, expected_frames=int(validation["records"]))
    expected = int(summary["records"])
    bridge_moments = StreamingMoments.empty(BRIDGE_SHAPE)
    final_moments = StreamingMoments.empty(FINAL_SHAPE)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_bank_path = args.output_dir / "bridge_bank_raw.npy"
    bridge_bank = np.lib.format.open_memmap(raw_bank_path, mode="w+", dtype=np.float32, shape=(expected,) + BRIDGE_SHAPE)
    cursor = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            bridge = np.asarray(payload["bridge"], dtype=np.float32)
            final = np.asarray(payload["action_expert_final"], dtype=np.float32)
        bridge_moments.update(bridge)
        final_moments.update(final)
        bridge_bank[cursor: cursor + len(bridge)] = bridge
        cursor += len(bridge)
    if cursor != expected or bridge_moments.count != expected or final_moments.count != expected:
        raise RuntimeError("streaming fit consumed the wrong number of features")
    bridge_bank.flush()
    bridge_llmd = _llmd_from_moments(bridge_moments, ridge=args.ridge, device=args.device)
    final_llmd = _llmd_from_moments(final_moments, ridge=args.ridge, device=args.device)
    principal_dim = ASSETS.vim_default_principal_dim(BRIDGE_SHAPE[-1])
    bridge_pca = _pca_from_moments(bridge_moments, principal_dim=principal_dim, device=args.device)
    # Exact normalized kNN bank; this copy becomes part of the immutable asset.
    tensor_bank = torch.from_numpy(np.asarray(bridge_bank)).to(dtype=torch.float32)
    bridge_knn = ASSETS.fit_knn_statistics(tensor_bank, k=args.knn_k)
    payload = {
        "format": "libero10_all_observation_reference_assets_v1",
        "reference_protocol": {
            "all_observations": True, "num_reference_anchors": expected, "action_horizon": 10,
            "flow_timestep": 0.0, "fixed_prior": True, "tail_frames_included": True,
            "knn_k": args.knn_k, "pca_principal_dim": principal_dim, "llmd_ridge": args.ridge,
        },
        "bank_validation_sha256": sha256(validation_path),
        "bank_request_sha256": sha256(args.bank_root / "reference_bank_request.json"),
        "num_reference_anchors": expected,
        "shapes": {"bridge": [expected, *BRIDGE_SHAPE], "action_expert_final": [expected, *FINAL_SHAPE]},
        "detectors": {
            "bridge_llmd": {"kind": "llmd", "feature_key": "bridge", "statistics": bridge_llmd.state_dict(), "paper_mapping": "Bridge LLMD"},
            "bridge_deep_knn": {"kind": "knn", "feature_key": "bridge", "statistics": bridge_knn.state_dict(), "paper_mapping": "Exact normalized k=10 bridge Deep kNN"},
            "bridge_pca_residual": {"kind": "pca_residual", "feature_key": "bridge", "statistics": bridge_pca.state_dict(), "paper_mapping": "Bridge PCA residual"},
            "final_llmd": {"kind": "llmd", "feature_key": "action_expert_final", "statistics": final_llmd.state_dict(), "paper_mapping": "VLA-FAIL final token-wise LLMD"},
        },
    }
    asset_path = args.output_dir / "reference_assets.pt"
    torch.save(payload, asset_path)
    manifest = {key: value for key, value in payload.items() if key != "detectors"}
    manifest.update({"reference_assets_path": str(asset_path), "reference_assets_sha256": sha256(asset_path),
                     "detectors": {name: {key: value for key, value in spec.items() if key != "statistics"} for name, spec in payload["detectors"].items()}})
    (args.output_dir / "reference_assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
