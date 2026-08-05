#!/usr/bin/env python3
"""Fit pooled and per-token independent PCA assets from airplane ID shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    fit_pca_residual_statistics,
    fit_tokenwise_pca_residual_statistics,
)
from tools.pick_single_ycb_airplane_tokenwise_pca import (  # noqa: E402
    FORMAT,
    LOCATIONS,
    iter_feature_shards,
    load_torch,
    sha256,
    valid_token_mean,
)


def _load_manifest(feature_dir: Path) -> dict[str, Any]:
    return json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))


def _inspect(feature_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    manifest = _load_manifest(feature_dir)
    paths = list(iter_feature_shards(feature_dir))
    if manifest["num_observations"] != sum(int(load_torch(path)["vlm_input"].shape[0]) for path in paths):
        raise ValueError("feature manifest observation count disagrees with shard tensors")
    first = load_torch(paths[0])
    source_ids = torch.as_tensor(first["source_ids"], dtype=torch.int64)
    source_names = tuple(first["source_names"])
    tokens, hidden_dim = first["vlm_input"].shape[1:]
    for path in paths:
        shard = load_torch(path)
        if tuple(shard["vlm_input"].shape[1:]) != (tokens, hidden_dim):
            raise ValueError("VLM input shard shape changed across ID observations")
        if tuple(shard["bridge"].shape[1:]) != (tokens, hidden_dim):
            raise ValueError("Bridge shard shape changed across ID observations")
        if tuple(shard["valid_mask"].shape[1:]) != (tokens,):
            raise ValueError("valid mask shape changed across ID observations")
        if not torch.equal(torch.as_tensor(shard["source_ids"], dtype=torch.int64), source_ids):
            raise ValueError("token source ids changed across ID observations")
        if tuple(shard["source_names"]) != source_names:
            raise ValueError("token source names changed across ID observations")
    manifest = {**manifest, "source_ids": source_ids.tolist(), "source_names": list(source_names), "prefix_tokens": tokens, "hidden_dim": hidden_dim}
    return paths, manifest


def _gather_block(
    paths: list[Path], *, key: str, token_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one small token block, never the whole prefix tensor."""

    features, masks = [], []
    for path in paths:
        shard = load_torch(path)
        value = torch.as_tensor(shard[key])[:, token_indices]
        mask = torch.as_tensor(shard["valid_mask"], dtype=torch.bool)
        mask = mask[:, token_indices]
        features.append(value)
        masks.append(mask)
    return torch.cat(features, dim=0), torch.cat(masks, dim=0)


def _fit_pooled(paths: list[Path], *, key: str) -> dict[str, Any]:
    # A pooled observation is just [1, hidden], so retaining all 9,109 of
    # them is small.  Crucially, do not concatenate the complete prefix tensor
    # here: that would be tens of GiB for this experiment.
    pooled_rows = []
    for path in paths:
        shard = load_torch(path)
        pooled_rows.append(
            valid_token_mean(
                torch.as_tensor(shard[key], dtype=torch.float32),
                torch.as_tensor(shard["valid_mask"], dtype=torch.bool),
            )
        )
    pooled = torch.cat(pooled_rows, dim=0)
    statistics = fit_pca_residual_statistics(pooled, principal_dim=1000)
    return {"statistics": statistics.state_dict(), "num_observations": int(pooled.shape[0])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--principal-dim", type=int, default=1000)
    parser.add_argument("--min-observations", type=int, default=1001)
    parser.add_argument("--token-block-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite token PCA assets: {args.output_dir}")
    if args.principal_dim != 1000:
        raise ValueError("the registered airplane token PCA protocol fixes principal_dim=1000")
    if args.min_observations < 1001 or args.token_block_size < 1:
        raise ValueError("registered protocol needs min_observations>=1001 and positive token blocks")
    paths, source = _inspect(args.feature_dir)
    if int(source["num_observations"]) != 9109:
        raise ValueError("registered airplane protocol requires all 9,109 frozen ID observations")
    if int(source["hidden_dim"]) < args.principal_dim:
        raise ValueError("π0.5 hidden dimension is smaller than registered PCA rank")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    locations: dict[str, Any] = {}
    for location in LOCATIONS:
        key = location
        location_dir = args.output_dir / location
        block_dir = location_dir / "token_blocks"
        block_dir.mkdir(parents=True)
        pooled_payload = _fit_pooled(paths, key=key)
        pooled_path = location_dir / "pooled_pca.pt"
        torch.save(pooled_payload, pooled_path)
        block_files: list[str] = []
        for start in range(0, int(source["prefix_tokens"]), args.token_block_size):
            indices = torch.arange(start, min(start + args.token_block_size, int(source["prefix_tokens"])), dtype=torch.int64)
            features, mask = _gather_block(paths, key=key, token_indices=indices)
            statistics = fit_tokenwise_pca_residual_statistics(
                features,
                mask,
                principal_dim=args.principal_dim,
                min_observations=args.min_observations,
            )
            payload = {"token_indices": indices, "statistics": statistics.state_dict()}
            path = block_dir / f"tokens_{start:04d}_{int(indices[-1]) + 1:04d}.pt"
            torch.save(payload, path)
            block_files.append(str(path.relative_to(args.output_dir)))
            print(f"[tokenwise-pca-assets] location={location} tokens={start}:{int(indices[-1]) + 1}", flush=True)
        locations[location] = {
            "tokens": int(source["prefix_tokens"]),
            "hidden_dim": int(source["hidden_dim"]),
            "pooled_statistics_file": str(pooled_path.relative_to(args.output_dir)),
            "pooled_statistics_sha256": sha256(pooled_path),
            "token_block_files": block_files,
            "token_block_sha256": {filename: sha256(args.output_dir / filename) for filename in block_files},
        }

    manifest = {
        "format": FORMAT,
        "task": "pick_single_ycb_airplane",
        "feature_dir": str(args.feature_dir),
        "feature_manifest_sha256": sha256(args.feature_dir / "feature_manifest.json"),
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "norm_stats": source["norm_stats"],
        "norm_stats_sha256": source["norm_stats_sha256"],
        "dataset_root": source["dataset_root"],
        "dataset_episodes_sha256": source["dataset_episodes_sha256"],
        "num_id_observations": source["num_observations"],
        "source_ids": source["source_ids"],
        "source_names": source["source_names"],
        "principal_dim": args.principal_dim,
        "min_observations": args.min_observations,
        "token_block_size": args.token_block_size,
        "locations": locations,
    }
    (args.output_dir / "assets_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "locations"}, indent=2))


if __name__ == "__main__":
    main()
