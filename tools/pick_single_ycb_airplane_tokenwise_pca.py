"""Reusable assets and scoring for airplane token-wise independent PCA.

The module contains no simulator/model loading.  Builders persist compact
token blocks, while evaluators load every block onto the requested device once
and score probes produced by the same action-generating forward.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from rlinf.algorithms.vla_fail import (
    PCAResidualStatistics,
    TokenwisePCAResidualStatistics,
    pca_residual_score,
    tokenwise_pca_z_scores,
    tokenwise_topk_mean,
)


FORMAT = "pick_single_ycb_airplane_tokenwise_pca_topk16_v1"
LOCATIONS = ("vlm_input", "bridge")
MAIN_METHODS = (
    "vlm_input_pooled_pca",
    "vlm_input_token_pca_top16",
    "bridge_pooled_pca",
    "bridge_token_pca_top16",
)
SOURCE_NAMES = ("base_camera", "wrist_camera", "language_state")
SOURCE_TOPK = {"base_camera": 8, "wrist_camera": 8, "language_state": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash either one immutable file or a directory tree deterministically."""

    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(sha256(child).encode("ascii"))
    return digest.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def valid_token_mean(features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Pool only real prefix tokens to one [B,1,D] PCA feature."""

    if features.ndim != 3 or valid_mask.shape != tuple(features.shape[:2]):
        raise ValueError("prefix features and mask must have [B,T,D] / [B,T] shapes")
    weights = valid_mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1, keepdim=True)
    if torch.any(counts <= 0):
        raise ValueError("every prefix must contain at least one valid token")
    return (features * weights).sum(dim=1, keepdim=True) / counts


def token_source_masks(source_ids: torch.Tensor, *, tokens: int) -> dict[str, torch.Tensor]:
    """Build the static base/wrist/language selectors from a probe source map."""

    ids = torch.as_tensor(source_ids, dtype=torch.int64).reshape(-1)
    if ids.shape != (tokens,):
        raise ValueError(f"source_ids must have shape [{tokens}], got {tuple(ids.shape)}")
    if torch.any((ids < 0) | (ids >= len(SOURCE_NAMES))):
        raise ValueError("probe source_ids contain an unknown modality")
    return {name: ids == index for index, name in enumerate(SOURCE_NAMES)}


def lerobot_sample_to_policy_observation(
    sample: Mapping[str, Any], *, task_description: str
) -> dict[str, Any]:
    """Convert one decoded LeRobot row to the two-view π0.5 input contract."""

    def as_hwc_batch(value: Any) -> torch.Tensor:
        image = torch.as_tensor(value)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"expected image [H,W,C] or [B,H,W,C], got {tuple(image.shape)}")
        if image.shape[1] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = image.permute(0, 2, 3, 1)
        if image.shape[-1] not in (1, 3, 4):
            raise ValueError(f"cannot identify image channel dimension in {tuple(image.shape)}")
        return image.contiguous()

    try:
        main = as_hwc_batch(sample["image"])
        wrist = as_hwc_batch(sample["wrist_image"])
        state = torch.as_tensor(sample["state"])
    except KeyError as error:
        raise KeyError("airplane ID data must expose image, wrist_image, and state fields") from error
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2:
        raise ValueError(f"expected state [D] or [B,D], got {tuple(state.shape)}")
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "states": state,
        "task_descriptions": [task_description] * state.shape[0],
        "task_ids": torch.zeros(state.shape[0], dtype=torch.long),
    }


@dataclass(frozen=True)
class TokenBlock:
    token_indices: torch.Tensor
    statistics: TokenwisePCAResidualStatistics

    def validate(self, *, tokens: int, hidden_dim: int) -> None:
        indices = self.token_indices.to(dtype=torch.int64)
        if indices.ndim != 1 or indices.numel() < 1:
            raise ValueError("token PCA block needs a non-empty one-dimensional token index")
        if torch.any(indices < 0) or torch.any(indices >= tokens):
            raise ValueError("token PCA block indices are outside prefix range")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("token PCA block indices must be unique")
        self.statistics.validate()
        if self.statistics.mean.shape != (indices.numel(), hidden_dim):
            raise ValueError("token PCA block statistic shape does not match its token indices")


class LocationScorer:
    """Keep pooled + independent token PCA statistics resident for one location."""

    def __init__(self, spec: Mapping[str, Any], *, root: Path, device: torch.device) -> None:
        self.device = device
        self.tokens = int(spec["tokens"])
        self.hidden_dim = int(spec["hidden_dim"])
        pooled_payload = load_torch(root / str(spec["pooled_statistics_file"]))
        self.pooled = PCAResidualStatistics.from_state_dict(pooled_payload["statistics"])
        self.pooled.validate()
        self.pooled = PCAResidualStatistics(
            mean=self.pooled.mean.to(device),
            principal_components=self.pooled.principal_components.to(device),
            principal_dim=self.pooled.principal_dim,
            num_observations=self.pooled.num_observations,
        )
        blocks: list[TokenBlock] = []
        seen = torch.zeros(self.tokens, dtype=torch.bool)
        for filename in spec["token_block_files"]:
            payload = load_torch(root / str(filename))
            block = TokenBlock(
                token_indices=torch.as_tensor(payload["token_indices"], dtype=torch.int64),
                statistics=TokenwisePCAResidualStatistics.from_state_dict(payload["statistics"]),
            )
            block.validate(tokens=self.tokens, hidden_dim=self.hidden_dim)
            if torch.any(seen[block.token_indices]):
                raise ValueError("token PCA asset has overlapping blocks")
            seen[block.token_indices] = True
            blocks.append(
                TokenBlock(
                    token_indices=block.token_indices.to(device),
                    statistics=TokenwisePCAResidualStatistics(
                        mean=block.statistics.mean.to(device),
                        principal_components=block.statistics.principal_components.to(device),
                        residual_mean=block.statistics.residual_mean.to(device),
                        residual_std=block.statistics.residual_std.to(device),
                        eligible_tokens=block.statistics.eligible_tokens.to(device),
                        observation_counts=block.statistics.observation_counts.to(device),
                        principal_dim=block.statistics.principal_dim,
                        min_observations=block.statistics.min_observations,
                    ),
                )
            )
        if not torch.all(seen):
            raise ValueError("token PCA asset does not cover every prefix position")
        self.blocks = tuple(blocks)

    def score(self, features: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or tuple(features.shape[1:]) != (self.tokens, self.hidden_dim):
            raise ValueError("runtime feature shape differs from fitted token PCA asset")
        if valid_mask.shape != tuple(features.shape[:2]):
            raise ValueError("runtime valid mask differs from fitted token PCA asset")
        values = features.to(device=self.device, dtype=torch.float32)
        mask = valid_mask.to(device=self.device, dtype=torch.bool)
        pooled = pca_residual_score(valid_token_mean(values, mask), self.pooled)
        z_scores = torch.full((values.shape[0], self.tokens), -torch.inf, device=self.device)
        for block in self.blocks:
            positions = block.token_indices
            z_scores[:, positions] = tokenwise_pca_z_scores(
                values[:, positions], mask[:, positions], block.statistics
            )
        return pooled, z_scores


class TokenwisePCAScorer:
    """Score all four registered methods from one π0.5 prefix probe."""

    def __init__(self, asset_dir: Path, *, device: torch.device | str = "cuda") -> None:
        self.asset_dir = Path(asset_dir)
        self.device = torch.device(device)
        manifest_path = self.asset_dir / "assets_manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"not a token-wise airplane PCA asset: {asset_dir}")
        self.source_ids = torch.as_tensor(self.manifest["source_ids"], dtype=torch.int64, device=self.device)
        self.locations = {
            location: LocationScorer(self.manifest["locations"][location], root=self.asset_dir, device=self.device)
            for location in LOCATIONS
        }
        tokens = self.locations["vlm_input"].tokens
        if self.source_ids.shape != (tokens,) or self.locations["bridge"].tokens != tokens:
            raise ValueError("token PCA source map and location assets disagree on prefix length")
        self.source_masks = {name: mask.to(self.device) for name, mask in token_source_masks(self.source_ids, tokens=tokens).items()}

    def score_probe(self, probe: Mapping[str, Any]) -> dict[str, Any]:
        if tuple(probe.get("source_names", ())) != SOURCE_NAMES:
            raise ValueError("runtime probe source naming differs from PCA asset")
        runtime_ids = torch.as_tensor(probe["source_ids"], dtype=torch.int64)
        if not torch.equal(runtime_ids.cpu(), self.source_ids.cpu()):
            raise ValueError("runtime probe source order differs from PCA asset")
        mask = torch.as_tensor(probe["valid_mask"], dtype=torch.bool, device=self.device)
        payload: dict[str, Any] = {"scores": {}, "modalities": {}, "topk": {}}
        for location in LOCATIONS:
            pooled, z_scores = self.locations[location].score(torch.as_tensor(probe[location]), mask)
            top16, indices = tokenwise_topk_mean(z_scores, k=16)
            prefix = f"{location}_"
            payload["scores"][f"{prefix}pooled_pca"] = float(pooled[0].item())
            payload["scores"][f"{prefix}token_pca_top16"] = float(top16[0].item())
            indices_list = [int(value) for value in indices[0].detach().cpu().tolist()]
            sources = [SOURCE_NAMES[int(self.source_ids[index].item())] for index in indices_list]
            payload["topk"][location] = {
                "indices": indices_list,
                "sources": sources,
                "source_fraction": {name: sources.count(name) / 16.0 for name in SOURCE_NAMES},
            }
            payload["modalities"][location] = {}
            for name, source_mask in self.source_masks.items():
                masked = torch.where(source_mask.unsqueeze(0), z_scores, torch.full_like(z_scores, -torch.inf))
                value, _ = tokenwise_topk_mean(masked, k=SOURCE_TOPK[name])
                payload["modalities"][location][name] = float(value[0].item())
        return payload

    def resident_asset_bytes(self) -> int:
        total = 0
        for location in self.locations.values():
            for value in (location.pooled.mean, location.pooled.principal_components):
                total += value.numel() * value.element_size()
            for block in location.blocks:
                for value in (
                    block.statistics.mean,
                    block.statistics.principal_components,
                    block.statistics.residual_mean,
                    block.statistics.residual_std,
                    block.statistics.eligible_tokens,
                    block.statistics.observation_counts,
                ):
                    total += value.numel() * value.element_size()
        return total


def iter_feature_shards(feature_dir: Path) -> Iterable[Path]:
    manifest = json.loads((feature_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "pick_single_ycb_airplane_prefix_feature_cache_v1":
        raise ValueError("not an airplane prefix feature cache")
    for item in manifest["shards"]:
        path = feature_dir / str(item["file"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"feature shard is missing or has SHA mismatch: {path}")
        yield path
