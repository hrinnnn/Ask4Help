#!/usr/bin/env python3
"""Build official FIDeL and CRSAIL reference assets from airplane ID data."""

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

from tools.build_stackcube_vla_fail_statistics import _load_dataset  # noqa: E402
from tools.pick_single_ycb_airplane_external_detectors import (  # noqa: E402
    CRSAIL_FORMAT,
    FIDEL_FORMAT,
    fit_crsail_bank,
    fit_official_fidel_memory,
)


def _chw_float_image(value: Any) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected one image with three dimensions, got {tuple(image.shape)}")
    if image.shape[-1] in (1, 3, 4):
        image = image.permute(2, 0, 1)
    if image.shape[0] not in (1, 3, 4):
        raise ValueError(f"cannot identify image channels in {tuple(image.shape)}")
    image = image[:3].to(dtype=torch.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image.contiguous()


def _scalar(sample: dict[str, Any], name: str) -> int:
    value = torch.as_tensor(sample[name]).reshape(-1)
    if value.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(value.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--crsail-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    from torchvision.models import ResNet18_Weights, resnet18

    dataset = _load_dataset(args.dataset_root)
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval().to(args.device)
    preprocess = weights.transforms()

    feature_parts: list[torch.Tensor] = []
    state_parts: list[torch.Tensor] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    try:
        for start in range(0, len(dataset), args.batch_size):
            stop = min(start + args.batch_size, len(dataset))
            samples = [dataset[index] for index in range(start, stop)]
            images = torch.stack([_chw_float_image(sample["image"]) for sample in samples])
            with torch.inference_mode():
                features = encoder(preprocess(images).to(args.device))
            feature_parts.append(features.detach().cpu().to(torch.float16))
            state_parts.append(
                torch.stack([torch.as_tensor(sample["state"], dtype=torch.float32).reshape(-1) for sample in samples])
            )
            episode_indices.extend(_scalar(sample, "episode_index") for sample in samples)
            frame_indices.extend(_scalar(sample, "frame_index") for sample in samples)
            print(f"[external-assets] observations={stop}/{len(dataset)}", flush=True)
    finally:
        del encoder
        torch.cuda.empty_cache()

    visual = torch.cat(feature_parts).float()
    observable_state = torch.cat(state_parts).float()
    episode_count = max(episode_indices) + 1
    lengths = []
    for episode in range(episode_count):
        frames = [frame for owner, frame in zip(episode_indices, frame_indices) if owner == episode]
        if not frames:
            raise ValueError(f"expert episode {episode} has no frames")
        if sorted(frames) != list(range(max(frames) + 1)):
            raise ValueError(f"expert episode {episode} frame indices are not contiguous")
        lengths.append(max(frames) + 1)
    common_horizon = min(lengths)
    dense = torch.empty((episode_count, common_horizon, 1, visual.shape[-1]), dtype=torch.float32)
    for row, (episode, frame) in enumerate(zip(episode_indices, frame_indices)):
        if frame < common_horizon:
            dense[episode, frame, 0] = visual[row]

    official_fidel = fit_official_fidel_memory(dense)
    crsail_state = fit_crsail_bank(observable_state, k=args.crsail_k, standardize=True)
    crsail_vision = fit_crsail_bank(visual, k=args.crsail_k, standardize=False)
    payload = {
        "format": "pick_single_ycb_airplane_external_detector_assets_v1",
        "dataset_root": str(args.dataset_root),
        "num_observations": len(dataset),
        "num_episodes": episode_count,
        "common_horizon": common_horizon,
        "encoder": {
            "name": "torchvision_resnet18_default",
            "source": "official FIDeL released default",
            "camera": "base_camera",
            "feature_dim": int(visual.shape[-1]),
        },
        "fidel": {
            "format": FIDEL_FORMAT,
            "variant": "official_representation_resnet18_euclidean",
            "memory": official_fidel.state_dict(),
        },
        "crsail": {
            "format": CRSAIL_FORMAT,
            "k": args.crsail_k,
            "observable_state": crsail_state.state_dict(),
            "vision_resnet18": crsail_vision.state_dict(),
            "notes": {
                "observable_state": "CRSAIL rule on the policy-observable proprio vector; standardized by ID statistics",
                "vision_resnet18": "deployable visual adaptation; cosine distance over the same external ResNet18 encoder",
            },
        },
    }
    asset_path = args.output_dir / "external_detector_assets.pt"
    torch.save(payload, asset_path)
    manifest = {
        "format": payload["format"],
        "dataset_root": payload["dataset_root"],
        "num_observations": payload["num_observations"],
        "num_episodes": payload["num_episodes"],
        "common_horizon": payload["common_horizon"],
        "encoder": payload["encoder"],
        "asset_path": str(asset_path),
        "methods": ["fidel_official", "crsail_observable_state_k5", "crsail_vision_k5"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
