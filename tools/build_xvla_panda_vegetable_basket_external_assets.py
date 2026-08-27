#!/usr/bin/env python3
"""Build the external visual/state baseline references for Panda basket data."""

from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.pick_single_ycb_airplane_external_detectors import (
    CRSAILBank,
    OfficialFIDeLMemory,
    fit_crsail_bank,
    fit_official_fidel_memory,
)


def read_image(value: np.ndarray | bytes) -> Image.Image:
    array = np.asarray(value)
    if array.ndim == 1:
        return Image.open(BytesIO(bytes(array))).convert("RGB")
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--crsail-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    files = sorted((args.dataset / "data").glob("episode_*.h5"))
    if not files:
        raise FileNotFoundError(f"no episodes under {args.dataset / 'data'}")
    args.output_dir.mkdir(parents=True)

    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    encoder.eval().to(args.device)
    preprocess = weights.transforms()

    visual_episodes: list[torch.Tensor] = []
    state_parts: list[torch.Tensor] = []
    total = 0
    try:
        for path in files:
            with h5py.File(path, "r") as h5:
                images = h5["images"]
                states = np.asarray(h5["proprio"], dtype=np.float32)
                count = min(len(images), len(states))
                episode_parts: list[torch.Tensor] = []
                for start in range(0, count, args.batch_size):
                    stop = min(start + args.batch_size, count)
                    batch = torch.stack(
                        [preprocess(read_image(images[index])) for index in range(start, stop)]
                    ).to(args.device)
                    with torch.inference_mode():
                        encoded = encoder(batch).detach().cpu().float()
                    episode_parts.append(encoded)
                    print(f"[panda-external-assets] observations={total + stop}", flush=True)
                visual = torch.cat(episode_parts, dim=0)
                visual_episodes.append(visual)
                state_parts.append(torch.from_numpy(states[:count]))
                total += count
    finally:
        del encoder
        torch.cuda.empty_cache()

    common_horizon = min(int(values.shape[0]) for values in visual_episodes)
    dense = torch.stack(
        [values[:common_horizon].unsqueeze(1) for values in visual_episodes], dim=0
    )
    visual = torch.cat(visual_episodes, dim=0)
    states = torch.cat(state_parts, dim=0)
    fidel = fit_official_fidel_memory(dense)
    state_bank = fit_crsail_bank(states, k=args.crsail_k, standardize=True)
    vision_bank = fit_crsail_bank(visual, k=args.crsail_k, standardize=False)
    payload = {
        "format": "xvla_panda_vegetable_basket_external_detector_assets_v1",
        "dataset": str(args.dataset.resolve()),
        "num_observations": total,
        "num_episodes": len(files),
        "common_horizon": common_horizon,
        "encoder": {
            "name": "torchvision_resnet18_default",
            "source": "official ResNet18 weights used by FIDeL baseline",
            "feature_dim": int(visual.shape[-1]),
        },
        "fidel": {
            "variant": "official_representation_resnet18_euclidean",
            "memory": fidel.state_dict(),
        },
        "crsail": {
            "k": args.crsail_k,
            "observable_state": state_bank.state_dict(),
            "vision_resnet18": vision_bank.state_dict(),
            "notes": {
                "observable_state": "KNN novelty over the Panda policy-observable 10D proprio vector",
                "vision_resnet18": "visual adaptation using the same external encoder",
            },
        },
    }
    asset_path = args.output_dir / "external_detector_assets.pt"
    torch.save(payload, asset_path)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": payload["format"],
                "dataset": payload["dataset"],
                "num_observations": payload["num_observations"],
                "num_episodes": payload["num_episodes"],
                "common_horizon": payload["common_horizon"],
                "encoder": payload["encoder"],
                "asset_path": str(asset_path),
                "methods": [
                    "fidel_official",
                    "crsail_observable_state_k5",
                    "crsail_vision_k5",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "EXTERNAL_ASSETS_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, default=lambda value: value.tolist() if isinstance(value, torch.Tensor) else value), flush=True)


if __name__ == "__main__":
    main()
