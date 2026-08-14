#!/usr/bin/env python3
"""Build a Bridge-PCA asset from the frozen StackPyramid ID anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


TASK = "stack the red cube next to the green cube and place the blue cube on top"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(args.xvla_root)]
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor
    from rlinf.algorithms.vla_fail import fit_pca_residual_statistics

    device = torch.device("cuda")
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    features = []
    h5_paths = sorted((args.collection_root / "id").rglob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"no H5 files under {args.collection_root / 'id'}")
    episode_count = 0
    anchor_count = 0
    tail_anchor_count = 0
    batch_images: list[list[Image.Image]] = []

    def encode_batch() -> None:
        if not batch_images:
            return
        with torch.autocast("cuda", dtype=torch.bfloat16):
            processed = processor.encode_image(batch_images)
            language = processor.encode_language([TASK] * len(batch_images))
            encoding = model.forward_vlm(
                language["input_ids"].to(device),
                processed["image_input"].to(device),
                processed["image_mask"].to(device),
            )
            tokens = torch.cat(
                [encoding["vlm_features"], encoding["aux_visual_inputs"]], dim=1
            )
            features.append(tokens.float().mean(dim=1).cpu())
        batch_images.clear()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as handle:
                for group_name in sorted(name for name in handle if name.startswith("traj_")):
                    if episode_count >= args.target_episodes:
                        break
                    group = handle[group_name]
                    base = np.asarray(group["obs/sensor_data/base_camera/rgb"], dtype=np.uint8)
                    wrist = np.asarray(group["obs/sensor_data/hand_camera/rgb"], dtype=np.uint8)
                    actions = group["actions"]
                    episode_count += 1
                    length = int(actions.shape[0])
                    anchor_count += length
                    tail_anchor_count += min(9, length)
                    for anchor in range(length):
                        batch_images.append([Image.fromarray(base[anchor]), Image.fromarray(wrist[anchor])])
                        if len(batch_images) >= args.batch_size:
                            encode_batch()
                    if anchor_count and anchor_count % 512 < length:
                        print(f"[stackpyramid-pca] anchors={anchor_count}", flush=True)
            if episode_count >= args.target_episodes:
                break
        encode_batch()

    matrix = torch.cat(features, dim=0).unsqueeze(1)
    statistics = fit_pca_residual_statistics(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "stackpyramid_bridge_pca_assets_v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "collection_root": str(args.collection_root.resolve()),
            "feature": "mean(concat(vlm_features,aux_visual_inputs))",
            "num_observations": int(matrix.shape[0]),
            "hidden_dim": int(matrix.shape[-1]),
            "statistics": statistics.state_dict(),
            "dataset_report": {
                "split": "id",
                "episodes": episode_count,
                "total_anchors": anchor_count,
                "tail_anchors": tail_anchor_count,
                "final_observation_valid_targets": 1,
                "action_horizon": 10,
            },
        },
        args.output,
    )
    manifest = {
        "format": "stackpyramid_bridge_pca_assets_v1",
        "asset": str(args.output.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "num_observations": int(matrix.shape[0]),
        "hidden_dim": int(matrix.shape[-1]),
        "principal_dim": int(statistics.principal_dim),
        "dataset_report": {
            "split": "id",
            "episodes": episode_count,
            "total_anchors": anchor_count,
            "tail_anchors": tail_anchor_count,
            "final_observation_valid_targets": 1,
            "action_horizon": 10,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
