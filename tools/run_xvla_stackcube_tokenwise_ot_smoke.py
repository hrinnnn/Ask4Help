#!/usr/bin/env python3
"""Run a small real-X-VLA StackCube token-wise PCA/OT feature smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.xvla_tokenwise_ot import asset_state_dict, fit_tokenwise_pca_ot, token_ot_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-observations", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--principal-dim", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.max_observations < 4 or args.batch_size < 1:
        raise ValueError("smoke requires at least four observations and a positive batch size")

    sys.path.insert(0, str(args.xvla_root.resolve()))
    from datasets.dataset import InfiniteDataReader
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    device = torch.device(args.device)
    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    dataset = InfiniteDataReader(
        metas_path=str(args.metadata),
        num_actions=model.num_actions,
        num_views=2,
        training=False,
        action_mode=model.action_mode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    token_parts: list[torch.Tensor] = []
    observations = 0
    token_count = None
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch in loader:
            language = processor.encode_language(batch.pop("language_instruction"))
            inputs = {**batch, **language}
            inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            images = inputs["image_input"][:, args.view_index]
            image_tokens = model.vlm._encode_image(images)
            if token_count is None:
                token_count = int(image_tokens.shape[1])
            if int(image_tokens.shape[1]) != token_count:
                raise RuntimeError("image token count changed across StackCube observations")
            token_parts.append(image_tokens.float().cpu())
            observations += int(image_tokens.shape[0])
            if observations >= args.max_observations:
                break

    features = torch.cat(token_parts, dim=0)[: args.max_observations]
    valid_mask = torch.ones(features.shape[:2], dtype=torch.bool)
    assets = fit_tokenwise_pca_ot(
        features,
        valid_mask,
        principal_dim=min(args.principal_dim, features.shape[-1]),
        min_observations=4,
    )
    score_first = token_ot_score(features[:1], valid_mask[:1], assets[0], topk=min(4, features.shape[1]))
    score_last = token_ot_score(features[-1:], valid_mask[-1:], assets[0], topk=min(4, features.shape[1]))
    result = {
        "format": "xvla_stackcube_tokenwise_ot_smoke_v1",
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "view_index": args.view_index,
        "observations": int(features.shape[0]),
        "token_count": int(features.shape[1]),
        "hidden_dim": int(features.shape[2]),
        "finite_features": bool(torch.isfinite(features).all()),
        "finite_asset": bool(all(torch.isfinite(value).all() for value in asset_state_dict(assets).values() if isinstance(value, torch.Tensor))),
        "first_ot_cost": float(score_first["ot_cost"][0].item()),
        "last_ot_cost": float(score_last["ot_cost"][0].item()),
        "first_aligned_topk_cost": float(score_first["aligned_topk_cost"][0].item()),
        "last_aligned_topk_cost": float(score_last["aligned_topk_cost"][0].item()),
    }
    args.output.mkdir(parents=True)
    torch.save(
        {
            "format": "xvla_stackcube_tokenwise_pca_ot_asset_v1",
            "checkpoint": str(args.checkpoint),
            "metadata": str(args.metadata),
            "view_index": args.view_index,
            "asset": asset_state_dict(assets[0]),
        },
        args.output / "tokenwise_ot_asset.pt",
    )
    (args.output / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "SMOKE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
