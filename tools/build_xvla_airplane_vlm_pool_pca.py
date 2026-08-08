#!/usr/bin/env python3
"""Fit VLM-to-action-transformer pooled PCA from ID expert observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "RLinf")]

from rlinf.algorithms.vla_fail import fit_pca_residual_statistics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    sys.path.insert(0, str(args.xvla_root.resolve()))
    from datasets.dataset import InfiniteDataReader
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    model = XVLA.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).cuda().eval()
    processor = XVLAProcessor.from_pretrained(args.checkpoint)
    dataset = InfiniteDataReader(
        metas_path=str(args.metadata),
        num_actions=model.num_actions,
        num_views=2,
        training=False,
        action_mode=model.action_mode,
    )
    # IterableDataset workers each traverse the full finite evaluation set.
    # A single worker guarantees every legal training anchor is counted once.
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    features: list[torch.Tensor] = []
    observations = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            language = processor.encode_language(batch.pop("language_instruction"))
            inputs = {**batch, **language}
            inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items()}
            encoding = model.forward_vlm(
                inputs["input_ids"], inputs["image_input"], inputs["image_mask"]
            )
            tokens = torch.cat(
                [encoding["vlm_features"], encoding["aux_visual_inputs"]], dim=1
            )
            features.append(tokens.float().mean(dim=1).cpu())
            observations += tokens.shape[0]
            if observations % 512 < tokens.shape[0]:
                print(f"[xvla-pca] observations={observations}", flush=True)

    matrix = torch.cat(features, dim=0).unsqueeze(1)
    statistics = fit_pca_residual_statistics(matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "xvla_airplane_vlm_input_pool_pca_v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "metadata": str(args.metadata.resolve()),
            "feature": "mean(concat(vlm_features,aux_visual_inputs))",
            "num_observations": matrix.shape[0],
            "hidden_dim": matrix.shape[-1],
            "statistics": statistics.state_dict(),
        },
        args.output,
    )
    manifest = {
        "asset": str(args.output),
        "num_observations": matrix.shape[0],
        "hidden_dim": matrix.shape[-1],
        "principal_dim": statistics.principal_dim,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
