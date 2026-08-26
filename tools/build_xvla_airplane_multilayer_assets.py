#!/usr/bin/env python3
"""Build every-layer X-VLA failure-detector assets from ID expert anchors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.xvla_airplane_failure_detection import (  # noqa: E402
    XVLAMultilayerProbe,
    fit_layer_asset,
    layer_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    feature_dir = args.output_dir / "feature_cache"
    feature_dir.mkdir()

    # Keep the Ask4Help root first so ``tools.xvla_panda_airplane_domain_handler``
    # resolves to this repository rather than an unrelated X-VLA tools package.
    sys.path.insert(1, str(args.xvla_root.resolve()))
    from datasets.dataset import InfiniteDataReader
    from tools.xvla_panda_airplane_domain_handler import install_panda_airplane_handler
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    # The metadata explicitly names ``panda_airplane``.  Register the same
    # handler used by the training shim before InfiniteDataReader iterates;
    # otherwise the first batch fails before any feature asset is written.
    install_panda_airplane_handler()
    from datasets.domain_handler import registry
    if "panda_airplane" not in registry._REGISTRY:
        raise RuntimeError("panda_airplane handler registration failed before asset build")

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
    probe = XVLAMultilayerProbe(
        model, probe_seed=args.probe_seed, probe_steps=args.probe_steps
    )
    names = layer_names(probe.vlm_layer_count, probe.action_layer_count)
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    observations = 0
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for batch in loader:
                language = processor.encode_language(batch.pop("language_instruction"))
                inputs = {**batch, **language}
                inputs = {
                    key: value.to(device, non_blocking=True)
                    for key, value in inputs.items()
                    if isinstance(value, torch.Tensor)
                }
                features, _encoding = probe.extract(inputs)
                for name in names:
                    parts[name].append(features[name].detach().cpu().to(torch.float16))
                observations += inputs["input_ids"].shape[0]
                if observations % 256 < inputs["input_ids"].shape[0]:
                    print(f"[xvla-multilayer-cache] observations={observations}", flush=True)
    finally:
        probe.close()

    layers: dict[str, dict] = {}
    for index, name in enumerate(names, start=1):
        values = torch.cat(parts.pop(name), dim=0)
        np.save(feature_dir / f"{name}.npy", values.numpy())
        print(
            f"[xvla-multilayer-fit] layer={index}/{len(names)} name={name} "
            f"observations={values.shape[0]}",
            flush=True,
        )
        layers[name] = fit_layer_asset(
            values.to(device=device, dtype=torch.float32),
            pca_dim=args.pca_dim,
            ridge=args.ridge,
        )
        torch.cuda.empty_cache()

    payload = {
        "format": "xvla_airplane_multilayer_detector_assets_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "metadata": str(args.metadata.resolve()),
        "fit_split": "ID expert anchors only",
        "feature_definition": {
            "vlm_input_pool": "attention-mask pooled Florence2 merged image-language embeddings before encoder block 1",
            "vlm": "mean pooled output of every Florence2 language encoder block",
            "bridge": "mean(concat(vlm_features, aux_visual_inputs))",
            "action": "mean of action tokens after every block under fixed flow probe",
            "probe_seed": args.probe_seed,
            "probe_steps": args.probe_steps,
        },
        "num_observations": observations,
        "hidden_dim": int(next(iter(layers.values()))["mean"].numel()),
        "vlm_layers": probe.vlm_layer_count,
        "action_layers": probe.action_layer_count,
        "pca_dim": args.pca_dim,
        "ridge": args.ridge,
        "knn_k": 10,
        "layers": layers,
    }
    asset_path = args.output_dir / "multilayer_detector_assets.pt"
    torch.save(payload, asset_path)
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"layers"}
    }
    manifest["layer_names"] = names
    manifest["asset_path"] = str(asset_path)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
