#!/usr/bin/env python3
"""Fit all X-VLA internal detector references from Panda ID expert anchors."""

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

from tools.xvla_airplane_failure_detection import (  # noqa: E402
    XVLAMultilayerProbe,
    fit_layer_asset,
    layer_names,
)


TASK = "put the vegetable into the yellow basket"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain-id", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-6)
    return parser.parse_args()


def read_image(value: np.ndarray | bytes) -> Image.Image:
    array = np.asarray(value)
    if array.ndim == 1:
        return Image.open(BytesIO(bytes(array))).convert("RGB")
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB image, got {array.shape}")
    return Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB")


def encode_batch(processor, rows: list[tuple[Image.Image, np.ndarray]], device: torch.device, domain_id: int):
    encoded = processor.encode_image([[image] for image, _ in rows])
    language = processor.encode_language([TASK] * len(rows))
    proprio = np.zeros((len(rows), 20), dtype=np.float32)
    for index, (_, value) in enumerate(rows):
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.size < 10:
            raise ValueError(f"Panda proprio must have 10 values, got {value.shape}")
        proprio[index, :10] = value[:10]
    inputs = {
        **encoded,
        "input_ids": language["input_ids"],
        "domain_id": torch.full((len(rows),), domain_id, dtype=torch.long),
        "proprio": torch.from_numpy(proprio),
    }
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    files = sorted((args.dataset / "data").glob("episode_*.h5"))
    if not files:
        raise FileNotFoundError(f"no HDF5 episodes under {args.dataset / 'data'}")
    args.output_dir.mkdir(parents=True)
    feature_dir = args.output_dir / "feature_cache"
    feature_dir.mkdir()

    sys.path.insert(0, str(args.xvla_root.resolve()))
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    device = torch.device("cuda")
    model = XVLA.from_pretrained(str(args.checkpoint), torch_dtype=torch.bfloat16).to(device).eval()
    processor = XVLAProcessor.from_pretrained(str(args.checkpoint))
    probe = XVLAMultilayerProbe(model, probe_seed=args.probe_seed, probe_steps=args.probe_steps)
    names = layer_names(probe.vlm_layer_count, probe.action_layer_count)
    parts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    pending: list[tuple[Image.Image, np.ndarray]] = []
    observations = 0

    def flush() -> None:
        nonlocal observations
        if not pending:
            return
        inputs = encode_batch(processor, pending, device, args.domain_id)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            features, _ = probe.extract(inputs)
        for name in names:
            parts[name].append(features[name].detach().float().cpu())
        observations += len(pending)
        pending.clear()
        if observations % 256 < args.batch_size:
            print(f"[panda-detector-assets] observations={observations}", flush=True)

    try:
        for path in files:
            with h5py.File(path, "r") as h5:
                images = h5["images"]
                proprio = np.asarray(h5["proprio"], dtype=np.float32)
                actions = np.asarray(h5["abs_action_6d"], dtype=np.float32)
                count = min(len(images), len(actions), len(proprio))
                if count < 1:
                    raise ValueError(f"empty episode {path}")
                for index in range(count):
                    pending.append((read_image(images[index]), proprio[index]))
                    if len(pending) >= args.batch_size:
                        flush()
        flush()
    finally:
        probe.close()

    layers: dict[str, dict] = {}
    for index, name in enumerate(names, start=1):
        values = torch.cat(parts[name], dim=0)
        np.save(feature_dir / f"{name}.npy", values.numpy().astype(np.float16))
        print(
            f"[panda-detector-fit] layer={index}/{len(names)} name={name} "
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
        "format": "xvla_panda_vegetable_basket_detector_assets_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "fit_split": "ID expert anchors only",
        "num_observations": observations,
        "vlm_layers": probe.vlm_layer_count,
        "action_layers": probe.action_layer_count,
        "layer_names": names,
        "pca_dim": args.pca_dim,
        "ridge": args.ridge,
        "knn_k": 10,
        "feature_definition": {
            "vlm_input_pool": "attention-mask pooled VLM inputs before encoder block 1",
            "vlm": "mean pooled output of each VLM encoder block",
            "bridge": "mean pooled VLM features concatenated with auxiliary visual inputs",
            "action": "mean pooled action tokens after each action-transformer block under fixed flow probe",
            "probe_seed": args.probe_seed,
            "probe_steps": args.probe_steps,
        },
        "layers": layers,
    }
    asset_path = args.output_dir / "multilayer_detector_assets.pt"
    torch.save(payload, asset_path)
    manifest = {key: value for key, value in payload.items() if key != "layers"}
    manifest["asset_path"] = str(asset_path)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "DETECTOR_ASSETS_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
