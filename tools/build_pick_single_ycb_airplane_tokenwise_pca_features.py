#!/usr/bin/env python3
"""Extract durable ID-only prefix/Bridge token shards for the airplane task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.data.datasets.recap.utils import decode_image_struct_batch  # noqa: E402
from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import PICK_SINGLE_YCB_AIRPLANE_TASK  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402
from tools.pick_single_ycb_airplane_tokenwise_pca import (  # noqa: E402
    FORMAT,
    SOURCE_NAMES,
    lerobot_sample_to_policy_observation,
    sha256,
    sha256_path,
)


def _load_dataset(dataset_root: Path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(dataset_root.name, root=dataset_root, download_videos=False)
    dataset.hf_dataset.set_transform(decode_image_struct_batch)
    return dataset


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--expected-observations", type=int, default=9109)
    parser.add_argument("--max-observations", type=int, default=0, help="positive value is a smoke-only cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite feature cache: {args.output_dir}")
    if args.shard_size < 1 or args.expected_observations < 1 or args.max_observations < 0:
        raise ValueError("shard size, expected observations, and max observations are invalid")
    dataset = _load_dataset(args.dataset_root)
    indices = list(range(len(dataset)))
    if args.max_observations:
        indices = indices[: args.max_observations]
    elif len(indices) != args.expected_observations:
        raise ValueError(
            f"expected exactly {args.expected_observations} frozen ID observations, found {len(indices)}; "
            "use --max-observations only for a smoke run"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    shard_dir = args.output_dir / "shards"
    shard_dir.mkdir()
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    shards: list[dict[str, Any]] = []
    source_ids: torch.Tensor | None = None
    source_names: tuple[str, ...] | None = None
    try:
        for start in range(0, len(indices), args.shard_size):
            batch_indices = indices[start : start + args.shard_size]
            vlm_inputs: list[torch.Tensor] = []
            bridges: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            for index in batch_indices:
                with torch.inference_mode():
                    probe = model.extract_prefix_probes_from_observation(
                        lerobot_sample_to_policy_observation(
                            dataset[index], task_description=PICK_SINGLE_YCB_AIRPLANE_TASK
                        )
                    )
                current_ids = torch.as_tensor(probe["source_ids"], dtype=torch.int64).cpu()
                current_names = tuple(probe["source_names"])
                if source_ids is None:
                    source_ids, source_names = current_ids, current_names
                    if source_names != SOURCE_NAMES:
                        raise RuntimeError(f"unexpected prefix source names: {source_names}")
                elif not torch.equal(source_ids, current_ids) or source_names != current_names:
                    raise RuntimeError("prefix token source order changed across ID observations")
                vlm_inputs.append(torch.as_tensor(probe["vlm_input"])[0].detach().cpu())
                bridges.append(torch.as_tensor(probe["bridge"])[0].detach().cpu())
                masks.append(torch.as_tensor(probe["valid_mask"], dtype=torch.bool)[0].detach().cpu())
            payload = {
                "format": "pick_single_ycb_airplane_prefix_feature_shard_v1",
                "indices": batch_indices,
                "vlm_input": torch.stack(vlm_inputs),
                "bridge": torch.stack(bridges),
                "valid_mask": torch.stack(masks),
                "source_ids": source_ids,
                "source_names": source_names,
            }
            path = shard_dir / f"id_prefix_{start:06d}_{start + len(batch_indices):06d}.pt"
            torch.save(payload, path)
            shards.append({"file": str(path.relative_to(args.output_dir)), "sha256": sha256(path), "observations": len(batch_indices)})
            print(f"[airplane-prefix-cache] observations={start + len(batch_indices)}/{len(indices)}", flush=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if source_ids is None or source_names is None:
        raise RuntimeError("prefix extraction produced no observations")
    manifest = {
        "format": "pick_single_ycb_airplane_prefix_feature_cache_v1",
        "task": "pick_single_ycb_airplane",
        "detector_format": FORMAT,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint / "full_weights.pt") if (args.checkpoint / "full_weights.pt").is_file() else None,
        "pi05_base": str(args.pi05_base),
        "norm_stats": str(args.norm_stats),
        "norm_stats_sha256": sha256_path(args.norm_stats),
        "dataset_root": str(args.dataset_root),
        "dataset_episodes_sha256": sha256(args.dataset_root / "meta" / "episodes.jsonl"),
        "num_observations": len(indices),
        "source_ids": source_ids.tolist(),
        "source_names": list(source_names),
        "prefix_tokens": int(source_ids.numel()),
        "hidden_dim": int(torch.load(args.output_dir / shards[0]["file"], map_location="cpu", weights_only=False)["bridge"].shape[-1]),
        "storage_dtype": str(torch.load(args.output_dir / shards[0]["file"], map_location="cpu", weights_only=False)["bridge"].dtype),
        "shards": shards,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    _write_json(args.output_dir / "feature_manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "shards"}, indent=2))


if __name__ == "__main__":
    main()
