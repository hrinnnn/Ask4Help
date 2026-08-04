#!/usr/bin/env python3
"""Build threshold-free detector reference assets for the airplane policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    fit_knn_statistics,
    fit_llmd_statistics,
    fit_pca_residual_statistics,
    fixed_gaussian_prior,
)
from rlinf.envs.maniskill.pick_single_ycb_airplane_variants import (  # noqa: E402
    PICK_SINGLE_YCB_AIRPLANE_TASK,
)
from tools.build_stackcube_vla_fail_statistics import (  # noqa: E402
    _as_hwc_batch,
    _load_dataset,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sample_to_env_obs(sample: dict[str, Any]) -> dict[str, Any]:
    main = _as_hwc_batch(sample["image"])
    wrist = _as_hwc_batch(sample["wrist_image"])
    state = torch.as_tensor(sample["state"])
    if state.ndim == 1:
        state = state.unsqueeze(0)
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "states": state,
        "task_descriptions": [PICK_SINGLE_YCB_AIRPLANE_TASK] * state.shape[0],
        "task_ids": torch.zeros(state.shape[0], dtype=torch.long),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-prior-seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    args.output_dir.mkdir(parents=True)

    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    dataset = _load_dataset(args.dataset_root)
    indices = list(range(len(dataset)))
    if args.max_observations and args.max_observations < len(indices):
        indices = torch.linspace(0, len(indices) - 1, args.max_observations).round().long().tolist()
    prior = fixed_gaussian_prior(
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
        seed=args.fixed_prior_seed,
        device="cuda",
    )
    bridge_rows: list[torch.Tensor] = []
    final_rows: list[torch.Tensor] = []
    try:
        for position, index in enumerate(indices, start=1):
            with torch.inference_mode():
                features = model.extract_multilayer_llmd_features(
                    sample_to_env_obs(dataset[index]), prior, include_action_expert_final=True
                )
            bridge_rows.append(features["vlm_bridge_final_mean"].detach().cpu().to(torch.float16))
            final_rows.append(features["action_expert_final"].detach().cpu().to(torch.float16))
            if position % 50 == 0 or position == len(indices):
                print(f"[airplane-assets] observations={position}/{len(indices)}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()

    bridge = torch.cat(bridge_rows, dim=0)
    final = torch.cat(final_rows, dim=0)
    payload = {
        "format": "pick_single_ycb_airplane_detector_assets_v1",
        "checkpoint": str(args.checkpoint),
        "norm_stats": str(args.norm_stats),
        "dataset_root": str(args.dataset_root),
        "num_id_observations": len(indices),
        "fixed_prior_seed": args.fixed_prior_seed,
        "fixed_prior": prior.cpu(),
        "feature_semantics": {
            "bridge": "mean valid final VLM prefix tokens passed to the Action Expert",
            "action_expert_final": "final Action Expert hidden state for ten action tokens",
        },
        "statistics": {
            "bridge_llmd": fit_llmd_statistics(bridge.float(), ridge=args.ridge).state_dict(),
            "bridge_deep_knn": fit_knn_statistics(bridge.float(), k=args.knn_k).state_dict(),
            "bridge_pca_residual": fit_pca_residual_statistics(bridge.float()).state_dict(),
            "final_llmd": fit_llmd_statistics(final.float(), ridge=args.ridge).state_dict(),
        },
        "knn_k": args.knn_k,
    }
    asset_path = args.output_dir / "detector_assets.pt"
    cache_path = args.output_dir / "feature_cache.pt"
    torch.save(payload, asset_path)
    torch.save(
        {
            "format": "pick_single_ycb_airplane_feature_cache_v1",
            "indices": indices,
            "bridge": bridge,
            "action_expert_final": final,
        },
        cache_path,
    )
    manifest = {
        **{key: value for key, value in payload.items() if key not in {"fixed_prior", "statistics"}},
        "asset_path": str(asset_path),
        "asset_sha256": sha256(asset_path),
        "feature_cache_path": str(cache_path),
        "feature_cache_sha256": sha256(cache_path),
        "feature_shapes": {"bridge": list(bridge.shape), "action_expert_final": list(final.shape)},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
