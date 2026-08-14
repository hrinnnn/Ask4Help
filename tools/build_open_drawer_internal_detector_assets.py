#!/usr/bin/env python3
"""Build ID-only pi0.5 internal detector assets for OpenDrawerRetrievePlace."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = Path(os.environ.get("ASK4HELP_RLINF_ROOT", ROOT / "RLinf"))
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import (  # noqa: E402
    fit_knn_statistics,
    fit_llmd_statistics,
    fit_pca_residual_statistics,
    fixed_gaussian_prior,
)
from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import TASK_INSTRUCTION  # noqa: E402
from tools.build_stackcube_vla_fail_statistics import _as_hwc_batch, _load_dataset  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def _sample_to_env_obs(sample: dict[str, Any]) -> dict[str, Any]:
    main = _as_hwc_batch(sample["image"])
    wrist = _as_hwc_batch(sample["wrist_image"])
    state = torch.as_tensor(sample["state"])
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2:
        raise ValueError(f"expected state [B,D], got {tuple(state.shape)}")
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "states": state,
        "task_descriptions": [TASK_INSTRUCTION] * state.shape[0],
        "task_ids": torch.zeros(state.shape[0], dtype=torch.long),
    }


def _indices(total: int, maximum: int) -> list[int]:
    if maximum < 0:
        raise ValueError("max-observations must be non-negative")
    if maximum == 0 or maximum >= total:
        return list(range(total))
    return torch.linspace(0, total - 1, maximum).round().long().tolist()


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
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite detector assets: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    dataset = _load_dataset(args.dataset_root)
    indices = _indices(len(dataset), args.max_observations)
    prior = fixed_gaussian_prior(
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
        seed=args.fixed_prior_seed,
        device="cuda",
    )
    cache: dict[str, list[torch.Tensor]] = {}
    layer_info: dict[str, dict[str, Any]] = {}
    try:
        for position, index in enumerate(indices, start=1):
            env_obs = _sample_to_env_obs(dataset[index])
            with torch.inference_mode():
                features = model.extract_multilayer_llmd_features(
                    env_obs, prior, include_action_expert_final=True
                )
            for name, feature in features.items():
                if feature.ndim != 3 or feature.shape[0] != 1:
                    raise RuntimeError(f"{name} has unexpected shape {tuple(feature.shape)}")
                shape = list(feature.shape[1:])
                if name in layer_info and layer_info[name]["feature_shape"] != shape:
                    raise RuntimeError(f"feature shape changed for {name}")
                layer_info.setdefault(
                    name,
                    {
                        "feature_shape": shape,
                        "semantics": (
                            "Action Expert token features"
                            if name.startswith("action_expert")
                            else "mean pooled VLM/bridge features"
                        ),
                    },
                )
                cache.setdefault(name, []).append(feature.detach().cpu().to(torch.float32))
            if position % 100 == 0 or position == len(indices):
                print(f"[open-drawer-assets] observations={position}/{len(indices)}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()

    feature_cache = {name: torch.cat(values, dim=0) for name, values in cache.items()}
    statistics: dict[str, Any] = {}
    detectors: dict[str, dict[str, Any]] = {}
    for name, bank in feature_cache.items():
        llmd = fit_llmd_statistics(bank, ridge=args.ridge)
        statistics[name] = {
            "statistics": llmd.state_dict(),
            "feature_shape": list(bank.shape[1:]),
            "semantics": layer_info[name]["semantics"],
        }
        detectors[f"{name}__llmd"] = {
            "kind": "llmd", "layer": name, "statistics": llmd.state_dict()
        }
        knn = fit_knn_statistics(bank, k=args.knn_k)
        detectors[f"{name}__knn_k{args.knn_k}"] = {
            "kind": "knn", "layer": name, "statistics": knn.state_dict()
        }
        pca = fit_pca_residual_statistics(bank)
        detectors[f"{name}__pca_residual"] = {
            "kind": "pca_residual", "layer": name, "statistics": pca.state_dict()
        }

    stats_payload = {
        "format": "open_drawer_multilayer_llmd_statistics_v1",
        "task": "OpenDrawerRetrievePlace",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_id_observations": len(indices),
        "fixed_prior": prior.cpu(),
        "fixed_prior_seed": args.fixed_prior_seed,
        "ridge": args.ridge,
        "layers": statistics,
    }
    cache_payload = {
        "format": "open_drawer_multilayer_feature_cache_v1",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "indices": indices,
        "layers": feature_cache,
    }
    asset_payload = {
        "format": "open_drawer_internal_detector_assets_v1",
        "task": "OpenDrawerRetrievePlace",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_id_observations": len(indices),
        "fixed_prior": prior.cpu(),
        "fixed_prior_seed": args.fixed_prior_seed,
        "candidate_layers": list(feature_cache),
        "knn_k": args.knn_k,
        "detectors": detectors,
    }
    torch.save(stats_payload, args.output_dir / "multilayer_statistics.pt")
    torch.save(cache_payload, args.output_dir / "feature_cache.pt")
    torch.save(asset_payload, args.output_dir / "detector_assets.pt")
    manifest = {
        "format": asset_payload["format"],
        "task": asset_payload["task"],
        "checkpoint": asset_payload["checkpoint"],
        "dataset_root": asset_payload["dataset_root"],
        "num_id_observations": len(indices),
        "fixed_prior_seed": args.fixed_prior_seed,
        "candidate_layers": list(feature_cache),
        "detector_names": list(detectors),
        "statistics_path": str(args.output_dir / "multilayer_statistics.pt"),
        "feature_cache_path": str(args.output_dir / "feature_cache.pt"),
        "detector_assets_path": str(args.output_dir / "detector_assets.pt"),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
