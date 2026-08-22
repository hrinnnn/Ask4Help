#!/usr/bin/env python3
"""Build ID-only internal-feature detector assets for object variation."""

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

from rlinf.algorithms.vla_fail import (  # noqa: E402
    fit_knn_statistics,
    fit_llmd_statistics,
    fit_pca_residual_statistics,
    fixed_gaussian_prior,
    resolve_feature_probe_indices,
)
from rlinf.envs.maniskill.pick_single_ycb_object_variation import PICK_SINGLE_YCB_OBJECT_TASK  # noqa: E402
from tools.build_stackcube_vla_fail_statistics import _as_hwc_batch, _load_dataset  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


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
        "task_descriptions": [PICK_SINGLE_YCB_OBJECT_TASK] * state.shape[0],
        "task_ids": torch.zeros(state.shape[0], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    dataset = _load_dataset(args.dataset_root)
    indices = list(range(len(dataset)))
    prior = fixed_gaussian_prior(
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
        seed=0,
        device="cuda",
    )
    bridge_rows: list[torch.Tensor] = []
    final_rows: list[torch.Tensor] = []
    action_mid_rows: list[torch.Tensor] = []
    vlm_mid_rows: list[torch.Tensor] = []
    action_mid_index = resolve_feature_probe_indices(
        len(model.paligemma_with_expert.gemma_expert.model.layers), (0.5,)
    )[0]
    vlm_mid_index = resolve_feature_probe_indices(
        len(model.paligemma_with_expert.paligemma.language_model.layers), (0.5,)
    )[0]
    for position, index in enumerate(indices, start=1):
        with torch.inference_mode():
            features = model.extract_multilayer_llmd_features(
                sample_to_env_obs(dataset[index]),
                prior,
                action_expert_fractions=(0.5,),
                capture_vlm=True,
                include_action_expert_final=True,
            )
        bridge_rows.append(features["vlm_bridge_final_mean"].detach().cpu().to(torch.float16))
        final_rows.append(features["action_expert_final"].detach().cpu().to(torch.float16))
        action_mid_rows.append(features[f"action_expert_block_{action_mid_index:02d}"].detach().cpu().to(torch.float16))
        vlm_mid_rows.append(features[f"vlm_block_{vlm_mid_index:02d}_mean"].detach().cpu().to(torch.float16))
        if position % 100 == 0 or position == len(indices):
            print(f"[object-variation-assets] observations={position}/{len(indices)}", flush=True)

    bridge = torch.cat(bridge_rows, dim=0)
    final = torch.cat(final_rows, dim=0)
    action_mid = torch.cat(action_mid_rows, dim=0)
    vlm_mid = torch.cat(vlm_mid_rows, dim=0)
    payload = {
        "format": "pick_single_ycb_object_variation_detector_assets_v1",
        "checkpoint": str(args.checkpoint),
        "norm_stats": str(args.norm_stats),
        "dataset_root": str(args.dataset_root),
        "num_id_observations": len(indices),
        "statistics": {
            "bridge_llmd": fit_llmd_statistics(bridge.float()).state_dict(),
            "bridge_deep_knn": fit_knn_statistics(bridge.float(), k=args.knn_k).state_dict(),
            "bridge_pca_residual": fit_pca_residual_statistics(bridge.float()).state_dict(),
            "final_llmd": fit_llmd_statistics(final.float()).state_dict(),
            "action_expert_50_pca": fit_pca_residual_statistics(action_mid.float()).state_dict(),
            "action_expert_final_pca": fit_pca_residual_statistics(final.float()).state_dict(),
            "vlm_50_pca": fit_pca_residual_statistics(vlm_mid.float()).state_dict(),
        },
        "fixed_prior": prior.cpu(),
        "layer_indices": {"action_expert_50": action_mid_index, "vlm_50": vlm_mid_index},
        "feature_semantics": {
            "bridge": "mean valid final VLM prefix tokens passed to Action Expert",
            "action_expert_final": "final Action Expert hidden state",
            "action_expert_50": "mid Action Expert hidden state",
            "vlm_50": "mid VLM prefix representation",
        },
    }
    torch.save(payload, args.output_dir / "detector_assets.pt")
    torch.save(
        {"format": "pick_single_ycb_object_variation_feature_cache_v1", "indices": indices, "bridge": bridge, "action_expert_final": final, "action_expert_50": action_mid, "vlm_50": vlm_mid},
        args.output_dir / "feature_cache.pt",
    )
    manifest = {
        "format": payload["format"],
        "checkpoint": payload["checkpoint"],
        "norm_stats": payload["norm_stats"],
        "dataset_root": payload["dataset_root"],
        "num_id_observations": len(indices),
        "asset_path": str(args.output_dir / "detector_assets.pt"),
        "feature_cache_path": str(args.output_dir / "feature_cache.pt"),
        "feature_shapes": {"bridge": list(bridge.shape), "action_expert_final": list(final.shape), "action_expert_50": list(action_mid.shape), "vlm_50": list(vlm_mid.shape)},
        "ood_or_failure_data_used": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "ASSETS_COMPLETE").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

