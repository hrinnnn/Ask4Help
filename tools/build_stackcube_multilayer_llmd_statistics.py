#!/usr/bin/env python3
"""Build reusable multi-layer π0.5 features and LLMD assets for StackCube.

The collector executes one fixed-prior forward pass per SFT observation and
captures three intermediate Action Expert blocks plus two VLM-side features.
The existing final Action Expert VLA-FAIL asset is intentionally not refit;
we only prove that the new collector returns the identical final feature on a
small deterministic audit subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.algorithms.vla_fail import fixed_gaussian_prior, fit_llmd_statistics  # noqa: E402
from tools.build_stackcube_vla_fail_statistics import (  # noqa: E402
    _load_dataset,
    _sha256,
    lerobot_sample_to_env_obs,
)
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty score list")
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _subset_indices(total: int, maximum: int) -> list[int]:
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
    parser.add_argument(
        "--final-baseline-statistics",
        type=Path,
        required=True,
        help="Existing strict final-layer VLA-FAIL statistics asset; never overwritten.",
    )
    parser.add_argument("--fixed-prior-seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--max-observations", type=int, default=0)
    parser.add_argument(
        "--final-parity-observations",
        type=int,
        default=5,
        help="Number of ID observations used to prove final collector equivalence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if not args.final_baseline_statistics.is_file():
        raise FileNotFoundError(args.final_baseline_statistics)
    if args.final_parity_observations < 1:
        raise ValueError("final-parity-observations must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    statistics_path = args.output_dir / "multilayer_statistics.pt"
    cache_path = args.output_dir / "multilayer_feature_cache.pt"
    manifest_path = args.output_dir / "multilayer_statistics.json"
    if statistics_path.exists() or cache_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "refusing to overwrite a multi-layer asset; choose a fresh output-dir: "
            f"{args.output_dir}"
        )

    final_payload = torch.load(args.final_baseline_statistics, map_location="cpu", weights_only=False)
    if final_payload.get("format") != "vla_fail_llmd_statistics_v1":
        raise ValueError("final-baseline-statistics is not the strict VLA-FAIL final-layer asset")
    if int(final_payload["fixed_prior_seed"]) != args.fixed_prior_seed:
        raise ValueError(
            "fixed-prior seed differs from the existing final baseline: "
            f"baseline={final_payload['fixed_prior_seed']}, requested={args.fixed_prior_seed}"
        )

    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    dataset = _load_dataset(args.dataset_root)
    indices = _subset_indices(len(dataset), args.max_observations)
    prior = fixed_gaussian_prior(
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
        seed=args.fixed_prior_seed,
        device="cuda",
    )
    cached_features: dict[str, list[torch.Tensor]] = {}
    parity_max_abs_errors: list[float] = []
    layer_schema: dict[str, dict[str, Any]] = {}
    try:
        for position, index in enumerate(indices, start=1):
            env_obs = lerobot_sample_to_env_obs(dataset[index])
            audit_final = position <= args.final_parity_observations
            with torch.inference_mode():
                features = model.extract_multilayer_llmd_features(
                    env_obs,
                    prior,
                    include_action_expert_final=audit_final,
                )
                if audit_final:
                    strict_final = model.extract_llmd_action_features(env_obs, prior)
                    candidate_final = features.pop("action_expert_final")
                    if candidate_final.shape != strict_final.shape:
                        raise RuntimeError(
                            "multi-layer final feature has wrong shape: "
                            f"new={tuple(candidate_final.shape)}, old={tuple(strict_final.shape)}"
                        )
                    parity_max_abs_errors.append(
                        float((candidate_final - strict_final).abs().max().item())
                    )
                    torch.testing.assert_close(candidate_final, strict_final, rtol=0.0, atol=0.0)
            for name, feature in features.items():
                if feature.ndim != 3 or feature.shape[0] != 1:
                    raise RuntimeError(f"{name} must have shape [1,T,D], got {tuple(feature.shape)}")
                layer_schema.setdefault(
                    name,
                    {
                        "feature_shape": list(feature.shape[1:]),
                        "feature_semantics": (
                            "token-wise Action Expert block output" if name.startswith("action_expert")
                            else "valid-prefix mean pooled VLM representation"
                        ),
                    },
                )
                if layer_schema[name]["feature_shape"] != list(feature.shape[1:]):
                    raise RuntimeError(f"feature shape changed across observations for {name}")
                cached_features.setdefault(name, []).append(feature.detach().cpu().to(torch.float16))
            if position % 100 == 0 or position == len(indices):
                print(f"[multilayer-llmd] observations={position}/{len(indices)}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()

    statistics_payload: dict[str, Any] = {}
    cache_payload: dict[str, torch.Tensor] = {}
    for name, values in cached_features.items():
        bank = torch.cat(values, dim=0)
        cache_payload[name] = bank
        statistics = fit_llmd_statistics(bank, ridge=args.ridge)
        statistics_payload[name] = {
            "statistics": statistics.state_dict(),
            "feature_shape": list(bank.shape[1:]),
            "feature_semantics": layer_schema[name]["feature_semantics"],
        }

    payload = {
        "format": "stackcube_multilayer_llmd_statistics_v1",
        "task": "stackcube",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_dataset_observations": len(indices),
        "fixed_prior": prior.cpu(),
        "fixed_prior_seed": args.fixed_prior_seed,
        "pi05_prior_timestep": 1.0,
        "ridge": args.ridge,
        "layers": statistics_payload,
        "final_baseline_statistics": str(args.final_baseline_statistics),
        "final_baseline_statistics_sha256": _sha256(args.final_baseline_statistics),
        "final_parity_max_abs_error": _summary(parity_max_abs_errors),
    }
    torch.save(payload, statistics_path)
    torch.save(
        {
            "format": "stackcube_multilayer_llmd_feature_cache_v1",
            "statistics_sha256": _sha256(statistics_path),
            "checkpoint": str(args.checkpoint),
            "dataset_root": str(args.dataset_root),
            "indices": indices,
            "layers": cache_payload,
        },
        cache_path,
    )
    manifest = {
        **{key: value for key, value in payload.items() if key not in {"fixed_prior", "layers"}},
        "statistics_path": str(statistics_path),
        "statistics_sha256": _sha256(statistics_path),
        "feature_cache_path": str(cache_path),
        "feature_cache_sha256": _sha256(cache_path),
        "layers": {
            name: {
                **layer_schema[name],
                "num_observations": int(cache_payload[name].shape[0]),
            }
            for name in cache_payload
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
