#!/usr/bin/env python3
"""Fit strict VLA-FAIL LLMD statistics from StackCube SFT demonstrations.

This reads the same local LeRobot demonstrations used for SFT.  It does not
use rollout outcomes or failure labels: every training observation contributes
one final-Action-Expert feature vector per action token under one globally
fixed Gaussian action prior.
"""

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

from rlinf.algorithms.vla_fail import fixed_gaussian_prior, fit_llmd_statistics  # noqa: E402
from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK  # noqa: E402
from tools.maniskill_pi05_vfd_online_awbc import _load_model  # noqa: E402


def _as_hwc_batch(value: Any) -> torch.Tensor:
    image = torch.as_tensor(value)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"expected image [H,W,C] or [B,H,W,C], got {tuple(image.shape)}")
    if image.shape[1] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = image.permute(0, 2, 3, 1)
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"cannot identify image channel dimension in {tuple(image.shape)}")
    return image.contiguous()


def lerobot_sample_to_env_obs(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert the native StackCube LeRobot row to the policy's rollout input."""

    try:
        main = _as_hwc_batch(sample["image"])
        wrist = _as_hwc_batch(sample["wrist_image"])
        state = torch.as_tensor(sample["state"])
    except KeyError as exc:
        raise KeyError(
            "StackCube SFT dataset must expose image, wrist_image, and state fields"
        ) from exc
    if state.ndim == 1:
        state = state.unsqueeze(0)
    if state.ndim != 2:
        raise ValueError(f"expected state [D] or [B,D], got {tuple(state.shape)}")
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "states": state,
        "task_descriptions": [STACK_CUBE_TASK] * state.shape[0],
        "task_ids": torch.zeros(state.shape[0], dtype=torch.long),
    }


def _load_dataset(dataset_root: Path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from rlinf.data.datasets.recap.utils import decode_image_struct_batch

    dataset = LeRobotDataset(dataset_root.name, root=dataset_root, download_videos=False)
    dataset.hf_dataset.set_transform(decode_image_struct_batch)
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-prior-seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--max-observations",
        type=int,
        default=0,
        help="0 uses every SFT observation; a positive value uniformly subsamples for a smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    if args.max_observations < 0:
        raise ValueError("max-observations must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.checkpoint, args.norm_stats, args.pi05_base)
    dataset = _load_dataset(args.dataset_root)
    indices = list(range(len(dataset)))
    if args.max_observations and args.max_observations < len(indices):
        indices = torch.linspace(0, len(dataset) - 1, args.max_observations).round().long().tolist()
    prior = fixed_gaussian_prior(
        action_horizon=int(model.config.action_horizon),
        action_dim=int(model.config.action_dim),
        seed=args.fixed_prior_seed,
        device="cuda",
    )
    features = []
    try:
        for position, index in enumerate(indices, start=1):
            env_obs = lerobot_sample_to_env_obs(dataset[index])
            with torch.inference_mode():
                feature = model.extract_llmd_action_features(env_obs, prior)
            features.append(feature.detach().cpu())
            if position % 100 == 0 or position == len(indices):
                print(f"[llmd-stats] observations={position}/{len(indices)}", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    statistics = fit_llmd_statistics(torch.cat(features, dim=0), ridge=args.ridge)
    payload = {
        "format": "vla_fail_llmd_statistics_v1",
        "task": "stackcube",
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "num_dataset_observations": len(indices),
        "fixed_prior": prior.cpu(),
        "fixed_prior_seed": args.fixed_prior_seed,
        "pi05_prior_timestep": 1.0,
        "statistics": statistics.state_dict(),
    }
    torch.save(payload, args.output)
    manifest = {
        **{key: value for key, value in payload.items() if key not in {"fixed_prior", "statistics"}},
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "feature_shape": list(statistics.mean.shape),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
