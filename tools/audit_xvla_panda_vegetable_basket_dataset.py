#!/usr/bin/env python3
"""Audit Panda basket trajectories and build the frozen ID-only norm."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


ACTION_HORIZON = 30
ACTIVE_ACTION_DIM = 10


def _audit_episode(path: Path) -> tuple[int, np.ndarray, dict]:
    with h5py.File(path, "r") as h5:
        actions = np.asarray(h5["abs_action_6d"], dtype=np.float32)
        proprio = np.asarray(h5["proprio"], dtype=np.float32)
        image_count = len(h5["images"])
        attrs = {str(key): h5.attrs[key].item() if hasattr(h5.attrs[key], "item") else h5.attrs[key] for key in h5.attrs}
    if actions.ndim != 2 or actions.shape[1] != ACTIVE_ACTION_DIM:
        raise ValueError(f"{path}: action shape {actions.shape}")
    if proprio.shape != actions.shape:
        raise ValueError(f"{path}: proprio/action shape mismatch {proprio.shape} vs {actions.shape}")
    if image_count < len(actions):
        raise ValueError(f"{path}: fewer images than actions")
    if not np.isfinite(actions).all() or not np.isfinite(proprio).all():
        raise ValueError(f"{path}: non-finite action or proprio")
    return len(actions), actions, {"attrs": attrs, "image_count": image_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    paths = sorted((args.dataset / "data").glob("episode_*.h5"))
    if len(paths) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} episodes, found {len(paths)}")
    lengths = []
    action_values = []
    image_counts = []
    for path in paths:
        length, actions, details = _audit_episode(path)
        lengths.append(length)
        action_values.append(actions)
        image_counts.append(details["image_count"])
    all_actions = np.concatenate(action_values, axis=0)
    valid_lengths = Counter()
    for length in lengths:
        for index in range(length):
            valid_lengths[min(ACTION_HORIZON, length - index)] += 1
    report = {
        "format": "xvla_panda_vegetable_basket_dataset_audit_v1",
        "dataset": str(args.dataset),
        "episodes": len(paths),
        "total_real_observations_anchors": int(sum(lengths)),
        "tail_anchor_count": int(sum(min(ACTION_HORIZON - 1, length) for length in lengths)),
        "valid_timesteps_histogram": {str(key): int(value) for key, value in sorted(valid_lengths.items())},
        "final_anchor_valid_timesteps": sorted({min(ACTION_HORIZON, length - (length - 1)) for length in lengths}),
        "action_dim": ACTIVE_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "all_finite": bool(np.isfinite(all_actions).all()),
        "min_episode_observations": int(min(lengths)),
        "max_episode_observations": int(max(lengths)),
        "image_count_min": int(min(image_counts)),
        "image_count_max": int(max(image_counts)),
        "norm_source": "ID expert actions only",
    }
    norm = {
        "format": "xvla_panda_vegetable_basket_id_norm_v1",
        "source_dataset": str(args.dataset),
        "episodes": len(paths),
        "mean": all_actions.mean(axis=0).tolist(),
        "std": np.maximum(all_actions.std(axis=0), 1e-6).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "frozen": True,
    }
    (args.output / "dataset_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "norm_stats.json").write_text(json.dumps(norm, indent=2) + "\n", encoding="utf-8")
    if report["final_anchor_valid_timesteps"] != [1] or not report["all_finite"]:
        raise SystemExit("DATASET_AUDIT_FAILED")
    (args.output / "DATASET_AUDIT_PASSED").write_text("passed\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
