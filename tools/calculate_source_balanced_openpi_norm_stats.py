#!/usr/bin/env python3
"""Compute OpenPI norm stats for an equal-source, valid-horizon SFT mixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
import openpi.shared.normalize as normalize  # noqa: E402


def valid_anchor_indices(episode_bounds: list[tuple[int, int]], horizon: int) -> np.ndarray:
    """Return global frame starts whose complete action horizon stays in-episode."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    parts = [np.arange(start, max(start, end - horizon + 1)) for start, end in episode_bounds]
    output = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if not len(output):
        raise ValueError("no complete action-horizon anchors")
    return output


def _source_arrays(dataset_path: Path, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = LeRobotDataset(str(dataset_path))
    bounds = [
        (int(dataset.episode_data_index["from"][episode]), int(dataset.episode_data_index["to"][episode]))
        for episode in range(dataset.meta.total_episodes)
    ]
    anchors = valid_anchor_indices(bounds, horizon)
    table = dataset.hf_dataset.with_format("numpy")
    states = np.asarray(table["state"])[anchors]
    actions = np.asarray(table["actions"])
    windows = np.stack([actions[index : index + horizon] for index in anchors])
    return states.astype(np.float32), windows.astype(np.float32)


def equal_source_statistics(
    source_arrays: list[tuple[np.ndarray, np.ndarray]], *, seed: int
) -> tuple[dict[str, normalize.NormStats], dict[str, int]]:
    """Estimate stats under equal source probability, independent of source size."""
    if len(source_arrays) != 2:
        raise ValueError("this source-balanced SFT utility expects exactly two sources")
    count = max(len(states) for states, _actions in source_arrays)
    rng = np.random.default_rng(seed)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for states, actions in source_arrays:
        draw = rng.choice(len(states), size=count, replace=len(states) < count)
        stats["state"].update(states[draw])
        stats["actions"].update(actions[draw])
    return {key: item.get_statistics() for key, item in stats.items()}, {
        "anchors_per_source_after_resampling": count,
        "source_count": len(source_arrays),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-replay", type=Path, required=True)
    parser.add_argument("--new-expert-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError("output-dir must be a fresh result directory")
    source_arrays = [
        _source_arrays(args.id_replay, args.horizon),
        _source_arrays(args.new_expert_dataset, args.horizon),
    ]
    stats, sampling = equal_source_statistics(source_arrays, seed=args.seed)
    normalize.save(args.output_dir, stats)
    report = {
        "id_replay": str(args.id_replay),
        "new_expert_dataset": str(args.new_expert_dataset),
        "source_sampling": "equal_probability_per_source",
        "horizon": args.horizon,
        "seed": args.seed,
        "id_valid_anchors": len(source_arrays[0][0]),
        "new_valid_anchors": len(source_arrays[1][0]),
        **sampling,
    }
    (args.output_dir / "source_balanced_norm_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
