#!/usr/bin/env python3
"""Validate a collected StackCube gated-DAgger expert-suffix dataset.

Unlike the original-ID validator, this collector stores visual videos in the
raw archive and uses the LeRobot parquet files only for trainable frames and
actions.  This script checks their one-to-one correspondence before SFT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--episodes-manifest", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _video_frame_count(path: Path) -> int:
    import imageio.v3 as iio

    frames = 0
    first: np.ndarray | None = None
    peak_delta = 0.0
    for frame in iio.imiter(path):
        value = np.asarray(frame, dtype=np.int16)
        if first is None:
            first = value
        else:
            peak_delta = max(peak_delta, float(np.abs(value - first).mean()))
        frames += 1
    if frames < 2 or peak_delta < 1.0:
        raise RuntimeError(f"invalid or static raw rollout video: {path}")
    return frames


def _validate_episode(
    *, parquet_path: Path, row: dict[str, Any], raw_row: dict[str, Any], raw_archive: Path
) -> dict[str, float | int]:
    import pyarrow.parquet as pq

    episode_index = int(row["dataset_episode_index"])
    raw_episode_index = int(row["raw_episode_index"])
    seed = int(row["seed"])
    start_step = int(row["start_step"])
    action_steps = int(row["action_steps"])
    if action_steps < 10:
        raise RuntimeError(f"episode {episode_index} has no complete 10-step anchor")
    if not raw_row.get("success") or raw_row.get("expert_start_step") != start_step:
        raise RuntimeError(f"raw manifest disagrees with training suffix for episode {episode_index}")

    table = pq.read_table(parquet_path, columns=["state", "actions", "frame_index", "episode_index"])
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if actions.shape != (action_steps, 8) or states.shape[0] != action_steps:
        raise RuntimeError(f"unexpected parquet trajectory shape in {parquet_path}: {states.shape}, {actions.shape}")
    if not np.array_equal(frame_indices, np.arange(action_steps)):
        raise RuntimeError(f"non-contiguous frame indices in {parquet_path}")
    if not np.all(episode_indices == episode_index):
        raise RuntimeError(f"wrong dataset episode index in {parquet_path}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise RuntimeError(f"non-finite state/action value in {parquet_path}")
    if float(np.abs(actions - actions[0]).max()) <= 1e-5:
        raise RuntimeError(f"constant expert actions in {parquet_path}")

    action_stem = raw_archive / "actions" / f"episode_{raw_episode_index:06d}_seed_{seed:06d}"
    raw_actions = np.load(str(action_stem) + ".npy")
    expected_actions = np.asarray(raw_actions[start_step : start_step + action_steps], dtype=np.float32)
    if expected_actions.shape != actions.shape or not np.allclose(actions, expected_actions, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"parquet actions do not exactly match raw expert suffix for episode {episode_index}")
    video = raw_archive / "videos" / f"episode_{raw_episode_index:06d}_seed_{seed:06d}.mp4"
    return {
        "action_steps": action_steps,
        "video_frames": _video_frame_count(video),
        "action_peak_delta": float(np.abs(actions - actions[0]).max()),
    }


def main() -> None:
    args = parse_args()
    train_rows = _jsonl(args.training_manifest)
    raw_rows = _jsonl(args.episodes_manifest)
    parquet = sorted((args.dataset / "data").rglob("*.parquet"))
    if len(train_rows) != args.episodes or len(parquet) != args.episodes:
        raise RuntimeError(f"expected {args.episodes} training episodes: manifest={len(train_rows)} parquet={len(parquet)}")
    raw_by_index = {int(row["episode_index"]): row for row in raw_rows}
    if len(raw_by_index) != len(raw_rows):
        raise RuntimeError("raw manifest has duplicate episode indices")

    results = []
    for index, row in enumerate(train_rows):
        if int(row["dataset_episode_index"]) != index:
            raise RuntimeError("training dataset episode indices are not contiguous")
        raw_index = int(row["raw_episode_index"])
        results.append(
            _validate_episode(
                parquet_path=parquet[index], row=row, raw_row=raw_by_index[raw_index], raw_archive=args.raw_archive
            )
        )
    report = {
        "valid": True,
        "episodes": args.episodes,
        "parquet_episodes": len(parquet),
        "raw_videos_decoded": len(results),
        "suffix_actions_total": sum(int(item["action_steps"]) for item in results),
        "suffix_actions_min": min(int(item["action_steps"]) for item in results),
        "video_frames_min": min(int(item["video_frames"]) for item in results),
        "action_peak_delta_min": min(float(item["action_peak_delta"]) for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
