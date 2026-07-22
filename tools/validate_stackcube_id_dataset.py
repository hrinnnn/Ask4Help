#!/usr/bin/env python3
"""Strictly validate a controlled StackCube LeRobot dataset before SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _video_frames(path: Path) -> int:
    import imageio.v3 as iio

    count = 0
    first = None
    peak = 0.0
    for frame in iio.imiter(path):
        value = np.asarray(frame, dtype=np.int16)
        if first is None:
            first = value
        else:
            peak = max(peak, float(np.abs(value - first).mean()))
        count += 1
    if count < 2 or peak < 1.0:
        raise RuntimeError(f"invalid visual motion in {path}: frames={count}, peak={peak}")
    return count


def _validate_parquet(path: Path, *, episode_index: int, expected_actions: int) -> dict[str, float]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["state", "actions", "frame_index", "episode_index"])
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if len(actions) != expected_actions:
        raise RuntimeError(
            f"{path} has {len(actions)} action rows, manifest expects {expected_actions}"
        )
    if states.shape != (expected_actions, 9) or actions.shape != (expected_actions, 8):
        raise RuntimeError(f"unexpected state/action shape in {path}: {states.shape}, {actions.shape}")
    if not np.array_equal(frame_indices, np.arange(expected_actions)):
        raise RuntimeError(f"non-contiguous frame_index in {path}")
    if not np.all(episode_indices == episode_index):
        raise RuntimeError(f"wrong episode_index in {path}")
    state_motion = float(np.abs(states - states[0]).max())
    action_motion = float(np.abs(actions - actions[0]).max())
    if state_motion <= 1e-5 or action_motion <= 1e-5:
        raise RuntimeError(
            f"constant state/action trajectory in {path}: state={state_motion}, action={action_motion}"
        )
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise RuntimeError(f"non-finite state/action values in {path}")
    return {"state_peak_delta": state_motion, "action_peak_delta": action_motion}


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    parquet = sorted((args.dataset / "data").rglob("*.parquet"))
    video_dir = args.dataset.with_name(f"{args.dataset.name}_videos")
    videos = sorted(video_dir.glob("*.mp4"))
    if len(rows) != args.episodes or len(parquet) != args.episodes or len(videos) != args.episodes:
        raise RuntimeError(
            f"expected {args.episodes}: manifest={len(rows)} parquet={len(parquet)} videos={len(videos)}"
        )
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != args.episodes:
        raise RuntimeError("manifest seeds are not unique")
    for row in rows:
        if not row.get("success"):
            raise RuntimeError(f"episode {row.get('episode_index')} is not successful")
        distance = float(row["relative_distance"])
        angle_offset = abs(float(row["relative_angle_offset"]))
        if not 0.08 <= distance <= 0.10 or angle_offset > np.deg2rad(10) + 1e-8:
            raise RuntimeError(f"episode escaped ID geometry: {row}")
        if int(row.get("num_actions", 0)) <= 0:
            raise RuntimeError(f"episode has no actions: {row}")
        motion = row.get("visual_motion", {})
        if max(float(motion.get("image", 0)), float(motion.get("wrist_image", 0))) < 1.0:
            raise RuntimeError(f"episode has static RGB according to manifest: {row}")
    frame_counts = [_video_frames(path) for path in videos]
    parquet_stats = [
        _validate_parquet(
            path,
            episode_index=index,
            expected_actions=int(rows[index]["num_actions"]),
        )
        for index, path in enumerate(parquet)
    ]
    report = {
        "valid": True,
        "episodes": args.episodes,
        "unique_seeds": len(set(seeds)),
        "parquet_files": len(parquet),
        "videos": len(videos),
        "video_frames_min": min(frame_counts),
        "video_frames_max": max(frame_counts),
        "distance_min": min(float(row["relative_distance"]) for row in rows),
        "distance_max": max(float(row["relative_distance"]) for row in rows),
        "angle_offset_deg_min": min(np.rad2deg(float(row["relative_angle_offset"])) for row in rows),
        "angle_offset_deg_max": max(np.rad2deg(float(row["relative_angle_offset"])) for row in rows),
        "state_peak_delta_min": min(item["state_peak_delta"] for item in parquet_stats),
        "action_peak_delta_min": min(item["action_peak_delta"] for item in parquet_stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
