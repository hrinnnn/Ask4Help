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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

