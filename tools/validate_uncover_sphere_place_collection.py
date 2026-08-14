#!/usr/bin/env python3
"""Validate the formal UncoverSpherePlace oracle collections."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


SPLITS = ("id", "handle_ood", "goal_ood")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_frames(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value or value == "N/A":
        raise ValueError(f"video frame count unavailable: {path}")
    return int(value)


def validate_split(output: Path, dataset: Path, split: str, episodes_expected: int) -> dict:
    rows = read_jsonl(output / "episodes.jsonl")
    if len(rows) != episodes_expected:
        raise ValueError(f"{split}: expected {episodes_expected} episodes, found {len(rows)}")
    if not all(row.get("success") is True for row in rows):
        raise ValueError(f"{split}: collection contains an unsuccessful episode")
    for index, row in enumerate(rows):
        if int(row["episode_index"]) != index:
            raise ValueError(f"{split}: non-contiguous output episode index at {index}")
        if int(row["frames"]) != int(row["actions"]) + 1:
            raise ValueError(f"{split}: action/frame mismatch at episode {index}")

    videos = sorted((output / "videos").glob("*.mp4"))
    if len(videos) != episodes_expected:
        raise ValueError(f"{split}: expected {episodes_expected} videos, found {len(videos)}")
    decoded_frames = [video_frames(video) for video in videos]
    if any(value <= 0 for value in decoded_frames):
        raise ValueError(f"{split}: an output video has no frames")

    info = read_json(dataset / "meta" / "info.json")
    dataset_rows = read_jsonl(dataset / "meta" / "episodes.jsonl")
    if len(dataset_rows) != episodes_expected:
        raise ValueError(f"{split}: dataset metadata has {len(dataset_rows)} episodes")
    if info["features"]["actions"]["shape"] != [8]:
        raise ValueError(f"{split}: action feature is not 8D")
    if info["features"]["state"]["shape"] != [9]:
        raise ValueError(f"{split}: state feature is not 9D")
    for key in ("image", "wrist_image"):
        if key not in info["features"]:
            raise ValueError(f"{split}: missing camera feature {key}")

    lengths = [int(row["length"]) for row in dataset_rows]
    valid_counts = {str(value): 0 for value in range(1, 11)}
    for length in lengths:
        for anchor in range(length):
            valid_counts[str(min(10, length - anchor))] += 1
    return {
        "split": split,
        "episodes": len(rows),
        "total_actions": sum(lengths),
        "total_anchors": sum(lengths),
        "tail_anchors": sum(min(9, length) for length in lengths),
        "valid_target_count_distribution": valid_counts,
        "final_observation_valid_targets": 1,
        "videos": len(videos),
        "video_frames_min": min(decoded_frames),
        "video_frames_max": max(decoded_frames),
        "dataset_root": str(dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for split in SPLITS:
        reports.append(
            validate_split(
                args.output_root / split,
                args.dataset_root / split,
                split,
                args.episodes,
            )
        )
    payload = {"format": "uncover_sphere_place_collection_validation_v1", "splits": reports}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
