#!/usr/bin/env python3
"""Audit one object-variation gated collection before matched training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


METHODS = ("bridge_pca", "diffdagger", "failure_recovery", "offline_oracle")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def video_is_readable(path: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return path.stat().st_size > 0
    return result.returncode == 0 and path.stat().st_size > 0


def audit_method(method: str, collection: Path, dataset: Path, expected: int) -> dict:
    attempts = read_jsonl(collection / "attempts.jsonl")
    accepted = [row for row in attempts if row.get("accepted") is True]
    summary_path = collection / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    episodes = read_jsonl(dataset / "meta" / "episodes.jsonl")
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    accepted_videos = sorted((collection / "accepted_suffix_videos").glob("*.mp4"))
    raw_videos = sorted((collection / "raw_archive/videos").glob("*.mp4"))
    sequence = [row.get("split") for row in attempts]
    alternating = all(sequence[i] != sequence[i - 1] for i in range(1, len(sequence)))
    split_counts = {split: sum(row.get("split") == split for row in accepted) for split in ("id", "ood")}
    object_ids = sorted(
        {
            row.get("reset_metadata", {}).get("object_model_id")
            for row in accepted
            if isinstance(row.get("reset_metadata"), dict)
        }
    )
    action_steps = [int(row.get("expert_action_steps", 0)) for row in accepted]
    all_videos_readable = all(video_is_readable(path) for path in raw_videos + accepted_videos)
    passed = (
        (collection / "COLLECTION_COMPLETE").is_file()
        and int(summary.get("accepted", -1)) == expected
        and len(accepted) == expected
        and len(episodes) == expected
        and len(accepted_videos) == expected
        and len(raw_videos) == len(attempts)
        and all(bool(row.get("success")) and steps > 0 for row, steps in zip(accepted, action_steps))
        and int(info.get("total_frames", -1)) == sum(int(row["length"]) for row in episodes)
        and all_videos_readable
        and (method == "offline_oracle" or alternating)
        and (method != "offline_oracle" or split_counts == {"id": 0, "ood": expected})
        and set(object_ids) <= {"005_tomato_soup_can", "008_pudding_box"}
    )
    return {
        "method": method,
        "collection_dir": str(collection),
        "dataset_dir": str(dataset),
        "collection_complete_marker": (collection / "COLLECTION_COMPLETE").is_file(),
        "raw_attempts": len(attempts),
        "accepted_episodes": len(accepted),
        "accepted_id_ood": split_counts,
        "dataset_episodes": len(episodes),
        "dataset_total_frames": int(info.get("total_frames", -1)),
        "dataset_episode_frame_sum": sum(int(row["length"]) for row in episodes),
        "accepted_expert_action_steps": sum(action_steps),
        "expert_action_min": min(action_steps) if action_steps else 0,
        "expert_action_max": max(action_steps) if action_steps else 0,
        "raw_video_count": len(raw_videos),
        "accepted_suffix_video_count": len(accepted_videos),
        "all_videos_readable": all_videos_readable,
        "raw_split_sequence_alternating": alternating,
        "object_model_ids": object_ids,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections-root", type=Path, required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=100)
    args = parser.parse_args()

    reports = {
        method: audit_method(
            method,
            args.collections_root / method,
            args.datasets_root / f"{method}_v1",
            args.expected,
        )
        for method in METHODS
    }
    payload = {
        "format": "pick_single_ycb_object_variation_gated_collection_audit_v1",
        "expected_accepted_episodes_per_method": args.expected,
        "methods": reports,
        "passed": all(row["passed"] for row in reports.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    marker = args.output.parent / (
        "GATED_COLLECTION_AUDIT_PASSED" if payload["passed"] else "GATED_COLLECTION_AUDIT_FAILED"
    )
    marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
