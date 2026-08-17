#!/usr/bin/env python3
"""Audit the OpenDrawer ID expansion dataset and temporal-mask boundary."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

TASK = "open the drawer, retrieve the blue object, and place it in the green tray"
HORIZON = 10
MAX_EPISODE_STEPS = 400


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_file(dataset: Path, episode_index: int, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    return dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _dataset_report(dataset: Path, label: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    errors: list[str] = []
    info_path = dataset / "meta" / "info.json"
    episodes_path = dataset / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        return {"label": label, "path": str(dataset), "errors": ["missing_meta"], "pass": False}

    info = json.loads(info_path.read_text())
    episodes = _jsonl(episodes_path)
    chunks_size = int(info["chunks_size"])
    lengths: list[int] = []
    valid_distribution: Counter[str] = Counter()
    action_dim: int | None = None
    state_dim: int | None = None
    total_rows = 0

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        lengths.append(length)
        for anchor in range(length):
            valid_distribution[str(min(HORIZON, length - anchor))] += 1
        parquet_path = _episode_file(dataset, episode_index, chunks_size)
        if not parquet_path.is_file():
            errors.append(f"missing_parquet:{episode_index}")
            continue
        table = pq.read_table(parquet_path)
        total_rows += table.num_rows
        for key, expected_dim in (("actions", 8), ("state", 9)):
            if key not in table.column_names:
                errors.append(f"missing_column:{episode_index}:{key}")
                continue
            values = table[key].to_pylist()
            if len(values) != length:
                errors.append(f"row_count:{episode_index}:{key}:{len(values)}!={length}")
            if values:
                observed_dim = len(values[0]) if isinstance(values[0], (list, tuple)) else None
                if observed_dim != expected_dim:
                    errors.append(f"dim:{episode_index}:{key}:{observed_dim}!={expected_dim}")
                if key == "actions":
                    action_dim = observed_dim
                else:
                    state_dim = observed_dim
        if "frame_index" in table.column_names:
            frame_indices = table["frame_index"].to_pylist()
            if frame_indices and int(frame_indices[0]) != 0:
                errors.append(f"frame_start:{episode_index}")

    if not lengths:
        errors.append("no_episodes")
    if lengths and total_rows != sum(lengths):
        errors.append(f"total_rows:{total_rows}!={sum(lengths)}")

    return {
        "label": label,
        "path": str(dataset),
        "episodes": len(episodes),
        "total_actions_and_anchors": sum(lengths),
        "total_parquet_rows": total_rows,
        "episode_length_min": min(lengths) if lengths else None,
        "episode_length_max": max(lengths) if lengths else None,
        "episode_length_median": statistics.median(lengths) if lengths else None,
        "episode_length_p95": statistics.quantiles(lengths, n=20, method="inclusive")[18] if len(lengths) > 1 else (lengths[0] if lengths else None),
        "timeout_count_at_max_episode_steps": sum(length >= MAX_EPISODE_STEPS for length in lengths),
        "action_dim": action_dim,
        "state_dim": state_dim,
        "action_horizon": HORIZON,
        "tail_anchor_count": 9 * len(episodes),
        "valid_target_timestep_distribution": dict(sorted(valid_distribution.items(), key=lambda item: int(item[0]))),
        "final_anchor_valid_timesteps": 1 if lengths and all(length >= 1 for length in lengths) else 0,
        "errors": errors,
        "pass": not errors and len(episodes) > 0,
        "info": info,
    }


def _collection_report(path: Path, label: str, expected: int, forbidden_seeds: set[int]) -> dict[str, Any]:
    errors: list[str] = []
    rows = _jsonl(path)
    seeds = [int(row["seed"]) for row in rows if "seed" in row]
    if len(rows) != expected:
        errors.append(f"episodes:{len(rows)}!={expected}")
    if len(set(seeds)) != len(seeds):
        errors.append("duplicate_seed")
    if forbidden_seeds.intersection(seeds):
        errors.append("seed_overlap_with_old_collection")
    for row in rows:
        if row.get("split") != "id":
            errors.append(f"non_id_split:{row.get('split')}")
        if row.get("success") is not True:
            errors.append(f"unsuccessful_row:{row.get('episode_index')}")
        if row.get("instruction") != TASK:
            errors.append(f"instruction_mismatch:{row.get('episode_index')}")
        if int(row.get("num_actions", 0)) < 1:
            errors.append(f"empty_actions:{row.get('episode_index')}")
    return {
        "label": label,
        "path": str(path),
        "episodes": len(rows),
        "seeds_min": min(seeds) if seeds else None,
        "seeds_max": max(seeds) if seeds else None,
        "total_actions": sum(int(row.get("num_actions", 0)) for row in rows),
        "errors": sorted(set(errors)),
        "pass": not errors,
    }


def _video_report(directory: Path, expected: int) -> dict[str, Any]:
    videos = sorted(directory.glob("*.mp4")) if directory.is_dir() else []
    errors: list[str] = []
    if len(videos) != expected:
        errors.append(f"video_count:{len(videos)}!={expected}")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        errors.append("ffprobe_missing")
    else:
        for video in videos:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                errors.append(f"video_decode:{video.name}")
    return {"path": str(directory), "videos": len(videos), "errors": errors, "pass": not errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dataset", type=Path, required=True)
    parser.add_argument("--new-dataset", type=Path, required=True)
    parser.add_argument("--old-collection", type=Path, required=True)
    parser.add_argument("--new-collection", type=Path, required=True)
    parser.add_argument("--old-videos", type=Path, required=True)
    parser.add_argument("--new-videos", type=Path, required=True)
    parser.add_argument("--merged-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_rows = _jsonl(args.old_collection)
    old_seeds = {int(row["seed"]) for row in old_rows if "seed" in row}
    report: dict[str, Any] = {
        "format": "open_drawer_id_expansion_audit_v1",
        "task": TASK,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "execute_horizon": 5,
        "action_horizon": HORIZON,
        "old_dataset": _dataset_report(args.old_dataset, "old_id_128"),
        "new_dataset": _dataset_report(args.new_dataset, "new_id_extra_128"),
        "old_collection": _collection_report(args.old_collection, "old_id_128", 128, set()),
        "new_collection": _collection_report(args.new_collection, "new_id_extra_128", 128, old_seeds),
        "old_videos": _video_report(args.old_videos, 128),
        "new_videos": _video_report(args.new_videos, 128),
    }
    if args.merged_dataset:
        report["merged_dataset"] = _dataset_report(args.merged_dataset, "merged_id_256")
    report["pass"] = all(
        report[key]["pass"]
        for key in ("old_dataset", "new_dataset", "old_collection", "new_collection", "old_videos", "new_videos")
    ) and ("merged_dataset" not in report or report["merged_dataset"]["pass"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
