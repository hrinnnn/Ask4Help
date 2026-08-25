#!/usr/bin/env python3
"""Merge disjoint fixed-timing LeRobot pools without slicing episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def merge(dataset_roots: list[Path], collection_roots: list[Path], output: Path) -> None:
    if len(dataset_roots) != len(collection_roots):
        raise ValueError("dataset and collection roots must have equal length")
    if output.exists():
        raise FileExistsError(output)
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_out = output / "data/chunk-000"
    meta_out = output / "meta"
    data_out.mkdir(parents=True)
    meta_out.mkdir()
    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    merged_training: list[dict[str, Any]] = []
    source_manifest: list[dict[str, Any]] = []
    total_frames = 0
    template_info: dict[str, Any] | None = None
    for dataset_root, collection_root in zip(dataset_roots, collection_roots):
        episodes = read_jsonl(dataset_root / "meta/episodes.jsonl")
        stats = read_jsonl(dataset_root / "meta/episodes_stats.jsonl")
        training = read_jsonl(collection_root / "training_episodes.jsonl")
        if not (len(episodes) == len(stats) == len(training)):
            raise RuntimeError(
                f"episode mismatch for {dataset_root}: "
                f"dataset={len(episodes)} stats={len(stats)} collection={len(training)}"
            )
        if template_info is None:
            template_info = read_json(dataset_root / "meta/info.json")
            shutil.copy2(dataset_root / "meta/tasks.jsonl", meta_out / "tasks.jsonl")
        for old_index, (episode, stat, train_row) in enumerate(zip(episodes, stats, training)):
            source = dataset_root / "data/chunk-000" / f"episode_{old_index:06d}.parquet"
            if not source.is_file():
                raise FileNotFoundError(source)
            new_index = len(merged_episodes)
            table = pq.read_table(source)
            column = table.schema.get_field_index("episode_index")
            rewritten = table.set_column(
                column,
                "episode_index",
                pa.array(np.full(table.num_rows, new_index, dtype=np.int64)),
            )
            pq.write_table(
                rewritten,
                data_out / f"episode_{new_index:06d}.parquet",
                compression="zstd",
            )
            episode_out = dict(episode)
            episode_out["episode_index"] = new_index
            stat_out = dict(stat)
            stat_out["episode_index"] = new_index
            train_out = dict(train_row)
            train_out["dataset_episode_index"] = new_index
            merged_episodes.append(episode_out)
            merged_stats.append(stat_out)
            merged_training.append(train_out)
            total_frames += int(episode_out["length"])
            source_manifest.append({
                "merged_episode_index": new_index,
                "source_dataset": str(dataset_root.resolve()),
                "source_collection": str(collection_root.resolve()),
                "source_episode_index": old_index,
                "seed": int(train_row["seed"]),
            })
    if template_info is None:
        raise RuntimeError("no dataset roots supplied")
    info = dict(template_info)
    info.update(
        total_episodes=len(merged_episodes),
        total_frames=total_frames,
        total_chunks=1,
        splits={"train": f"0:{len(merged_episodes)}"},
    )
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_jsonl(meta_out / "episodes.jsonl", merged_episodes)
    write_jsonl(meta_out / "episodes_stats.jsonl", merged_stats)
    write_jsonl(output / "training_episodes.jsonl", merged_training)
    (output / "merge_manifest.json").write_text(
        json.dumps(
            {
                "format": "xvla_fixed_timing_merged_pool_v1",
                "source_roots": source_manifest,
                "total_episodes": len(merged_episodes),
                "total_frames": total_frames,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--collection-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge(args.dataset_root, args.collection_root, args.output)


if __name__ == "__main__":
    main()
