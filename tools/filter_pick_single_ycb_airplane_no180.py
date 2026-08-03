#!/usr/bin/env python3
"""Create an immutable no-180-degree subset of airplane LeRobot episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


NO180_YAW_DEGREES = (-110.0, -70.0)


def select_no180_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the direct-grasp source family without altering source records."""

    lower, upper = np.deg2rad(NO180_YAW_DEGREES)
    selected = [row for row in rows if lower <= float(row["object_yaw"]) <= upper]
    if not selected:
        raise RuntimeError("no source rows belong to the configured no-180 yaw family")
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _rewrite_episode_index(table: Any, episode_index: int) -> Any:
    import pyarrow as pa

    column_index = table.schema.get_field_index("episode_index")
    if column_index < 0:
        raise RuntimeError("source parquet has no episode_index column")
    values = pa.array(np.full(table.num_rows, episode_index, dtype=np.int64))
    return table.set_column(column_index, "episode_index", values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-videos", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_root}")
    source_rows = _read_jsonl(args.source_manifest)
    selected = select_no180_rows(source_rows)
    args.output_root.mkdir(parents=True)
    output_dataset = args.output_root / "lerobot"
    output_data = output_dataset / "data" / "chunk-000"
    output_meta = output_dataset / "meta"
    output_videos = args.output_root / "lerobot_videos"
    output_collection = args.output_root / "collection"
    output_data.mkdir(parents=True)
    output_meta.mkdir(parents=True)
    output_videos.mkdir()
    output_collection.mkdir()

    import pyarrow.parquet as pq

    metadata_rows: list[dict[str, Any]] = []
    output_episode_rows: list[dict[str, Any]] = []
    output_stats_rows: list[dict[str, Any]] = []
    source_stats = {int(row["episode_index"]): row for row in _read_jsonl(args.source_dataset / "meta" / "episodes_stats.jsonl")}
    for new_index, source_row in enumerate(selected):
        old_index = int(source_row["episode_index"])
        source_parquet = args.source_dataset / "data" / "chunk-000" / f"episode_{old_index:06d}.parquet"
        table = pq.read_table(source_parquet)
        rewritten = _rewrite_episode_index(table, new_index)
        destination_parquet = output_data / f"episode_{new_index:06d}.parquet"
        pq.write_table(rewritten, destination_parquet, compression="zstd")
        seed = int(source_row["seed"])
        source_video = args.source_videos / f"episode_{old_index:06d}_seed_{seed:06d}.mp4"
        if not source_video.is_file():
            raise FileNotFoundError(source_video)
        destination_video = output_videos / f"episode_{new_index:06d}_seed_{seed:06d}.mp4"
        shutil.copy2(source_video, destination_video)
        count = table.num_rows
        output_episode_rows.append({"episode_index": new_index, "tasks": ["pick up the toy airplane and move it to the green goal"], "length": count})
        source_stat = dict(source_stats[old_index]); source_stat["episode_index"] = new_index; output_stats_rows.append(source_stat)
        metadata_rows.append({
            "dataset_episode_index": new_index,
            "source_episode_index": old_index,
            "seed": seed,
            "object_yaw": float(source_row["object_yaw"]),
            "object_yaw_deg": float(np.rad2deg(source_row["object_yaw"])),
            "filter": "no_180_degree_reorientation",
            "actions": count,
            "source_parquet": str(source_parquet),
            "source_video": str(source_video),
        })

    info = json.loads((args.source_dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    info["total_episodes"] = len(selected)
    info["total_frames"] = sum(row["length"] for row in output_episode_rows)
    info["splits"] = {"train": f"0:{len(selected)}"}
    (output_meta / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    _write_jsonl(output_meta / "episodes.jsonl", output_episode_rows)
    _write_jsonl(output_meta / "episodes_stats.jsonl", output_stats_rows)
    shutil.copy2(args.source_dataset / "meta" / "tasks.jsonl", output_meta / "tasks.jsonl")
    _write_jsonl(output_collection / "subset_manifest.jsonl", metadata_rows)
    (output_collection / "summary.json").write_text(json.dumps({
        "source_dataset": str(args.source_dataset), "source_manifest": str(args.source_manifest),
        "source_videos": str(args.source_videos), "episodes": len(selected),
        "yaw_degrees": list(NO180_YAW_DEGREES), "norm_policy": "reuse_original_frozen_id_norm",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(selected), "total_frames": info["total_frames"], "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
