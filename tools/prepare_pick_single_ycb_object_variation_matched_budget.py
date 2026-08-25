#!/usr/bin/env python3
"""Build exact common expert-action-budget LeRobot subsets.

The four collection branches are allowed to produce different suffix lengths.
This stage selects the largest exact common budget not exceeding the frozen
protocol cap, then rewrites only the selected episode copies. Source datasets
and their raw evidence are never modified.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


METHODS = ("bridge_pca", "diffdagger", "failure_recovery", "offline_oracle")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def exact_reachable(lengths: list[int], cap: int) -> dict[int, tuple[int, ...]]:
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, length in enumerate(lengths):
        if length <= 0:
            continue
        for total, chosen in list(reachable.items()):
            candidate = total + length
            if candidate <= cap and candidate not in reachable:
                reachable[candidate] = (*chosen, index)
    return reachable


def rewrite_episode_stats(stats: dict, *, new_index: int, frame_start: int) -> dict:
    out = copy.deepcopy(stats)
    if "episode_index" in out:
        source = out["episode_index"]
        if isinstance(source, dict):
            count = source.get("count", [1])
            out["episode_index"] = {
                "min": [new_index],
                "max": [new_index],
                "mean": [float(new_index)],
                "std": [0.0],
                "count": count,
            }
        else:
            # Current LeRobot exports store this field as a scalar episode id.
            out["episode_index"] = new_index
    if "index" in out:
        source = out["index"]
        count = source.get("count", [1])
        minimum = source.get("min", [0])
        maximum = source.get("max", [0])
        mean = source.get("mean", [0.0])
        out["index"] = {
            "min": [int(value) - int(minimum[0]) + frame_start for value in minimum],
            "max": [int(value) - int(minimum[0]) + frame_start for value in maximum],
            "mean": [float(value) - float(minimum[0]) + frame_start for value in mean],
            "std": source.get("std", [0.0]),
            "count": count,
        }
    return out


def episode_path(root: Path, info: dict, episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    chunk = episode_index // chunk_size
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def copy_subset(source: Path, destination: Path, selected: list[int], budget: int) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    (destination / "data" / "chunk-000").mkdir(parents=True)
    (destination / "meta").mkdir(parents=True)

    info = json.loads((source / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = read_jsonl(source / "meta" / "episodes.jsonl")
    stats_path = source / "meta" / "episodes_stats.jsonl"
    stats = read_jsonl(stats_path) if stats_path.is_file() else []
    output_episodes: list[dict] = []
    output_stats: list[dict] = []
    frame_start = 0
    for new_index, old_index in enumerate(selected):
        table = pq.read_table(episode_path(source, info, old_index))
        column = table.schema.get_field_index("episode_index")
        if column >= 0:
            table = table.set_column(
                column,
                "episode_index",
                pa.array(np.full(table.num_rows, new_index, dtype=np.int64)),
            )
        pq.write_table(
            table,
            destination / "data" / "chunk-000" / f"episode_{new_index:06d}.parquet",
            compression="zstd",
        )
        episode = dict(episodes[old_index])
        episode["episode_index"] = new_index
        output_episodes.append(episode)
        if stats:
            output_stats.append(
                rewrite_episode_stats(
                    stats[old_index], new_index=new_index, frame_start=frame_start
                )
            )
        frame_start += int(episode["length"])

    info.update(
        total_episodes=len(selected),
        total_frames=budget,
        total_videos=0,
        total_chunks=1,
        splits={"train": f"0:{len(selected)}"},
    )
    (destination / "meta" / "info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    write_jsonl(destination / "meta" / "episodes.jsonl", output_episodes)
    if output_stats:
        write_jsonl(destination / "meta" / "episodes_stats.jsonl", output_stats)
    tasks = source / "meta" / "tasks.jsonl"
    if tasks.is_file():
        (destination / "meta" / "tasks.jsonl").write_text(
            tasks.read_text(encoding="utf-8"), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expert-action-cap", type=int, required=True)
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    sources = {method: getattr(args, method) for method in METHODS}
    lengths = {
        method: [int(row["length"]) for row in read_jsonl(path / "meta" / "episodes.jsonl")]
        for method, path in sources.items()
    }
    totals = {method: sum(values) for method, values in lengths.items()}
    cap = int(args.expert_action_cap)
    if cap <= 0:
        raise ValueError("expert-action cap must be positive")
    reachable = {method: exact_reachable(values, cap) for method, values in lengths.items()}
    common = set.intersection(*(set(values) for values in reachable.values()))
    common.discard(0)
    if not common:
        raise RuntimeError(f"no exact common budget <= {cap}; source totals={totals}")
    budget = max(common)

    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    selected: dict[str, list[int]] = {}
    selected_totals: dict[str, int] = {}
    for method, source in sources.items():
        indices = list(reachable[method][budget])
        selected[method] = indices
        selected_totals[method] = sum(lengths[method][index] for index in indices)
        copy_subset(source, args.output_root / method, indices, budget)

    manifest = {
        "format": "pick_single_ycb_object_variation_matched_expert_budget_v1",
        "expert_action_cap": cap,
        "common_expert_action_budget": budget,
        "source_total_expert_actions": totals,
        "selected_expert_actions": selected_totals,
        "selected_source_episode_indices": selected,
        "source_datasets": {method: str(path.resolve()) for method, path in sources.items()},
        "selection_rule": "largest exact common budget not exceeding frozen cap",
    }
    (args.output_root / "budget_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "BUDGET_SELECTION_COMPLETE").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
