#!/usr/bin/env python3
"""Select whole gated expert episodes at an exact fixed-grid budget.

The gate collection is deliberately allowed to overshoot the target.  This
utility then chooses a deterministic subset of complete episodes whose real
action count equals the frozen task budget.  It never truncates an expert
suffix or rewrites an episode's temporal order.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def exact_subset(lengths: list[int], budget: int) -> list[int] | None:
    """Return the first deterministic full-episode subset that reaches budget."""

    if budget <= 0 or any(length <= 0 for length in lengths):
        raise ValueError("budget and episode lengths must be positive")
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, length in enumerate(lengths):
        for total, chosen in list(reachable.items())[::-1]:
            candidate = total + length
            if candidate <= budget and candidate not in reachable:
                reachable[candidate] = (*chosen, index)
        if budget in reachable:
            return list(reachable[budget])
    return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def select(
    *,
    pool: Path,
    collection: Path,
    allowed_seeds: Path,
    output: Path,
    budget: int,
    task: str,
    method: str,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    episodes = read_jsonl(pool / "meta/episodes.jsonl")
    stats = read_jsonl(pool / "meta/episodes_stats.jsonl")
    training = read_jsonl(collection / "training_episodes.jsonl")
    if not episodes or len(episodes) != len(stats) or len(episodes) != len(training):
        raise RuntimeError(
            f"pool/collection episode mismatch: pool={len(episodes)} "
            f"stats={len(stats)} training={len(training)}"
        )
    allowed_payload = read_json(allowed_seeds)
    allowed = {int(seed) for seed in allowed_payload.get("seeds", [])}
    if not allowed:
        raise RuntimeError("allowed seed manifest is empty")
    candidate_indices = [
        index for index, row in enumerate(training) if int(row["seed"]) in allowed
    ]
    lengths = [int(episodes[index]["length"]) for index in candidate_indices]
    relative = exact_subset(lengths, budget)
    if relative is None:
        raise RuntimeError(
            f"no whole-episode subset reaches exact budget={budget}; "
            f"eligible_episodes={len(candidate_indices)} eligible_actions={sum(lengths)}"
        )
    selected = [candidate_indices[index] for index in relative]

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_out = output / "data/chunk-000"
    meta_out = output / "meta"
    data_out.mkdir(parents=True)
    meta_out.mkdir()
    output_episodes: list[dict[str, Any]] = []
    output_stats: list[dict[str, Any]] = []
    output_training: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(selected):
        source = pool / "data/chunk-000" / f"episode_{old_index:06d}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
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
        episode = dict(episodes[old_index])
        episode["episode_index"] = new_index
        stat = dict(stats[old_index])
        stat["episode_index"] = new_index
        train_row = dict(training[old_index])
        train_row["dataset_episode_index"] = new_index
        output_episodes.append(episode)
        output_stats.append(stat)
        output_training.append(train_row)

    info = read_json(pool / "meta/info.json")
    info.update(
        total_episodes=len(selected),
        total_frames=budget,
        total_chunks=1,
        splits={"train": f"0:{len(selected)}"},
    )
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_jsonl(meta_out / "episodes.jsonl", output_episodes)
    write_jsonl(meta_out / "episodes_stats.jsonl", output_stats)
    write_jsonl(output / "training_episodes.jsonl", output_training)
    shutil.copy2(pool / "meta/tasks.jsonl", meta_out / "tasks.jsonl")
    manifest = {
        "format": "xvla_fixedgrid_gate_exact_budget_v1",
        "task": task,
        "method": method,
        "source_pool": str(pool.resolve()),
        "source_collection": str(collection.resolve()),
        "allowed_seed_manifest": str(allowed_seeds.resolve()),
        "budget": budget,
        "selected_source_episode_indices": selected,
        "selected_lengths": [int(episodes[index]["length"]) for index in selected],
        "selected_episode_count": len(selected),
        "selection_rule": "deterministic whole-episode exact subset; no suffix slicing",
    }
    (output / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--allowed-seeds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    payload = select(
        pool=args.pool,
        collection=args.collection,
        allowed_seeds=args.allowed_seeds,
        output=args.output,
        budget=args.budget,
        task=args.task,
        method=args.method,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
