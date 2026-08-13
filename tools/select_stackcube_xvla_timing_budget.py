#!/usr/bin/env python3
"""Select complete StackCube expert suffixes at an exact action budget."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tools.collect_stackcube_xvla_dagger import exact_budget_subset


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=2000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    episodes = read_jsonl(args.pool / "meta/episodes.jsonl")
    lengths = [int(row["length"]) for row in episodes]
    selected = exact_budget_subset(lengths, args.budget)
    if selected is None:
        raise RuntimeError(
            f"no full-episode subset reaches exact budget {args.budget}; pool={sum(lengths)}"
        )

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_out = args.output / "data/chunk-000"
    meta_out = args.output / "meta"
    data_out.mkdir(parents=True)
    meta_out.mkdir()
    source_stats = read_jsonl(args.pool / "meta/episodes_stats.jsonl")
    output_episodes, output_stats = [], []
    for new_index, old_index in enumerate(selected):
        source = args.pool / "data/chunk-000" / f"episode_{old_index:06d}.parquet"
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
        row = dict(episodes[old_index])
        row["episode_index"] = new_index
        output_episodes.append(row)
        stat = dict(source_stats[old_index])
        stat["episode_index"] = new_index
        output_stats.append(stat)

    info = json.loads((args.pool / "meta/info.json").read_text(encoding="utf-8"))
    info.update(
        total_episodes=len(selected),
        total_frames=args.budget,
        total_chunks=1,
        splits={"train": f"0:{len(selected)}"},
    )
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_jsonl(meta_out / "episodes.jsonl", output_episodes)
    write_jsonl(meta_out / "episodes_stats.jsonl", output_stats)
    shutil.copy2(args.pool / "meta/tasks.jsonl", meta_out / "tasks.jsonl")
    (args.output / "selection_manifest.json").write_text(
        json.dumps(
            {
                "format": "xvla_stackcube_timing_exact_budget_v1",
                "source_pool": str(args.pool.resolve()),
                "budget": args.budget,
                "selected_source_episode_indices": selected,
                "selected_lengths": [lengths[index] for index in selected],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
