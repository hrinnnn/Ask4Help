#!/usr/bin/env python3
"""Build four exact-action-budget OpenDrawer expert subsets for one OOD split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.collect_stackcube_xvla_dagger import exact_budget_subset


METHODS = ("pca_only", "diffdagger", "failure_recovery", "offline_oracle")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def dataset_lengths(dataset: Path) -> list[int]:
    return [int(row["length"]) for row in read_jsonl(dataset / "meta/episodes.jsonl")]


def copy_subset(source: Path, output: Path, selected: list[int], budget: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    data_out = output / "data/chunk-000"
    meta_out = output / "meta"
    data_out.mkdir(parents=True)
    meta_out.mkdir()

    episodes = read_jsonl(source / "meta/episodes.jsonl")
    stats = read_jsonl(source / "meta/episodes_stats.jsonl")
    output_episodes: list[dict] = []
    output_stats: list[dict] = []
    for new_index, old_index in enumerate(selected):
        src = source / "data/chunk-000" / f"episode_{old_index:06d}.parquet"
        table = pq.read_table(src)
        column = table.schema.get_field_index("episode_index")
        rewritten = table.set_column(
            column,
            "episode_index",
            pa.array(np.full(table.num_rows, new_index, dtype=np.int64)),
        )
        pq.write_table(rewritten, data_out / f"episode_{new_index:06d}.parquet", compression="zstd")
        episode = dict(episodes[old_index])
        episode["episode_index"] = new_index
        output_episodes.append(episode)
        stat = dict(stats[old_index])
        stat["episode_index"] = new_index
        output_stats.append(stat)

    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    info.update(
        total_episodes=len(selected),
        total_frames=budget,
        total_chunks=1,
        splits={"train": f"0:{len(selected)}"},
    )
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_jsonl(meta_out / "episodes.jsonl", output_episodes)
    write_jsonl(meta_out / "episodes_stats.jsonl", output_stats)
    tasks = source / "meta/tasks.jsonl"
    if tasks.is_file():
        (meta_out / "tasks.jsonl").write_text(tasks.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    sources = {method: getattr(args, method) for method in METHODS}
    lengths = {method: dataset_lengths(path) for method, path in sources.items()}
    totals = {method: sum(values) for method, values in lengths.items()}
    budget = min(totals.values())
    if budget <= 0:
        raise RuntimeError(f"no positive common expert-action budget: {totals}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    selected: dict[str, list[int]] = {}
    actual: dict[str, int] = {}
    for method, source in sources.items():
        subset = exact_budget_subset(lengths[method], budget)
        if subset is None:
            raise RuntimeError(f"cannot select exact budget={budget} for {method}: {totals[method]}")
        selected[method] = subset
        actual[method] = sum(lengths[method][index] for index in subset)
        copy_subset(source, args.output_root / method, subset, budget)

    manifest = {
        "format": "open_drawer_matched_expert_action_budget_v1",
        "common_expert_action_budget": budget,
        "source_total_expert_actions": totals,
        "selected_expert_actions": actual,
        "selected_source_episode_indices": selected,
        "source_datasets": {method: str(path.resolve()) for method, path in sources.items()},
    }
    (args.output_root / "budget_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
