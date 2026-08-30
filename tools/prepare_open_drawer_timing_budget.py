#!/usr/bin/env python3
"""Select one exact whole-episode expert-action budget across timing anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_subset(lengths: list[int], budget: int) -> list[int] | None:
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, length in enumerate(lengths):
        if length <= 0:
            continue
        for total, chosen in list(reachable.items())[::-1]:
            candidate = total + int(length)
            if candidate <= budget and candidate not in reachable:
                reachable[candidate] = (*chosen, index)
        if budget in reachable:
            return list(reachable[budget])
    return None


def reachable_sums(lengths: list[int]) -> set[int]:
    """Return whole-episode sums reachable without slicing or padding."""

    reachable = {0}
    for length in lengths:
        reachable |= {total + int(length) for total in tuple(reachable)}
    return reachable


def copy_dataset(source: Path, output: Path, selected: list[int], budget: int) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_out = output / "data/chunk-000"
    meta_out = output / "meta"
    data_out.mkdir(parents=True)
    meta_out.mkdir()
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    stats = read_jsonl(source / "meta/episodes_stats.jsonl")
    out_episodes: list[dict] = []
    out_stats: list[dict] = []
    frame_count = 0
    for new_index, old_index in enumerate(selected):
        src = source / "data/chunk-000" / f"episode_{old_index:06d}.parquet"
        table = pq.read_table(src)
        column = table.schema.get_field_index("episode_index")
        table = table.set_column(column, "episode_index", pa.array(np.full(table.num_rows, new_index, dtype=np.int64)))
        pq.write_table(table, data_out / f"episode_{new_index:06d}.parquet", compression="zstd")
        episode = dict(episodes[old_index]); episode["episode_index"] = new_index
        stat = dict(stats[old_index]); stat["episode_index"] = new_index
        out_episodes.append(episode); out_stats.append(stat)
        frame_count += int(episode.get("length", table.num_rows))
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    info.update(
        total_episodes=len(selected),
        total_frames=frame_count,
        total_chunks=1,
        splits={"train": f"0:{len(selected)}"},
    )
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (meta_out / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in out_episodes), encoding="utf-8")
    (meta_out / "episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in out_stats), encoding="utf-8")
    tasks = source / "meta/tasks.jsonl"
    if tasks.is_file():
        (meta_out / "tasks.jsonl").write_text(tasks.read_text(encoding="utf-8"), encoding="utf-8")


def parse_condition(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=DATASET_PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=DATASET_PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True)
    parser.add_argument(
        "--budget",
        type=str,
        help="integer exact budget, or auto_max_common for the largest common reachable sum",
    )
    args = parser.parse_args()
    conditions = dict(args.condition)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_root}")
    lengths: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    for name, source in conditions.items():
        episodes = read_jsonl(source / "meta/episodes.jsonl")
        values = [int(row["length"]) for row in episodes]
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"invalid episode lengths for {name}: {source}")
        lengths[name] = values
        totals[name] = sum(values)
    if args.budget == "auto_max_common":
        common = set.intersection(*(reachable_sums(lengths[name]) for name in conditions))
        common.discard(0)
        if not common:
            raise RuntimeError("no positive common whole-episode budget is reachable")
        budget = max(common)
        budget_selection_rule = "maximum_common_reachable_whole_episode_sum"
    else:
        budget = int(args.budget) if args.budget is not None else min(totals.values())
        budget_selection_rule = "explicit_or_minimum_source_total"
    if budget <= 0 or budget > min(totals.values()):
        raise ValueError(f"budget {budget} exceeds a source total: {totals}")
    selected: dict[str, list[int]] = {}
    actual: dict[str, int] = {}
    for name, source in conditions.items():
        chosen = exact_subset(lengths[name], budget)
        if chosen is None:
            raise RuntimeError(f"no whole-episode subset reaches exact budget={budget} for {name}; total={totals[name]}")
        selected[name] = chosen
        actual[name] = sum(lengths[name][index] for index in chosen)
    args.output_root.mkdir(parents=True)
    for name, source in conditions.items():
        copy_dataset(source, args.output_root / name, selected[name], budget)
    manifest = {
        "format": "open_drawer_timing_exact_budget_v1",
        "common_expert_action_budget": budget,
        "source_total_expert_actions": totals,
        "selected_expert_actions": actual,
        "selected_source_episode_indices": selected,
        "source_datasets": {name: str(path.resolve()) for name, path in conditions.items()},
        "budget_selection_rule": budget_selection_rule,
    }
    (args.output_root / "budget_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "BUDGET_AUDIT_PASS").write_text("all timing anchors selected exact whole-episode budget\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
