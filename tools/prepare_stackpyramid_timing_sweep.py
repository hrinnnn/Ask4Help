#!/usr/bin/env python3
"""Select a common complete-suffix action budget for StackPyramid timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


CONDITIONS = ("immediate", "pre_stage", "capability_boundary", "failure_recovery")


def _groups(handle: h5py.File) -> list[str]:
    return sorted(
        (name for name in handle if name.startswith("traj_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )


def _items(path: Path) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    with h5py.File(path, "r") as handle:
        for name in _groups(handle):
            group = handle[name]
            action_count = int(group["actions"].shape[0])
            state_count = int(group["obs/state"].shape[0])
            if action_count <= 0 or state_count != action_count + 1:
                raise ValueError(
                    f"invalid complete suffix in {path}:{name}: "
                    f"states={state_count}, actions={action_count}"
                )
            values.append((name, action_count))
    if not values:
        raise ValueError(f"no trajectories in {path}")
    return values


def _reachable(items: list[tuple[str, int]], limit: int) -> dict[int, tuple[int, int] | None]:
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for index, (_name, length) in enumerate(items):
        for current in list(parents):
            candidate = current + length
            if candidate <= limit and candidate not in parents:
                parents[candidate] = (current, index)
    return parents


def _backtrack(parents: dict[int, tuple[int, int] | None], target: int) -> list[int]:
    selected: list[int] = []
    current = target
    while current:
        parent = parents.get(current)
        if parent is None:
            raise RuntimeError(f"cannot backtrack exact budget {target}")
        previous, index = parent
        selected.append(index)
        current = previous
    selected.reverse()
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--max-budget", type=int, default=2002)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="CONDITION=H5",
        help="one source H5 for each timing condition",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_map: dict[str, Path] = {}
    for value in args.source:
        condition, separator, path = value.partition("=")
        if not separator or condition not in CONDITIONS or not path:
            raise ValueError(f"invalid --source {value!r}")
        if condition in source_map:
            raise ValueError(f"duplicate condition {condition}")
        source_map[condition] = Path(path)
    if set(source_map) != set(CONDITIONS):
        raise ValueError(f"expected exactly {CONDITIONS}, got {sorted(source_map)}")
    if args.max_budget <= 0:
        raise ValueError("--max-budget must be positive")

    item_map = {condition: _items(path) for condition, path in source_map.items()}
    parent_map = {
        condition: _reachable(items, args.max_budget)
        for condition, items in item_map.items()
    }
    common = set.intersection(*(set(parents) for parents in parent_map.values()))
    common.discard(0)
    if not common:
        raise RuntimeError(
            f"no positive common complete-suffix budget <= {args.max_budget}"
        )
    budget = max(common)

    args.output.mkdir(parents=True)
    manifest: dict[str, object] = {
        "format": "stackpyramid_timing_exact_budget_v1",
        "stage": args.stage,
        "max_budget": args.max_budget,
        "common_expert_action_budget": budget,
        "conditions": {},
    }
    for condition in CONDITIONS:
        items = item_map[condition]
        selected_indices = _backtrack(parent_map[condition], budget)
        selected_names = [items[index][0] for index in selected_indices]
        destination = args.output / condition / "accepted_suffixes.h5"
        destination.parent.mkdir(parents=True)
        with h5py.File(source_map[condition], "r") as source, h5py.File(destination, "w") as target:
            for new_index, name in enumerate(selected_names):
                source.copy(name, target, name=f"traj_{new_index:06d}")
        manifest["conditions"][condition] = {
            "source_h5": str(source_map[condition].resolve()),
            "source_episodes": len(items),
            "selected_episodes": len(selected_names),
            "source_expert_action_steps": sum(length for _name, length in items),
            "selected_expert_action_steps": sum(items[index][1] for index in selected_indices),
            "selected_groups": selected_names,
            "selected_h5": str(destination.resolve()),
        }

    (args.output / "budget_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "BUDGET_SELECTION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
