#!/usr/bin/env python3
"""Select a common expert-action budget without altering source H5 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def groups(handle: h5py.File) -> list[str]:
    return sorted((name for name in handle if name.startswith("traj_")), key=lambda name: int(name.rsplit("_", 1)[1]))


def lengths(path: Path) -> list[tuple[str, int]]:
    with h5py.File(path, "r") as handle:
        return [(name, int(handle[name]["actions"].shape[0])) for name in groups(handle)]


def reachable(items: list[tuple[str, int]]) -> dict[int, tuple[int, int] | None]:
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for index, (_name, length) in enumerate(items):
        if length <= 0:
            continue
        for current in list(parents):
            candidate = current + length
            if candidate not in parents:
                parents[candidate] = (current, index)
    return parents


def choose_indices(parents: dict[int, tuple[int, int] | None], target: int) -> list[int]:
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
    parser.add_argument("--id-h5", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, metavar="METHOD=H5")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_map: dict[str, Path] = {}
    for value in args.source:
        method, separator, path = value.partition("=")
        if not separator or not method or not path:
            raise ValueError(f"invalid --source {value!r}; expected METHOD=H5")
        if method in source_map:
            raise ValueError(f"duplicate method {method}")
        source_map[method] = Path(path)
    if set(source_map) != {"bridge_pca", "offline_oracle", "failure_recovery", "diffdagger"}:
        raise ValueError("four methods are required")

    item_map = {method: lengths(path) for method, path in source_map.items()}
    parent_map = {method: reachable(items) for method, items in item_map.items()}
    common = set.intersection(*(set(parents) for parents in parent_map.values()))
    common.discard(0)
    if not common:
        raise RuntimeError("no positive common expert-action budget exists")
    budget = max(common)

    args.output.mkdir(parents=True)
    manifest: dict[str, object] = {
        "format": "stackpyramid_four_method_budget_v1",
        "stage": args.stage,
        "id_h5": str(args.id_h5.resolve()),
        "common_expert_action_budget": budget,
        "methods": {},
    }
    for method, source in source_map.items():
        items = item_map[method]
        selected_indices = choose_indices(parent_map[method], budget)
        selected_names = [items[index][0] for index in selected_indices]
        destination = args.output / method / "accepted_suffixes.h5"
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True)
        with h5py.File(source, "r") as source_handle, h5py.File(destination, "w") as destination_handle:
            for new_index, name in enumerate(selected_names):
                source_handle.copy(name, destination_handle, name=f"traj_{new_index:06d}")
        manifest["methods"][method] = {
            "source_h5": str(source.resolve()),
            "selected_episodes": len(selected_names),
            "source_episodes": len(items),
            "source_expert_action_steps": sum(length for _name, length in items),
            "selected_expert_action_steps": sum(items[index][1] for index in selected_indices),
            "selected_groups": selected_names,
            "selected_h5": str(destination.resolve()),
        }
    (args.output / "budget_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output / "BUDGET_SELECTION_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
