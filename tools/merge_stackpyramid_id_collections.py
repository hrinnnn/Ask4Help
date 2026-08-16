#!/usr/bin/env python3
"""Merge two audited StackPyramid ID HDF5 collections into a fresh root.

The source collections remain untouched. Groups are copied into a new
contiguous trajectory index so the training loader sees one 256-episode ID
split without mixing any OOD data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py


TASK = "stack the red cube next to the green cube and place the blue cube on top"


def _groups(path: Path) -> list[str]:
    with h5py.File(path, "r") as source:
        return sorted(name for name in source if name.startswith("traj_"))


def _summary(root: Path) -> dict[str, Any]:
    summaries = sorted(root.glob("summary.json"))
    if summaries:
        value = json.loads(summaries[0].read_text(encoding="utf-8"))
        return {
            "raw_attempts": int(value.get("raw_attempts", 0)),
            "raw_successes": int(value.get("raw_successes", 0)),
            "accepted_total": int(value.get("accepted_total", 0)),
        }
    episodes = root / "episodes.jsonl"
    rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines() if line.strip()] if episodes.exists() else []
    return {
        "raw_attempts": len(rows),
        "raw_successes": sum(bool(row.get("success")) for row in rows),
        "accepted_total": sum(bool(row.get("success")) for row in rows),
    }


def merge(
    original: Path,
    additional: Path,
    output_root: Path,
    manifest: Path,
    original_root: Path | None = None,
    additional_root: Path | None = None,
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_h5 = output_root / "id" / "accepted_suffixes.h5"
    output_h5.parent.mkdir(parents=True, exist_ok=False)

    sources = [("original", original), ("additional", additional)]
    counts: dict[str, int] = {}
    with h5py.File(output_h5, "w") as destination:
        index = 0
        for label, source_path in sources:
            names = _groups(source_path)
            counts[label] = len(names)
            with h5py.File(source_path, "r") as source:
                for name in names:
                    source.copy(name, destination, name=f"traj_{index:06d}")
                    index += 1

    if index != 256:
        raise ValueError(f"expected 256 merged trajectories, found {index}")
    provenance = {
        "format": "stackpyramid_id_recovery_256_v1",
        "task": "StackPyramid",
        "geometry_version": "v4",
        "split": "id_only",
        "instruction": TASK,
        "original_collection_root": str((original_root or original.parent.parent).resolve()),
        "additional_collection_root": str((additional_root or additional.parent.parent).resolve()),
        "manifest": str(manifest.resolve()),
        "source_group_counts": counts,
        "merged_episode_count": index,
        "source_summaries": {
            "original": _summary(original_root or original.parent.parent),
            "additional": _summary(additional_root or additional.parent.parent),
        },
        "norm": {
            "mode": "xvla_action_space",
            "external_norm_asset": None,
            "recompute_for_merged_id": True,
        },
        "ood_included": False,
        "source_h5_paths": [str(original.resolve()), str(additional.resolve())],
    }
    (output_root / "collection_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "MERGE_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-h5", type=Path, required=True)
    parser.add_argument("--additional-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--additional-root", type=Path)
    args = parser.parse_args()
    merge(
        args.original_h5,
        args.additional_h5,
        args.output_root,
        args.manifest,
        original_root=args.original_root,
        additional_root=args.additional_root,
    )


if __name__ == "__main__":
    main()
