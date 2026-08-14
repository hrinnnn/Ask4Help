#!/usr/bin/env python3
"""Select the exact nominal episode subset used by the Bridge-PCA asset builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite selected dataset artifacts")

    with h5py.File(args.source, "r") as source:
        groups = sorted(name for name in source if name.startswith("traj_"))
        selected = groups[: args.target]
        if len(selected) != args.target:
            raise ValueError(f"expected {args.target} trajectory groups, found {len(selected)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as output:
            for name in selected:
                source.copy(name, output, name=name)

    args.manifest.write_text(
        json.dumps(
            {
                "format": "stackpyramid_id_h5_asset_subset_v1",
                "source": str(args.source.resolve()),
                "target_episodes": args.target,
                "selection_order": "lexicographic trajectory-group order, matching build_stackpyramid_bridge_pca.py",
                "selected_groups": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"output": str(args.output), "episodes": len(selected)}))


if __name__ == "__main__":
    main()
