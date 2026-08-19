#!/usr/bin/env python3
"""Create a logical merged StackPyramid ID H5 without copying large arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-h5", type=Path, required=True)
    parser.add_argument("--additional-h5", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    output_h5 = args.output_root / "id" / "accepted_suffixes.h5"
    output_h5.parent.mkdir(parents=True)
    sources = [args.original_h5.resolve(), args.additional_h5.resolve()]
    counts = []
    with h5py.File(output_h5, "w") as destination:
        index = 0
        for source_path in sources:
            with h5py.File(source_path, "r") as source:
                names = sorted(name for name in source if name.startswith("traj_"))
                counts.append(len(names))
                for name in names:
                    destination[f"traj_{index:06d}"] = h5py.ExternalLink(str(source_path), name)
                    index += 1
    if index != 512:
        raise ValueError(f"expected 512 trajectories, found {index}")
    provenance = {
        "format": "stackpyramid_id_recovery_512_external_links_v1",
        "geometry_version": "v4",
        "split": "id_only",
        "instruction": "stack the red cube next to the green cube and place the blue cube on top",
        "source_h5_paths": [str(path) for path in sources],
        "source_episode_counts": counts,
        "merged_episode_count": index,
        "external_links": True,
        "manifest": str(args.manifest.resolve()),
        "ood_included": False,
        "norm_mode": "xvla_action_space",
        "recompute_norm": True,
    }
    (args.output_root / "collection_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output_root / "MERGE_COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
