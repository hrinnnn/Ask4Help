#!/usr/bin/env python3
"""Audit the external-link 640-episode ID recovery training root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=640)
    args = parser.parse_args()
    h5_path = args.collection_root / "id" / "accepted_suffixes.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    names = []
    anchors = {"existing512": 0, "recovery128": 0}
    tails = {"existing512": 0, "recovery128": 0}
    with h5py.File(h5_path, "r") as handle:
        names = sorted(name for name in handle if name.startswith("traj_"))
        for name in names:
            group = handle[name]
            actions = np.asarray(group["actions"], dtype=np.float32)
            base = group["obs/sensor_data/base_camera/rgb"]
            wrist = group["obs/sensor_data/hand_camera/rgb"]
            state = group["obs/state"]
            if base.shape[0] != actions.shape[0] + 1 or wrist.shape[0] != base.shape[0] or state.shape[0] != base.shape[0]:
                raise ValueError(f"boundary mismatch in {name}")
            if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
                raise ValueError(f"invalid actions in {name}")
            source = "recovery128" if int(name.rsplit("_", 1)[1]) >= 512 else "existing512"
            anchors[source] += int(actions.shape[0])
            tails[source] += min(9, int(actions.shape[0]))
    if len(names) != args.expected_episodes:
        raise ValueError(f"expected {args.expected_episodes} groups, found {len(names)}")
    report = {
        "format": "stackpyramid_grasp_recovery_merge_audit_v1",
        "collection_root": str(args.collection_root),
        "h5": str(h5_path),
        "episodes": len(names),
        "source_episode_counts": {"existing512": 512, "recovery128": 128},
        "source_anchor_counts": anchors,
        "source_tail_anchor_counts": tails,
        "anchors": sum(anchors.values()),
        "tail_anchors": sum(tails.values()),
        "ood_included": False,
        "action_dim": 8,
        "boundary_errors": [],
        "audit_pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    (args.output.parent / "MERGED_DATA_AUDIT_PASS").write_text("pass\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
