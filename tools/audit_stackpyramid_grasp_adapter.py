#!/usr/bin/env python3
"""Audit the frozen custom StackPyramid action/gripper adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (args.baseline_root / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    arrays = [np.load(Path(row["actions"])) for row in rows]
    actions = np.concatenate(arrays, axis=0)
    gripper = actions[:, -1]
    report = {
        "format": "stackpyramid_grasp_adapter_audit_v1",
        "baseline_root": str(args.baseline_root),
        "episodes": len(rows),
        "action_shapes": [list(shape) for shape in sorted({tuple(array.shape) for array in arrays})],
        "real_action_dim": int(actions.shape[1]),
        "gripper_min": float(gripper.min()),
        "gripper_max": float(gripper.max()),
        "gripper_negative_fraction": float(np.mean(gripper < 0.0)),
        "gripper_positive_fraction": float(np.mean(gripper >= 0.0)),
        "finite_actions": bool(np.isfinite(actions).all()),
        "custom_adapter": {
            "action_mode": "auto",
            "real_action_dim": 8,
            "max_action_dim": 20,
            "num_actions": 10,
            "domain_id": 0,
        },
        "status": "DIAGNOSTIC_AUDIT_COMPLETE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
