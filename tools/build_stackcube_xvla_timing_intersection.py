#!/usr/bin/env python3
"""Build the common successful-seed intersection across timing conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.run_xvla_stackcube_stage2_training import METHODS


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def common_seeds(root: Path) -> list[int]:
    ordered = [int(row["seed"]) for row in read_jsonl(root / METHODS[0] / "training_episodes.jsonl")]
    sets = [
        {int(row["seed"]) for row in read_jsonl(root / method / "training_episodes.jsonl")}
        for method in METHODS
    ]
    return [seed for seed in ordered if all(seed in values for values in sets)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    seeds = common_seeds(args.collections)
    if not seeds:
        raise RuntimeError("timing conditions have no common successful seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "format": "xvla_stackcube_timing_success_intersection_v1",
        "methods": list(METHODS),
        "count": len(seeds),
        "seeds": seeds,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
