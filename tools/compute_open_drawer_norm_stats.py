#!/usr/bin/env python3
"""Compute frozen OpenPI state/action norm statistics from a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    episodes = [
        json.loads(line)
        for line in (args.dataset / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    chunks_size = int(info["chunks_size"])
    state_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    for episode in episodes:
        index = int(episode["episode_index"])
        path = args.dataset / "data" / f"chunk-{index // chunks_size:03d}" / f"episode_{index:06d}.parquet"
        table = pq.read_table(path, columns=["state", "actions"])
        state_batches.append(np.asarray(table["state"].to_pylist(), dtype=np.float64))
        action_batches.append(np.asarray(table["actions"].to_pylist(), dtype=np.float64))

    if not state_batches or not action_batches:
        raise RuntimeError("dataset contains no state/action rows")

    def stats(values: np.ndarray) -> dict[str, list[float]]:
        std = values.std(axis=0)
        std[std < 1e-8] = 1.0
        return {
            "mean": values.mean(axis=0).tolist(),
            "std": std.tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }

    payload = {
        "norm_stats": {
            "state": stats(np.concatenate(state_batches, axis=0)),
            "actions": stats(np.concatenate(action_batches, axis=0)),
        }
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "norm_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "provenance.json").write_text(
        json.dumps({"dataset": str(args.dataset), "episodes": len(episodes), "frozen": True}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
