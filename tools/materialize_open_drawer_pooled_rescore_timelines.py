#!/usr/bin/env python3
"""Materialize one pooled-rescore timeline JSON per asset and episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads((args.summary).read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(payload["rows"]):
        timeline = row["timelines"][args.asset]
        materialized = {
            "format": "open_drawer_pooled_rescore_timeline_v1",
            "asset": args.asset,
            "split": row["split"],
            "seed": row["seed"],
            "source_success": row["source_success"],
            "first_drawer_opened_env_step": row.get("first_drawer_opened_env_step"),
            "first_alarm_env_step": row["first_alarm_env_step"][args.asset],
            "thresholds": payload.get("thresholds", {}).get(args.asset, {}),
            "timeline": timeline,
        }
        (args.output_dir / f"episode_{index:06d}.json").write_text(
            json.dumps(materialized, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"output_dir": str(args.output_dir), "asset": args.asset, "episodes": len(payload["rows"])}, indent=2))


if __name__ == "__main__":
    main()
