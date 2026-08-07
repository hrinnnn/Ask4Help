#!/usr/bin/env python3
"""Summarize all airplane detector scores on one shared rollout set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from pick_single_ycb_airplane_detector_protocol import threshold_free_summary  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "pick_single_ycb_airplane_detector_rollouts_v1":
        raise ValueError(f"not an airplane detector rollout summary: {path}")
    return list(payload["rows"])


def _flatten(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        methods = sorted({name for point in row["timeline"] for name in point["scores"]})
        traces = {}
        for method in methods:
            trace = [
                float(point["scores"][method])
                for point in row["timeline"]
                if point["scores"].get(method) is not None
            ]
            if trace:
                traces[method] = trace
        result.append({
            "episode_index": row["episode_index"],
            "seed": row["seed"],
            "split": row["split"],
            "ever_grasped": bool(row["ever_grasped"]),
            "scores": traces,
        })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-summary", type=Path, required=True)
    parser.add_argument("--ood-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    episodes = _flatten(_load(args.id_summary) + _load(args.ood_summary))
    methods = sorted({name for episode in episodes for name in episode["scores"]})
    metrics = {method: threshold_free_summary(episodes, method) for method in methods}
    ranking = sorted(
        ({"method": method, **summary} for method, summary in metrics.items()),
        key=lambda row: (-1.0 if row["auprc"] is None else -float(row["auprc"]), row["method"]),
    )
    result = {
        "format": "pick_single_ycb_airplane_detector_auprc_v1",
        "success_label": "ever_grasped",
        "failure_label": "not ever_grasped",
        "episode_score": "maximum detector score over the trajectory",
        "episodes": len(episodes),
        "id_episodes": sum(row["split"] == "id" for row in episodes),
        "ood_episodes": sum(row["split"] == "ood" for row in episodes),
        "successes": sum(row["ever_grasped"] for row in episodes),
        "failures": sum(not row["ever_grasped"] for row in episodes),
        "ranking": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
