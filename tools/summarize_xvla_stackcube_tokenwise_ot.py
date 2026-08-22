#!/usr/bin/env python3
"""Summarize passive StackCube token-wise PCA/OT failure detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.pick_single_ycb_airplane_detector_protocol import (  # noqa: E402
    summary_for_method,
    threshold_free_summary,
    threshold_from_success_maxima,
)

METHODS = ("ot_cost", "aligned_topk_cost", "pca_topk_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    episodes = []
    for row in source["rows"]:
        traces = {
            method: [float(decision[method]) for decision in row["timeline"]]
            for method in METHODS
        }
        episodes.append(
            {
                "episode_index": row["episode_index"],
                "split": row["split"],
                "success": bool(row["success"]),
                "execute_horizon": int(source["execute_horizon"]),
                "scores": traces,
            }
        )

    successful_id = [row for row in episodes if row["split"] == "id" and row["success"]]
    if not successful_id:
        raise RuntimeError("q95 calibration requires at least one successful ID rollout")
    metrics = {}
    for method in METHODS:
        calibration = threshold_from_success_maxima(
            [row["scores"][method] for row in successful_id], q=args.q
        )
        fixed = summary_for_method(episodes, method, float(calibration["threshold"]))
        metrics[method] = {
            "calibration": calibration,
            "fixed_threshold": fixed,
            "threshold_free": threshold_free_summary(episodes, method),
        }

    result = {
        "format": "xvla_stackcube_tokenwise_ot_metrics_v1",
        "source": str(args.input),
        "failure_definition": "not strict task success",
        "calibration": "successful ID trajectory maxima only",
        "episodes": len(episodes),
        "id_episodes": sum(row["split"] == "id" for row in episodes),
        "ood_episodes": sum(row["split"] == "ood" for row in episodes),
        "id_successes": sum(row["split"] == "id" and row["success"] for row in episodes),
        "ood_successes": sum(row["split"] == "ood" and row["success"] for row in episodes),
        "metrics": metrics,
    }
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output / "METRICS_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
