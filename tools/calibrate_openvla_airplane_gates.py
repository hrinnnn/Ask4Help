#!/usr/bin/env python3
"""Calibrate PCA and DiffDAgger temporal gates from successful ID rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openvla_airplane.gated import calibrate_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-successes", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_successes < 1:
        raise ValueError("minimum-successes must be positive")
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    successful = [row for row in payload["rows"] if row["ever_grasped"]]
    if len(successful) < args.minimum_successes:
        raise ValueError(
            f"expected at least {args.minimum_successes} successful ID trajectories, got {len(successful)}"
        )
    successful = successful[: max(20, args.minimum_successes)]

    def sequences(method: str) -> list[list[float]]:
        return [
            [float(point["scores"][method]) for point in row["timeline"] if method in point.get("scores", {})]
            for row in successful
        ]

    result = {
        "source": str(args.summary),
        "successful_id_trajectories": len(successful),
        "minimum_successes": args.minimum_successes,
        "siglip_pca": calibrate_gate(sequences("siglip_pooled_residual_pca"), args.quantile, args.patience),
        "diffdagger": calibrate_gate(sequences("c10_action_total_variance"), args.quantile, args.patience),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["threshold"] for key, value in result.items() if isinstance(value, dict) and "threshold" in value}, indent=2))


if __name__ == "__main__":
    main()
