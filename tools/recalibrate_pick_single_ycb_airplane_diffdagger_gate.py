#!/usr/bin/env python3
"""Calibrate DiffDAgger at the same temporal-gate unit used online."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.collect_pick_single_ycb_airplane_gated_dagger import (
    calibrate_patience_gate_threshold,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    rows = [json.loads(line) for line in args.episodes.read_text().splitlines() if line]
    admitted = [row for row in rows if row.get("admitted_to_calibration")]
    sequences = [row["scores"] for row in admitted]
    threshold, episode_scores = calibrate_patience_gate_threshold(
        sequences, alpha=args.alpha, patience=args.patience
    )
    payload = {
        "format": "pick_airplane_diffdagger_temporal_gate_calibration_v1",
        "source_episodes": str(args.episodes.resolve()),
        "source_episodes_sha256": _sha256(args.episodes),
        "successful_id_trajectories": len(sequences),
        "alpha": args.alpha,
        "patience": args.patience,
        "threshold": threshold,
        "episode_gate_scores": episode_scores,
        "strict_calibration_alarms": sum(score > threshold for score in episode_scores),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
