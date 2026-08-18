#!/usr/bin/env python3
"""Pure paired-reset smoke for the independent radial StackCube split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stackcube_stage1_radial_ood import (
    RADIAL_SHIFT_DISTANCE_M,
    radial_reset_record,
    sample_stage1_radial_paired_xy,
    validate_paired_reset_records,
)


def run(num_seeds: int, seed_start: int) -> dict:
    pairs = []
    for seed in range(seed_start, seed_start + num_seeds):
        rng = np.random.default_rng(seed)
        green_xy, red_id_xy, red_ood_xy = sample_stage1_radial_paired_xy(rng, 1)
        green, red_id, red_ood = green_xy[0], red_id_xy[0], red_ood_xy[0]
        pairs.append(
            {
                "id": radial_reset_record(
                    paired_seed=seed,
                    split="id",
                    green_xy=green,
                    red_id_xy=red_id,
                    red_xy=red_id,
                ),
                "ood": radial_reset_record(
                    paired_seed=seed,
                    split="stage1_radial_distance_ood",
                    green_xy=green,
                    red_id_xy=red_id,
                    red_xy=red_ood,
                ),
            }
        )
    validation = validate_paired_reset_records(pairs)
    validation.update(
        {
            "task": "StackCube Stage1 radial-distance OOD",
            "seed_start": seed_start,
            "num_seeds": num_seeds,
            "radial_shift_distance_m": RADIAL_SHIFT_DISTANCE_M,
            "paired_reset": True,
        }
    )
    return {"validation": validation, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=971000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiment_management/diagnostics/stackcube_stage1_radial_two_way/paired_reset_smoke.json"
        ),
    )
    args = parser.parse_args()
    report = run(args.num_seeds, args.seed_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["validation"], indent=2))
    if not report["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
