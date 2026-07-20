#!/usr/bin/env python3
"""Select the planned weak-but-viable pi0.5 checkpoint from ID evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def success_rate(path: Path) -> float:
    rows = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
    if not rows:
        raise ValueError(f"No evaluations in {path}")
    return sum(bool(row["success"]) for row in rows) / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", nargs=2, metavar=("STEP", "DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measured = sorted((int(step), success_rate(Path(directory))) for step, directory in args.evaluation)
    preferred = [item for item in measured if 0.25 <= item[1] <= 0.50]
    eligible = [item for item in measured if item[1] >= 0.25]
    selected = (preferred or eligible or measured[-1])[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"selected_step": selected[0], "selected_success_rate": selected[1], "all": measured}, indent=2) + "\n")


if __name__ == "__main__":
    main()
