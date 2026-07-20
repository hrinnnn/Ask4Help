#!/usr/bin/env python3
"""Select one paired SFT step that is weak-but-viable for both VFD members."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from select_pi05_checkpoint import success_rate


def select_common_step(member_0: dict[int, float], member_1: dict[int, float]) -> dict[str, object]:
    common = sorted(set(member_0) & set(member_1))
    if not common:
        raise ValueError("No checkpoint step was evaluated for both members")
    records = [
        {"step": step, "member_0_success_rate": member_0[step], "member_1_success_rate": member_1[step]}
        for step in common
    ]
    preferred = [
        record
        for record in records
        if 0.25 <= record["member_0_success_rate"] <= 0.50
        and 0.25 <= record["member_1_success_rate"] <= 0.50
    ]
    viable = [
        record
        for record in records
        if record["member_0_success_rate"] >= 0.25 and record["member_1_success_rate"] >= 0.25
    ]
    selected = preferred[0] if preferred else viable[0] if viable else records[-1]
    return {"selected_step": selected["step"], "selected": selected, "all": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member-0", action="append", nargs=2, metavar=("STEP", "DIR"), required=True)
    parser.add_argument("--member-1", action="append", nargs=2, metavar=("STEP", "DIR"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    member_0 = {int(step): success_rate(Path(directory)) for step, directory in args.member_0}
    member_1 = {int(step): success_rate(Path(directory)) for step, directory in args.member_1}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(select_common_step(member_0, member_1), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
