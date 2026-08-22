#!/usr/bin/env python3
"""Restart-tolerant ID-then-OOD Oracle gate controller."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_state(path: Path, **updates: object) -> None:
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(updates)
    state["updated_at"] = now()
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def run_gate(root: Path, run_root: Path, split: str, seed_start: int) -> int:
    gate = root / "tools/run_pick_single_ycb_object_variation_oracle_gate.py"
    output = run_root / "oracle_gate_v4" / split
    log_path = run_root / "logs" / f"oracle_gate_v4_{split}.log"
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    overlay = run_root / "runtime_overlay" / "numpy126"
    env["PYTHONPATH"] = f"{overlay}:{root / 'RLinf'}"
    command = [
        sys.executable,
        str(gate),
        "--split",
        split,
        "--seed-start",
        str(seed_start),
        "--count",
        "20",
        "--min-success",
        "19",
        "--output-dir",
        str(output),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run_root = Path(
        "/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "pipeline_state.json"
    write_state(
        state_path,
        pipeline_id="pick_single_ycb_object_variation_pi05_v1",
        owner_thread="019ffbc4-f3a9-78f3-8684-e0b4cba3552a",
        authorized=True,
        current_stage="oracle_gate_v4_id",
        next_stage="oracle_gate_v4_ood",
        controller_pid=os.getpid(),
        run_root=str(run_root),
    )
    id_rc = run_gate(root, run_root, "id", 11000)
    if id_rc != 0:
        (run_root / "ORACLE_NOT_ACCEPTED").write_text(
            json.dumps({"split": "id", "returncode": id_rc}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_state(
            state_path,
            current_stage="oracle_gate_id_failed",
            next_stage="needs_user_decision_or_preapproved_recovery",
            terminal_marker="ORACLE_NOT_ACCEPTED",
        )
        raise SystemExit(id_rc)

    write_state(state_path, current_stage="oracle_gate_v4_ood", next_stage="oracle_gate_v4_complete")
    ood_rc = run_gate(root, run_root, "ood", 12000)
    if ood_rc != 0:
        (run_root / "ORACLE_NOT_ACCEPTED").write_text(
            json.dumps({"split": "ood", "returncode": ood_rc}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_state(
            state_path,
        current_stage="oracle_gate_v4_ood_failed",
            next_stage="needs_user_decision_or_preapproved_recovery",
            terminal_marker="ORACLE_NOT_ACCEPTED",
        )
        raise SystemExit(ood_rc)

    (run_root / "ORACLE_GATE_PASSED").write_text(
        json.dumps({"id": "oracle_gate_v4/id", "ood": "oracle_gate_v4/ood"}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_state(
        state_path,
        current_stage="oracle_gate_v4_complete",
        next_stage="id_collection",
        terminal_marker="ORACLE_GATE_PASSED",
    )


if __name__ == "__main__":
    main()
