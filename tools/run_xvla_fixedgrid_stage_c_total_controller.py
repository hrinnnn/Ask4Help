#!/usr/bin/env python3
"""Durable Stage-C supervisor: passive audit -> gated utility -> reconciliation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch_data(args: argparse.Namespace, log: Path) -> int:
    airplane_internal_assets = (
        args.passive_root / "assets_airplane_input_pool" / "multilayer_detector_assets.pt"
    )
    if not airplane_internal_assets.is_file():
        airplane_internal_assets = args.airplane_internal_assets
    command = [
        args.python,
        str(args.repo / "tools/run_xvla_fixedgrid_stage_c_gate_data_controller.py"),
        "--manifest", str(args.manifest),
        "--passive-root", str(args.passive_root),
        "--run-root", str(args.data_root),
        "--repo", str(args.repo),
        "--xvla-root", str(args.xvla_root),
        "--python", args.python,
        "--gpu", str(args.gpu),
        "--cpu-set", args.cpu_set,
        "--interval-seconds", str(args.interval_seconds),
        "--stackcube-checkpoint", str(args.stackcube_checkpoint),
        "--stackcube-internal-assets", str(args.stackcube_internal_assets),
        "--airplane-checkpoint", str(args.airplane_checkpoint),
        "--airplane-internal-assets", str(airplane_internal_assets),
        "--airplane-pca-asset", str(args.airplane_pca_asset),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=args.repo,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)},
            start_new_session=True,
        )
    return process.pid


def run(args: argparse.Namespace) -> None:
    args.supervisor_root.mkdir(parents=True, exist_ok=True)
    state_path = args.supervisor_root / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {
        "pipeline_id": "xvla_fixedgrid_taskpolicy_knee_v1",
        "stage": "stage_c_total_waiting_for_passive_audit",
        "started_at": now(),
        "passive_root": str(args.passive_root),
        "data_root": str(args.data_root),
    }
    log = args.supervisor_root / "supervisor.log"
    passive_marker = args.passive_root / "STAGE_C_GATE_AUDIT_COMPLETE_DATA_PENDING"
    data_marker = args.data_root / "STAGE_C_GATE_UTILITY_COMPLETE"
    while not passive_marker.is_file():
        state.update({"stage": "stage_c_total_waiting_for_passive_audit", "updated_at": now()})
        write_json(state_path, state)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"{now()} waiting for passive Stage-C marker\n")
        time.sleep(args.interval_seconds)

    pid_path = args.supervisor_root / "data_controller.pid"
    while not data_marker.is_file():
        data_state_path = args.data_root / "pipeline_state.json"
        if data_state_path.is_file() and read_json(data_state_path).get("stage") == "stage_c_gate_failed":
            state.update({"stage": "stage_c_total_data_failed", "updated_at": now()})
            write_json(state_path, state)
            raise RuntimeError("Stage-C data controller failed; preserved its artifacts and state")
        pid = int(pid_path.read_text(encoding="utf-8")) if pid_path.exists() else 0
        if not pid or not pid_alive(pid):
            pid = launch_data(args, args.supervisor_root / "data_controller.log")
            pid_path.write_text(f"{pid}\n", encoding="utf-8")
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"{now()} launched Stage-C data controller pid={pid}\n")
        state.update({"stage": "stage_c_gate_data_running", "data_pid": pid, "updated_at": now()})
        write_json(state_path, state)
        time.sleep(args.interval_seconds)

    report = args.supervisor_root / "final_reconciliation.json"
    if not report.is_file() or not (args.base_root / "PIPELINE_COMPLETE").is_file():
        command = [
            args.python,
            str(args.repo / "tools/reconcile_xvla_fixedgrid_pipeline.py"),
            "--base-root", str(args.base_root),
            "--output", str(report),
        ]
        with (args.supervisor_root / "reconciliation.log").open("a", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=args.repo, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0 or not (args.base_root / "PIPELINE_COMPLETE").is_file():
            state.update({"stage": "stage_c_total_reconciliation_failed", "returncode": result.returncode, "updated_at": now()})
            write_json(state_path, state)
            raise RuntimeError("independent fixed-grid reconciliation failed; see reconciliation.log")
    state.update({"stage": "pipeline_complete", "completed_at": now(), "updated_at": now()})
    write_json(state_path, state)
    (args.supervisor_root / "STAGE_C_TOTAL_COMPLETE").write_text("complete\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--passive-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--supervisor-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--cpu-set", default="0-19")
    parser.add_argument("--interval-seconds", type=int, default=900)
    parser.add_argument("--stackcube-checkpoint", type=Path, required=True)
    parser.add_argument("--stackcube-internal-assets", type=Path, required=True)
    parser.add_argument("--airplane-checkpoint", type=Path, required=True)
    parser.add_argument("--airplane-internal-assets", type=Path, required=True)
    parser.add_argument("--airplane-pca-asset", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
