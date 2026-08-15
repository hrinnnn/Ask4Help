#!/usr/bin/env python3
"""Launch the corrected three-stage StackPyramid comparison after preflight gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--id-h5", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--pca-asset", type=Path, required=True)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--cpu-sets", default="80-99,100-119")
    parser.add_argument("--training-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "orchestrator_state.json"
    log = root / "logs" / "orchestrator.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    def state(**updates: object) -> None:
        current = json.loads(state_path.read_text()) if state_path.is_file() else {"format": "stackpyramid_corrected_pipeline_v1"}
        current.update(updates)
        current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state_path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")

    try:
        state(stage="wait_for_preflight", preflight_root=str(args.preflight_root))
        while True:
            if (args.preflight_root / "PREFLIGHT_COMPLETE").is_file():
                break
            if (args.preflight_root / "PREFLIGHT_FAILED").is_file():
                raise RuntimeError("preflight failed; refusing formal collection/training")
            time.sleep(60)

        preflight = json.loads((args.preflight_root / "preflight_state.json").read_text())
        pca_calibration = Path(preflight["pca_calibration"])
        diff_calibration = Path(preflight["diff_calibration"])
        pilot = Path(preflight["pca_pilot"])
        if not (pca_calibration.is_file() and diff_calibration.is_file() and pilot.is_dir()):
            raise RuntimeError("preflight marker exists but calibration/pilot paths are incomplete")
        state(stage="formal_pipeline", pca_calibration=str(pca_calibration), diff_calibration=str(diff_calibration), pilot=str(pilot))
        command = [
            str(args.python), str(args.worktree / "tools/run_stackpyramid_four_method_comparison.py"),
            "--output-root", str(root), "--repo-root", str(args.worktree), "--xvla-root", str(args.xvla_root),
            "--python", str(args.python), "--base-model", str(args.checkpoint), "--id-h5", str(args.id_h5),
            "--pca-asset", str(args.pca_asset), "--pca-calibration", str(pca_calibration),
            "--protocol-audit", str(args.audit_root), "--gpus", args.gpus, "--cpu-sets", args.cpu_sets,
            "--training-steps", str(args.training_steps), "--batch-size", str(args.batch_size),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(args.worktree), str(args.xvla_root)])
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"command": command}) + "\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=args.worktree, env=env, stdout=stream, stderr=subprocess.STDOUT)
            state(stage="formal_pipeline_running", controller_pid=process.pid)
            rc = process.wait()
            stream.write(json.dumps({"return_code": rc}) + "\n")
        if rc != 0:
            raise RuntimeError(f"corrected controller failed with return code {rc}")
        if not (root / "PIPELINE_COMPLETE").is_file():
            raise RuntimeError("corrected controller exited without PIPELINE_COMPLETE")
        state(stage="complete")
    except Exception as exc:
        state(stage="failed", error=repr(exc))
        (root / "PIPELINE_FAILED").write_text(repr(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
