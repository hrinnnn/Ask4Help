#!/usr/bin/env python3
"""Advance the corrected StackPyramid protocol through audit and calibration gates only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def allocate(base: Path, marker: str) -> Path:
    if (base / marker).is_file():
        return base
    if not base.exists():
        return base
    for index in range(1, 100):
        candidate = base.with_name(f"{base.name}_retry{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"no fresh output available for {base}")


def run_stage(label: str, command: list[str], gpu: int, cpu_set: str, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"label": label, "command": command, "gpu": gpu, "cpu_set": cpu_set}) + "\n")
        stream.flush()
        process = subprocess.Popen(
            ["taskset", "-c", cpu_set, *command],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        rc = process.wait()
        stream.write(json.dumps({"label": label, "return_code": rc}) + "\n")
    if rc not in (0, -6):
        raise RuntimeError(f"{label} failed with return code {rc}; see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, required=True)
    parser.add_argument("--pca-asset", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--cpu-set", default="80-99")
    parser.add_argument("--locality-gpus", nargs=2, default=("4", "5"))
    parser.add_argument("--locality-cpu-sets", nargs=2, default=("80-99", "100-119"))
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    state = root / "preflight_state.json"
    log = root / "logs" / "preflight.log"

    def write_state(**updates: object) -> None:
        current = json.loads(state.read_text()) if state.is_file() else {"format": "stackpyramid_protocol_preflight_v1"}
        current.update(updates)
        current["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        state.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")

    try:
        write_state(stage="wait_for_protocol_audit", audit_root=str(args.audit_root))
        while True:
            if (args.audit_root / "PROTOCOL_AUDIT_COMPLETE").is_file():
                break
            if (args.audit_root / "PROTOCOL_AUDIT_FAILED").is_file():
                raise RuntimeError("protocol audit failed; refusing calibration and collection")
            time.sleep(60)

        audit = json.loads((args.audit_root / "audit.json").read_text())
        if not all(audit.get("gates", {}).values()):
            raise RuntimeError(f"audit gates are not all true: {audit.get('gates')}")
        write_state(stage="pca_calibration")
        pca_dir = allocate(root / "calibration" / "bridge_pca", "PCA_CALIBRATION_COMPLETE")
        pca_json = pca_dir / "calibration.json"
        if not pca_json.is_file():
            run_stage(
                "calibrate_pca",
                [str(args.python), str(args.worktree / "tools/calibrate_stackpyramid_bridge_pca.py"),
                 "--checkpoint", str(args.checkpoint), "--xvla-root", str(args.xvla_root),
                 "--asset", str(args.pca_asset), "--output", str(pca_json),
                 "--successful-rollouts", "25", "--max-attempts", "60", "--start-seed", "45000",
                 "--flow-steps", "5", "--sim-backend", "cpu", "--render-backend", "cpu"],
                args.gpu, args.cpu_set, log,
            )
            (pca_dir / "PCA_CALIBRATION_COMPLETE").write_text("complete\n")

        write_state(stage="diff_calibration", pca_calibration=str(pca_json))
        diff_dir = allocate(root / "calibration" / "diffdagger", "DIFF_CALIBRATION_COMPLETE")
        diff_json = diff_dir / "calibration.json"
        if not diff_json.is_file():
            run_stage(
                "calibrate_diffdagger",
                [str(args.python), str(args.worktree / "tools/calibrate_stackpyramid_diffdagger.py"),
                 "--checkpoint", str(args.checkpoint), "--xvla-root", str(args.xvla_root),
                 "--output", str(diff_json), "--successful-rollouts", "25", "--max-attempts", "60",
                 "--start-seed", "46000", "--flow-steps", "5", "--diff-timesteps", "16",
                 "--max-episode-steps", "150", "--sim-backend", "cpu", "--render-backend", "cpu"],
                args.gpu, args.cpu_set, log,
            )
            (diff_dir / "DIFF_CALIBRATION_COMPLETE").write_text("complete\n")

        write_state(stage="stage_locality_gate", pca_calibration=str(pca_json), diff_calibration=str(diff_json))
        locality = allocate(root / "stage_locality_gate", "STAGE_LOCALITY_GATE_COMPLETE")
        if not (locality / "STAGE_LOCALITY_GATE_COMPLETE").is_file():
            seed_manifest = root / "stage_locality_seed_manifest.json"
            if not seed_manifest.is_file():
                seed_manifest.write_text(json.dumps({
                    "id": 74000,
                    "stage1_ood": 75000,
                    "stage2_ood": 76000,
                    "stage3_ood": 77000,
                }, indent=2) + "\n", encoding="utf-8")
            run_stage(
                "stage_locality_gate",
                [str(args.python), str(args.worktree / "tools/run_stackpyramid_stage_locality_gate.py"),
                 "--output-root", str(locality), "--repo-root", str(args.worktree),
                 "--xvla-root", str(args.xvla_root), "--checkpoint", str(args.checkpoint),
                 "--python", str(args.python), "--seed-manifest", str(seed_manifest),
                 "--gpus", *args.locality_gpus, "--cpu-sets", *args.locality_cpu_sets],
                args.gpu, args.cpu_set, log,
            )
            if not (locality / "STAGE_LOCALITY_GATE_COMPLETE").is_file():
                raise RuntimeError("stage locality gate did not produce STAGE_LOCALITY_GATE_COMPLETE")

        write_state(stage="pca_mixed_stream_pilot", pca_calibration=str(pca_json), diff_calibration=str(diff_json), locality_gate=str(locality))
        pca_threshold = float(json.loads(pca_json.read_text())["threshold"])
        pilot = allocate(root / "pca_mixed_stream_pilot", "PILOT_COMPLETE")
        if not (pilot / "PILOT_COMPLETE").is_file():
            run_stage(
                "pca_mixed_stream_pilot",
                [str(args.python), str(args.worktree / "tools/collect_stackpyramid_xvla_dagger.py"),
                 "--method", "bridge_pca", "--checkpoint", str(args.checkpoint), "--xvla-root", str(args.xvla_root),
                 "--asset", str(args.pca_asset), "--pca-threshold", str(pca_threshold),
                 "--output-dir", str(pilot), "--split", "stage1_ood", "--target", "20",
                 "--id-seed", "91000", "--ood-seed", "92000", "--max-attempts", "200",
                 "--min-ood-fraction", "0.80",
                 "--flow-steps", "5", "--sim-backend", "cpu", "--render-backend", "cpu"],
                args.gpu, args.cpu_set, log,
            )
            if not (pilot / "COLLECTION_COMPLETE").is_file():
                raise RuntimeError("PCA pilot did not produce COLLECTION_COMPLETE")
            summary = json.loads((pilot / "summary.json").read_text())
            accepted = summary.get("accepted_by_split", {})
            fraction = int(accepted.get("stage1_ood", 0)) / max(1, int(summary.get("accepted_total", 0)))
            if int(summary.get("accepted_total", 0)) != 20 or fraction < 0.80:
                raise RuntimeError(f"PCA pilot is not OOD-dominant: {accepted}")
            (pilot / "PILOT_COMPLETE").write_text("complete\n")
        summary = json.loads((pilot / "summary.json").read_text())
        accepted = summary.get("accepted_by_split", {})
        fraction = int(accepted.get("stage1_ood", 0)) / max(1, int(summary.get("accepted_total", 0)))
        locality_report = json.loads((locality / "stage_locality_gate.json").read_text())
        audit_report = {
            "format": "stackpyramid_protocol_gate_report_v1",
            "audit": str((args.audit_root / "audit.json").resolve()),
            "oracle": {
                split: {
                    "episodes": int(value["episodes"]),
                    "strict_success": int(value["strict_success"]),
                    "success_rate": int(value["strict_success"]) / max(1, int(value["episodes"])),
                    "summary": str((args.audit_root / "oracle" / split / "summary.json").resolve()),
                }
                for split, value in audit["oracle"].items()
            },
            "base_policy": {
                split: {
                    "episodes": int(value["episodes"]),
                    "strict_success": int(value["strict_success"]),
                    "success_rate": int(value["strict_success"]) / max(1, int(value["episodes"])),
                    "summary": str((args.audit_root / "policy" / split / "summary.json").resolve()),
                }
                for split, value in audit["base_policy"].items()
            },
            "stage_locality": locality_report,
            "pca_pilot": {
                "accepted_total": int(summary["accepted_total"]),
                "accepted_by_split": summary.get("accepted_by_split", {}),
                "required_ood_fraction": 0.80,
                "actual_ood_fraction": fraction,
                "pass": int(summary["accepted_total"]) == 20 and fraction >= 0.80,
                "summary": str((pilot / "summary.json").resolve()),
            },
            "gates": {
                "oracle_pass": bool(audit["gates"]["oracle_pass"]),
                "base_policy_pass": bool(audit["gates"]["base_policy_pass"]),
                "stage_locality_pass": bool(locality_report["passed"]),
                "pca_pilot_pass": int(summary["accepted_total"]) == 20 and fraction >= 0.80,
            },
        }
        (root / "protocol_gate_report.json").write_text(json.dumps(audit_report, indent=2) + "\n", encoding="utf-8")
        if not all(audit_report["gates"].values()):
            raise RuntimeError(f"protocol gate report failed: {audit_report['gates']}")
        write_state(stage="preflight_complete", pca_pilot=str(pilot), locality_gate=str(locality), gate_report=str(root / "protocol_gate_report.json"))
        (root / "PREFLIGHT_COMPLETE").write_text("complete\n")
    except Exception as exc:
        write_state(stage="failed", error=repr(exc))
        (root / "PREFLIGHT_FAILED").write_text(repr(exc) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
