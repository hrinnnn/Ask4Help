#!/usr/bin/env python3
"""Independent final reconciliation for the fixed-grid knee pipeline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METHODS = ["input_pca", "bridge_pca", "action_pca", "diffdagger", "failure_recovery"]
TRAINING_SEEDS = [17001, 17002, 17003]
TASKS = {"stackcube": {"budget": 520, "anchors": [0, 10, 20, 30, 45]}, "airplane": {"budget": 2820, "anchors": [0, 10, 20, 30]}}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path, label: str) -> None:
    if not path.is_file() and not path.is_dir():
        raise RuntimeError(f"missing {label}: {path}")


def require_marker(path: Path, label: str) -> None:
    require(path, label)


def check_summary(path: Path, episodes: int, label: str) -> dict[str, Any]:
    payload = read_json(path)
    if int(payload.get("episodes", -1)) != episodes:
        raise RuntimeError(f"{label} denominator mismatch: {path}")
    return payload


def reconcile(base: Path, output: Path) -> dict[str, Any]:
    manifest = base / "../.."  # only used to keep paths visibly rooted below
    calibration = {}
    for task, spec in TASKS.items():
        calibration_root = base / ("formal_calibration_merged_v2" if task == "stackcube" else "airplane_calibration_merged_v2")
        valid = calibration_root / "calibration_audit_valid.json"
        knee = calibration_root / "knee_summary_recoverable.json"
        require_marker(valid, f"{task} calibration audit")
        require_marker(knee, f"{task} knee summary")
        valid_payload = read_json(valid)
        knee_payload = read_json(knee)
        if valid_payload.get("anchors") != spec["anchors"]:
            raise RuntimeError(f"{task} calibration anchors mismatch")
        if int(knee_payload.get("knee_anchor", -1)) != 20:
            raise RuntimeError(f"{task} calibration knee is not the frozen step 20")
        for anchor in spec["anchors"]:
            dataset = calibration_root / f"timing_datasets_budget_{spec['budget']}" / f"step_{anchor}"
            info = dataset / "meta/info.json"
            require_marker(info, f"{task} Stage-B selected dataset step {anchor}")
            if int(read_json(info).get("total_frames", -1)) != spec["budget"]:
                raise RuntimeError(f"{task} Stage-B dataset step {anchor} budget mismatch")
        calibration[task] = {"audit": str(valid), "knee": str(knee), "knee_anchor": 20, "budget": spec["budget"]}

    stage_b_training = base / "stage_b_training_v1"
    stage_b_eval = base / "stage_b_evaluation_v1"
    stage_b_total = base / "stage_b_total_supervisor_v1"
    require_marker(stage_b_training / "STAGE_B_TRAINING_COMPLETE", "Stage-B training marker")
    require_marker(stage_b_eval / "STAGE_B_EVAL_COMPLETE", "Stage-B evaluation marker")
    require_marker(stage_b_total / "STAGE_B_UTILITY_COMPLETE", "Stage-B utility marker")
    train_state = read_json(stage_b_training / "pipeline_state.json")
    eval_state = read_json(stage_b_eval / "pipeline_state.json")
    if len(train_state.get("completed_jobs", [])) != 27:
        raise RuntimeError("Stage-B training job denominator mismatch")
    if len(eval_state.get("completed_evals", [])) != 54:
        raise RuntimeError("Stage-B evaluation job denominator mismatch")
    stage_b_utility = {
        task: read_json(stage_b_eval / "utility_summaries" / f"{task}.json")
        for task in TASKS
    }

    passive = base / "stage_c_gate_v1"
    require_marker(passive / "STAGE_C_GATE_AUDIT_COMPLETE_DATA_PENDING", "Stage-C passive gate marker")
    passive_evidence = {}
    for task in TASKS:
        for name in ("validation_id", "heldout_ood"):
            check_summary(passive / task / name / "summary.json", 50, f"Stage-C {task} {name}")
        calibration_path = passive / task / "calibration_q95.json"
        gate_path = passive / task / "gate_to_knee.json"
        require_marker(calibration_path, f"Stage-C {task} q=.95 calibration")
        require_marker(gate_path, f"Stage-C {task} gate-to-knee audit")
        gate_payload = read_json(gate_path)
        if set(gate_payload.get("methods", {})) != set(METHODS):
            raise RuntimeError(f"Stage-C {task} gate method set mismatch")
        passive_evidence[task] = {"calibration": str(calibration_path), "gate_to_knee": str(gate_path)}

    data = base / "stage_c_gate_data_v1"
    require_marker(data / "STAGE_C_GATE_UTILITY_COMPLETE", "Stage-C gate utility marker")
    data_state = read_json(data / "pipeline_state.json")
    if len(data_state.get("completed_collections", [])) != 10:
        raise RuntimeError("Stage-C collection denominator mismatch")
    if len(data_state.get("completed_selections", [])) != 10:
        raise RuntimeError("Stage-C exact-selection denominator mismatch")
    if len(data_state.get("completed_training", [])) != 30:
        raise RuntimeError("Stage-C training denominator mismatch")
    if len(data_state.get("completed_evaluations", [])) != 60:
        raise RuntimeError("Stage-C evaluation denominator mismatch")
    gate_utility = {}
    for task, spec in TASKS.items():
        gate_utility[task] = {}
        for method in METHODS:
            selected = data / "datasets" / task / method / "selected" / "meta/info.json"
            require_marker(selected, f"Stage-C {task}/{method} selected dataset")
            if int(read_json(selected).get("total_frames", -1)) != spec["budget"]:
                raise RuntimeError(f"Stage-C {task}/{method} exact budget mismatch")
            for seed in TRAINING_SEEDS:
                checkpoint = data / "training" / task / method / f"seed_{seed}" / "train/ckpt-2500/model.safetensors"
                require_marker(checkpoint, f"Stage-C {task}/{method}/{seed} checkpoint")
                for split in ("id", "ood"):
                    summary = data / "evaluation" / task / method / f"seed_{seed}" / split / "summary.json"
                    check_summary(summary, 100, f"Stage-C {task}/{method}/{seed}/{split}")
            gate_utility[task][method] = {
                "id": read_json(data / "utility_summaries" / f"{task}_id.json")["rows"],
                "ood": read_json(data / "utility_summaries" / f"{task}_ood.json")["rows"],
            }

    report = {
        "format": "xvla_fixedgrid_taskpolicy_knee_final_reconciliation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "frozen anchors, validation-ID q=.95 gates, whole-episode exact budgets, 2500-step 3-seed utility",
        "calibration": calibration,
        "stage_b": {
            "training_jobs": len(train_state["completed_jobs"]),
            "evaluation_jobs": len(eval_state["completed_evals"]),
            "utility": stage_b_utility,
        },
        "stage_c_passive": passive_evidence,
        "stage_c_gate_utility": gate_utility,
        "completion_definition": {
            "calibration": True,
            "passive_gate_audit": True,
            "matched_budget_training": True,
            "matched_budget_utility": True,
            "independent_denominator_reconciliation": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = reconcile(args.base_root, args.output)
    marker = args.base_root / "PIPELINE_COMPLETE"
    marker.write_text("independent reconciliation passed\n", encoding="utf-8")
    print(json.dumps({"format": report["format"], "pipeline_complete": True}, indent=2))


if __name__ == "__main__":
    main()
