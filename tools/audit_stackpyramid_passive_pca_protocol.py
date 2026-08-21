#!/usr/bin/env python3
"""Audit provenance and paired reset evidence before PCA metrics registration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {
        "stage2_id": ("id", 892100, None),
        "stage2_ood": ("stage2_ood", 892100, "green"),
        "stage3_id": ("id", 893100, None),
        "stage3_ood": ("stage3_ood", 893100, "blue"),
    }
    errors: list[str] = []
    reports: dict[str, object] = {}
    rows_by_condition: dict[str, dict[int, dict]] = {}
    for name, (split, seed_start, affected) in conditions.items():
        folder = args.root / name
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        rows = {int(row["seed"]): row for row in summary.get("rows", [])}
        rows_by_condition[name] = rows
        expected = list(range(seed_start, seed_start + 100))
        if config.get("checkpoint", "").rstrip("/").split("/")[-1] != "ckpt-40000": errors.append(f"{name}: checkpoint mismatch")
        if config.get("split") != split or config.get("max_episode_steps") != 600 or config.get("execute_horizon") != 5: errors.append(f"{name}: config protocol mismatch")
        if set(rows) != set(expected): errors.append(f"{name}: seed manifest mismatch")
        if summary.get("geometry") != "v4" or summary.get("episodes") != 100: errors.append(f"{name}: summary geometry/denominator mismatch")
        if summary.get("passive_only") is not True or summary.get("expert_involved") is not False or summary.get("training_involved") is not False: errors.append(f"{name}: non-passive execution provenance")
        if summary.get("video_count") != 100 or summary.get("action_array_count") != 100 or summary.get("state_timeline_count") != 100: errors.append(f"{name}: incomplete evidence counts")
        bad_meta = [seed for seed, row in rows.items() if row.get("reset_metadata", {}).get("ood_geometry") != "v4" or row.get("reset_metadata", {}).get("runtime_max_episode_steps") != 600 or row.get("reset_metadata", {}).get("runtime_execute_horizon") != 5]
        if bad_meta: errors.append(f"{name}: runtime reset metadata mismatch for {len(bad_meta)} episodes")
        wrong_object = [seed for seed, row in rows.items() if row.get("reset_metadata", {}).get("affected_object") != affected]
        if wrong_object: errors.append(f"{name}: affected object metadata mismatch")
        reports[name] = {"episodes": len(rows), "bad_runtime_metadata": len(bad_meta), "affected_object": affected}
    for stage, target, shift, seed_start in (("stage2", "green", np.array([0.04, 0.03]), 892100), ("stage3", "blue", np.array([0.10, -0.12]), 893100)):
        id_rows, ood_rows = rows_by_condition[f"{stage}_id"], rows_by_condition[f"{stage}_ood"]
        pair_errors = []
        for seed in range(seed_start, seed_start + 100):
            a, b = id_rows.get(seed), ood_rows.get(seed)
            if a is None or b is None: continue
            pa, pb = a["reset_metadata"]["cube_poses"], b["reset_metadata"]["cube_poses"]
            colors = ("red", "blue") if target == "green" else ("red", "green")
            for color in colors:
                if not np.allclose(pa[color]["p"], pb[color]["p"], atol=1e-5): pair_errors.append(f"{seed}:{color}_changed")
            actual = np.asarray(pb[target]["p"][:2]) - np.asarray(pa[target]["p"][:2])
            if not np.allclose(actual, shift, atol=1e-5): pair_errors.append(f"{seed}:{target}_shift={actual.tolist()}")
            if not np.allclose(a["reset_metadata"]["robot_qpos"], b["reset_metadata"]["robot_qpos"], atol=1e-5): pair_errors.append(f"{seed}:robot_changed")
        if pair_errors: errors.append(f"{stage}: paired reset mismatch ({len(pair_errors)})")
        reports[f"{stage}_pairing"] = {"target": target, "shift": shift.tolist(), "errors": pair_errors[:10], "error_count": len(pair_errors)}
    calibration = json.loads((args.root / "calibration.json").read_text())
    if calibration.get("q") != 0.95 or calibration.get("fresh_env_per_episode") is not True or calibration.get("max_episode_steps") != 600: errors.append("calibration provenance mismatch")
    report = {"format": "stackpyramid_passive_pca_protocol_audit_v1", "status": "PASS" if not errors else "ENGINEERING_PROTOCOL_DIAGNOSTIC", "errors": errors, "conditions": reports, "calibration": {"q": calibration.get("q"), "threshold": calibration.get("threshold")}, "registration_blocked": bool(errors)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    marker = args.output.parent / ("PROTOCOL_AUDIT_PASS" if not errors else "ENGINEERING_PROTOCOL_DIAGNOSTIC")
    marker.write_text("complete\n")
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
