#!/usr/bin/env python3
"""Independently reconcile OpenDrawer timing collection, training and SR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np


ANCHORS_DEFAULT = (0, 50, 80, 120, 160, 220)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def audit_eval(path: Path, episodes: int) -> dict[str, Any]:
    errors: list[str] = []
    if not (path / "EVAL_COMPLETE").is_file():
        errors.append("missing EVAL_COMPLETE")
    result: dict[str, Any] = {}
    for split in ("id", "grasp_ood"):
        out = path / split
        summary_path = out / "summary.json"
        if not summary_path.is_file():
            errors.append(f"missing {summary_path}")
            continue
        payload = read_json(summary_path)
        rows = payload.get("rows", [])
        if payload.get("episodes") != episodes or len(rows) != episodes:
            errors.append(f"denominator mismatch in {summary_path}")
        if len(list((out / "videos").glob("*.mp4"))) != episodes:
            errors.append(f"video denominator mismatch in {out}")
        for row in rows:
            for key in ("actions", "states", "timeline", "reset_metadata", "video"):
                if not Path(str(row.get(key, ""))).is_file():
                    errors.append(f"missing {key} artifact in {summary_path}")
                    break
        result[split] = {
            "episodes": payload.get("episodes"),
            "successes": payload.get("successes"),
            "success_rate": payload.get("success_rate"),
            "drawer_opened_rate": payload.get("drawer_opened_rate"),
            "grasp_rate": payload.get("grasp_rate"),
            "lift_rate": payload.get("lift_rate"),
            "in_target_rate": payload.get("in_target_rate"),
        }
    return {"path": str(path), "splits": result, "errors": errors, "pass": not errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", default=list(ANCHORS_DEFAULT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[9301, 9302, 9303])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    required = ["TIMING_COLLECTION_COMPLETE", "BUDGET_AUDIT_PASS", "TIMING_TRAINING_COMPLETE", "TIMING_EVALUATION_COMPLETE"]
    missing = [name for name in required if not (args.root / name).is_file()]
    errors.extend(f"missing marker {name}" for name in missing)
    d_path = args.root / "formal" / "d_path_summary.json"
    if not d_path.is_file():
        errors.append("missing formal d_path_summary.json")
    else:
        d_summary = read_json(d_path)
    budget_path = args.root / "formal_budget" / "budget_manifest.json"
    if not budget_path.is_file():
        errors.append("missing budget_manifest.json")
        budget = None
    else:
        budget = read_json(budget_path)
        values = budget.get("selected_expert_actions", {})
        if not values or len(set(int(v) for v in values.values())) != 1:
            errors.append("selected expert-action budgets are not identical")

    training_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for anchor in args.anchors:
        condition = f"anchor_{anchor}"
        for seed in args.seeds:
            ckpt = args.root / "training" / condition / f"seed_{seed}" / "run" / "checkpoints" / "global_step_2500"
            if not (ckpt / "actor/model_state_dict/full_weights.pt").is_file():
                errors.append(f"missing full_weights for {condition}/seed_{seed}")
            if not (ckpt / "actor/dcp_checkpoint").is_dir():
                errors.append(f"missing dcp for {condition}/seed_{seed}")
            training_rows.append({"anchor": anchor, "seed": seed, "checkpoint": str(ckpt), "checkpoint_complete": (ckpt / "actor/model_state_dict/full_weights.pt").is_file() and (ckpt / "actor/dcp_checkpoint").is_dir()})
            evaluated = audit_eval(args.root / "evaluation" / condition / f"seed_{seed}", args.episodes)
            eval_rows.append({"anchor": anchor, "seed": seed, **evaluated})
            errors.extend(f"anchor={anchor} seed={seed}: {err}" for err in evaluated["errors"])

    by_anchor: dict[str, Any] = {}
    for anchor in args.anchors:
        rows = [row for row in eval_rows if row["anchor"] == anchor and row["pass"]]
        id_rates = [float(row["splits"]["id"]["success_rate"]) for row in rows]
        ood_rates = [float(row["splits"]["grasp_ood"]["success_rate"]) for row in rows]
        by_anchor[str(anchor)] = {
            "training_seeds": args.seeds,
            "evaluation_count": len(rows),
            "id_sr_mean": float(mean(id_rates)) if id_rates else None,
            "id_sr_std": float(stdev(id_rates)) if len(id_rates) > 1 else 0.0 if id_rates else None,
            "grasp_ood_sr_mean": float(mean(ood_rates)) if ood_rates else None,
            "grasp_ood_sr_std": float(stdev(ood_rates)) if len(ood_rates) > 1 else 0.0 if ood_rates else None,
            "per_seed_grasp_ood_sr": dict(zip((str(seed) for seed in args.seeds), ood_rates)),
        }
    payload = {
        "format": "open_drawer_grasp_timing_final_reconciliation_v1",
        "anchors": list(args.anchors),
        "training_seeds": args.seeds,
        "episodes_per_split": args.episodes,
        "d_path_summary": d_summary if d_path.is_file() else None,
        "budget_manifest": budget,
        "training": training_rows,
        "evaluation": eval_rows,
        "utility_by_anchor": by_anchor,
        "errors": errors,
        "pass": not errors,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if errors:
        (args.root / "INDEPENDENT_RECONCILIATION_FAILED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(1)
    lines = [
        "# OpenDrawer Grasp-OOD timing sweep final report",
        "",
        "| Anchor | ID SR mean±std | Grasp-OOD SR mean±std |",
        "|---:|---:|---:|",
    ]
    for anchor in args.anchors:
        row = by_anchor[str(anchor)]
        lines.append(f"| {anchor} | {row['id_sr_mean']:.3f}±{row['id_sr_std']:.3f} | {row['grasp_ood_sr_mean']:.3f}±{row['grasp_ood_sr_std']:.3f} |")
    (args.root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.root / "INDEPENDENT_RECONCILIATION_COMPLETE").write_text("all timing checkpoints/evaluations and denominators independently audited\n", encoding="utf-8")
    print(json.dumps({"pass": True, "utility_by_anchor": by_anchor}, indent=2), flush=True)


if __name__ == "__main__":
    main()
