#!/usr/bin/env python3
"""Strictly recompute AUCPR/AUCPDT from already-scored LIBERO-Plus traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load(TOOLS / "libero_plus_failure_protocol.py", "strict_libero_plus_failure_protocol")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def empirical_cdf(values: Sequence[float]) -> dict[float, float]:
    """Return tie-aware mid-rank CDF values in the interval (0, 1)."""

    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("rank transformation requires scores")
    result: dict[float, float] = {}
    start = 0
    while start < ordered.size:
        end = start + 1
        while end < ordered.size and ordered[end] == ordered[start]:
            end += 1
        result[float(ordered[start])] = float((start + 1 + end) / (2.0 * ordered.size))
        start = end
    return result


def attach_strict_rank_fusion(records: list[dict[str, Any]]) -> None:
    """Apply VLA-FAIL's uniform ranks and logical OR at every decision point."""

    final_values = [float(value) for record in records for value in record["scores"]["final_llmd"]]
    acc_values = [float(value) for record in records for value in record["scores"]["acc"]]
    final_cdf = empirical_cdf(final_values)
    acc_cdf = empirical_cdf(acc_values)
    for record in records:
        final_trace = record["scores"]["final_llmd"]
        acc_trace = record["scores"]["acc"]
        fused = []
        for index, final_score in enumerate(final_trace):
            # ACC at current replanning point compares this chunk with the
            # previous one, so the saved ACC trace is one element shorter.
            acc_score = acc_trace[index - 1] if index > 0 and index - 1 < len(acc_trace) else None
            acc_anomaly_rank = 0.0 if acc_score is None else acc_cdf[float(acc_score)]
            fused.append(max(final_cdf[float(final_score)], acc_anomaly_rank))
        record["scores"]["vla_fail_rank_or_strict"] = fused


def protocol_records(records: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    return [
        {"episode_id": row["episode_id"], "success": bool(row["success"]), "scores": row["scores"][method]}
        for row in records
        if row["scores"].get(method)
    ]


def independent_metrics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    metrics = PROTOCOL.threshold_independent_metrics(rows)
    return {
        "roc_auc": metrics["roc_auc"],
        "aucpr": metrics["average_precision"],
        "aucpdt": metrics["aucpdt"],
    }


def fixed_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return PROTOCOL.fixed_threshold_metrics(PROTOCOL.evaluate_fixed_threshold(rows, threshold=threshold))


def summarize(records: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    methods = [name for name in thresholds["thresholds"] if any(row["scores"].get(name) for row in records)]
    result: dict[str, Any] = {}
    for method in methods:
        rows = protocol_records(records, method)
        result[method] = {
            "fixed_threshold": fixed_metrics(rows, float(thresholds["thresholds"][method]["threshold"])),
            "threshold_independent": independent_metrics(rows),
        }

    deployed_rows = protocol_records(records, "vla_fail_final_or_acc")
    strict_rows = protocol_records(records, "vla_fail_rank_or_strict")
    result["vla_fail"] = {
        "fixed_threshold": fixed_metrics(deployed_rows, 1.0),
        "threshold_independent": independent_metrics(strict_rows),
        "fixed_score": "max(final_llmd/final_threshold, aligned_acc/acc_threshold)",
        "threshold_independent_score": "1 - min(uniform survival rank(final_llmd), uniform survival rank(acc)); emitted as max empirical CDF",
    }
    return result


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Strict LIBERO-Plus Offline Rescore",
        "",
        f"Episodes: {payload['episodes']} ({payload['successes']} success, {payload['failures']} failure).",
        "",
        "| Method | ROC-AUC | AUCPR | AUCPDT (lower) | Fixed recall | Fixed FPR | Fixed precision | Fixed BA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    pct = lambda value: "NA" if value is None else f"{100.0 * value:.2f}%"
    for method, row in payload["methods"].items():
        fixed, independent = row["fixed_threshold"], row["threshold_independent"]
        lines.append(
            f"| {method} | {pct(independent['roc_auc'])} | {pct(independent['aucpr'])} | "
            f"{pct(independent['aucpdt'])} | {pct(fixed['recall'])} | {pct(fixed['fpr'])} | "
            f"{pct(fixed['precision'])} | {pct(fixed['balanced_accuracy'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-episodes", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    records = json.loads(args.scored_episodes.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if not records:
        raise ValueError("scored episode list is empty")
    attach_strict_rank_fusion(records)
    payload = {
        "format": "libero_plus_strict_offline_rescore_v1",
        "episodes": len(records),
        "successes": sum(bool(row["success"]) for row in records),
        "failures": sum(not bool(row["success"]) for row in records),
        "protocol": {
            "aucpdt_thresholds": "all unique timestep scores",
            "aucpdt_pareto_axes": "precision and PDT",
            "aucpdt_integration": "right Riemann over precision from failure rate to one",
            "rank_pool": "all valid timestep scores in this retrospective evaluation set, separately for final LLMD and ACC",
        },
        "input_sha256": {
            "scored_episodes": sha256(args.scored_episodes),
            "thresholds": sha256(args.thresholds),
        },
        "methods": summarize(records, thresholds),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
