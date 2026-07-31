#!/usr/bin/env python3
"""Offline fixed-threshold diagnostics for already scored failure rollouts.

This tool deliberately does *not* calibrate a deployable detector.  It uses
outcome labels from an evaluation split to expose the operating points a score
could attain, which is useful for diagnosing a poor conformal threshold.  Any
threshold selected here must be validated on a disjoint split before being
reported as a generalization result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _load_protocol() -> Any:
    path = ROOT / "tools" / "libero_plus_failure_protocol.py"
    spec = importlib.util.spec_from_file_location("libero_plus_failure_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import failure protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load_protocol()
DEFAULT_FRACTIONS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)


def method_records(records: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    """Convert stored score traces into the shared metric-protocol records."""

    result: list[dict[str, Any]] = []
    for record in records:
        trace = [float(value) for value in record["scores"].get(method, [])]
        if not trace:
            continue
        result.append({"episode_id": str(record["episode_id"]), "success": bool(record["success"]), "scores": trace})
    if not result:
        raise ValueError(f"no usable traces for method {method}")
    return result


def evaluate(records: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    evaluated = PROTOCOL.evaluate_fixed_threshold(records, threshold=float(threshold))
    return {"threshold": float(threshold), **PROTOCOL.fixed_threshold_metrics(evaluated)}


def choose_best(candidates: Iterable[dict[str, Any]], *, key: str) -> dict[str, Any]:
    """Choose deterministically: highest target metric, then lower FPR, then higher threshold."""

    usable = [row for row in candidates if row.get(key) is not None]
    if not usable:
        raise ValueError(f"no candidate exposes metric {key}")
    return max(
        usable,
        key=lambda row: (
            float(row[key]),
            -float(row["fpr"] if row["fpr"] is not None else 1.0),
            float(row["threshold"]),
        ),
    )


def best_under_fpr(candidates: Iterable[dict[str, Any]], budget: float) -> dict[str, Any] | None:
    feasible = [row for row in candidates if row["fpr"] is not None and float(row["fpr"]) <= budget]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda row: (
            float(row["recall"] if row["recall"] is not None else -1.0),
            float(row["balanced_accuracy"] if row["balanced_accuracy"] is not None else -1.0),
            float(row["threshold"]),
        ),
    )


def scan_method(
    records: Sequence[Mapping[str, Any]], *, method: str, calibrated_threshold: float, fractions: Sequence[float]
) -> dict[str, Any]:
    """Evaluate requested fractions and every observable operating point up to calibration."""

    if calibrated_threshold < 0:
        raise ValueError("this diagnostic only supports non-negative thresholds")
    rows = method_records(records, method)
    fraction_rows = [
        {"fraction_of_calibrated_threshold": fraction, **evaluate(rows, calibrated_threshold * fraction)}
        for fraction in fractions
    ]
    trajectory_maxima = sorted({max(row["scores"]) for row in rows if 0.0 <= max(row["scores"]) <= calibrated_threshold})
    exact_rows = [evaluate(rows, threshold) for threshold in trajectory_maxima]
    # Always include the deployed conformal operating point, even if no test
    # maximum is exactly equal to it.
    if not any(row["threshold"] == calibrated_threshold for row in exact_rows):
        exact_rows.append(evaluate(rows, calibrated_threshold))
    exact_rows.sort(key=lambda row: float(row["threshold"]))
    return {
        "calibrated_threshold": calibrated_threshold,
        "fraction_sweep": fraction_rows,
        "exact_candidate_count": len(exact_rows),
        "best_balanced_accuracy": choose_best(exact_rows, key="balanced_accuracy"),
        "best_f1": choose_best(exact_rows, key="f1"),
        "best_recall_at_fpr": {
            "0.05": best_under_fpr(exact_rows, 0.05),
            "0.10": best_under_fpr(exact_rows, 0.10),
            "0.20": best_under_fpr(exact_rows, 0.20),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-episodes", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=("bridge_llmd", "bridge_deep_knn", "bridge_pca_residual"),
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=DEFAULT_FRACTIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    fractions = tuple(float(value) for value in args.fractions)
    if not fractions or any(value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("fractions must be in [0, 1]")
    records = json.loads(args.scored_episodes.read_text(encoding="utf-8"))
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "format": "libero_plus_failure_threshold_sweep_v1",
        "warning": "Post-hoc diagnostic: outcome labels from this evaluation split select candidates. Validate any choice on disjoint calibration/test data.",
        "scored_episodes": str(args.scored_episodes),
        "thresholds": str(args.thresholds),
        "methods": {},
    }
    for method in args.methods:
        spec = thresholds["thresholds"].get(method)
        if not isinstance(spec, Mapping):
            raise ValueError(f"threshold file has no base threshold for {method}")
        result["methods"][method] = scan_method(
            records,
            method=method,
            calibrated_threshold=float(spec["threshold"]),
            fractions=fractions,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
