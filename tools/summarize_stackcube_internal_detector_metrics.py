#!/usr/bin/env python3
"""Compute full passive failure metrics for saved StackCube detector traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from libero_plus_failure_protocol import (  # noqa: E402
    aucpdt,
    average_precision,
    evaluate_fixed_threshold,
    fixed_threshold_metrics,
    roc_auc,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_rollouts(path: Path, source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "stackcube_internal_detector_rollout_v1":
        raise ValueError(f"unexpected rollout format in {path}")
    expected = set(payload["metrics"])
    records = []
    for episode in payload["episodes"]:
        timeline = list(episode["timeline"])
        if not timeline:
            raise ValueError(f"empty timeline for seed {episode['seed']}")
        if any(set(row["scores"]) != expected for row in timeline):
            raise ValueError(f"detector set changed within seed {episode['seed']}")
        scores = {name: [float(row["scores"][name]) for row in timeline] for name in expected}
        if any(not np.isfinite(values).all() for values in scores.values()):
            raise ValueError(f"non-finite score in seed {episode['seed']}")
        records.append(
            {
                "episode_id": f"{source}_seed_{int(episode['seed']):06d}",
                "source": source,
                "success": bool(episode["success"]),
                "scores": scores,
            }
        )
    return payload, records


def _method_episodes(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [
        {"episode_id": row["episode_id"], "success": row["success"], "scores": row["scores"][method]}
        for row in records
    ]


def _threshold_independent(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    labels = [not bool(row["success"]) for row in episodes]
    maxima = [float(max(row["scores"])) for row in episodes]
    return {
        "roc_auc": roc_auc(labels, maxima),
        "aucpr": average_precision(labels, maxima),
        "aucpdt": aucpdt(episodes),
    }


def _evaluate_group(
    records: list[dict[str, Any]], methods: list[str], thresholds: Mapping[str, float]
) -> dict[str, Any]:
    result = {}
    for method in methods:
        episodes = _method_episodes(records, method)
        result[method] = {
            "fixed_threshold": fixed_threshold_metrics(
                evaluate_fixed_threshold(episodes, threshold=float(thresholds[method]))
            ),
            "threshold_independent": _threshold_independent(episodes),
        }
    return result


def build_summary(
    *,
    id_payload: Mapping[str, Any],
    ood_payload: Mapping[str, Any],
    threshold_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    id_records: list[dict[str, Any]],
    ood_records: list[dict[str, Any]],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    methods = sorted(threshold_payload["detectors"])
    if set(methods) != set(id_payload["metrics"]) or set(methods) != set(ood_payload["metrics"]):
        raise ValueError("threshold and rollout detector sets differ")
    if id_payload.get("detector_assets_sha256") != threshold_payload.get("detector_assets_sha256"):
        raise ValueError("ID rollout and threshold detector assets differ")
    if ood_payload.get("detector_assets_sha256") != threshold_payload.get("detector_assets_sha256"):
        raise ValueError("OOD rollout and threshold detector assets differ")
    thresholds = {name: float(threshold_payload["detectors"][name]["threshold"]) for name in methods}
    all_records = [*id_records, *ood_records]
    return {
        "format": "stackcube_internal_detector_full_metrics_v2",
        "scope": {
            "description": "Retrospective full metrics from the saved StackCube internal_detector_matrix_v1 traces.",
            "checkpoint": id_payload.get("checkpoint"),
            "detector_assets_sha256": threshold_payload.get("detector_assets_sha256"),
            "calibration": {
                "method": "existing q=0.95 trajectory-maximum split-conformal threshold",
                "attempts": int(threshold_payload["attempts"]),
                "successful_trajectories": len([row for row in calibration_payload["episodes"] if row["success"]]),
            },
            "id_episodes": len(id_records),
            "ood_episodes": len(ood_records),
            "successes": sum(row["success"] for row in all_records),
            "failures": sum(not row["success"] for row in all_records),
        },
        "input_sha256": dict(input_sha256),
        "thresholds": thresholds,
        "overall": _evaluate_group(all_records, methods, thresholds),
        "id": _evaluate_group(id_records, methods, thresholds),
        "ood": _evaluate_group(ood_records, methods, thresholds),
    }


def _percent(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# StackCube Internal Detector Full Metrics",
        "",
        f"Evaluation set: {summary['scope']['id_episodes']} ID + {summary['scope']['ood_episodes']} OOD trajectories; "
        f"{summary['scope']['successes']} successes and {summary['scope']['failures']} failures.",
        "",
        "| Detector | ROC-AUC | AUCPR | AUCPDT (lower) | Recall | FPR | Precision | Balanced accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, row in summary["overall"].items():
        fixed = row["fixed_threshold"]
        free = row["threshold_independent"]
        lines.append(
            f"| {method} | {_percent(free['roc_auc'])} | {_percent(free['aucpr'])} | "
            f"{_percent(free['aucpdt'])} | {_percent(fixed['recall'])} | {_percent(fixed['fpr'])} | "
            f"{_percent(fixed['precision'])} | {_percent(fixed['balanced_accuracy'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-episodes", type=Path, required=True)
    parser.add_argument("--ood-episodes", type=Path, required=True)
    parser.add_argument("--calibration-episodes", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    id_payload, id_records = _load_rollouts(args.id_episodes, "id")
    ood_payload, ood_records = _load_rollouts(args.ood_episodes, "ood")
    calibration_payload = json.loads(args.calibration_episodes.read_text(encoding="utf-8"))
    threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    summary = build_summary(
        id_payload=id_payload,
        ood_payload=ood_payload,
        threshold_payload=threshold_payload,
        calibration_payload=calibration_payload,
        id_records=id_records,
        ood_records=ood_records,
        input_sha256={
            "id_episodes": _sha256(args.id_episodes),
            "ood_episodes": _sha256(args.ood_episodes),
            "calibration_episodes": _sha256(args.calibration_episodes),
            "thresholds": _sha256(args.thresholds),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
