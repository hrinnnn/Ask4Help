#!/usr/bin/env python3
"""Make an auditable, offline scorecard from legacy StackCube VLA-FAIL traces.

This never touches a simulator or policy checkpoint.  It converts the saved
per-chunk LLMD/ACC timelines into fixed-threshold and threshold-independent
trajectory metrics, while preserving the original calibration protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

# Allow both ``python tools/...`` and test-time module loading.
TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from libero_plus_failure_protocol import (
    aucpdt,
    average_precision,
    evaluate_fixed_threshold,
    fixed_threshold_metrics,
    roc_auc,
)


METHODS = ("llmd", "acc", "vla_fail_rank_or")
FIXED_THRESHOLD_METHODS = ("llmd", "acc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_rollouts(path: Path, *, source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "stackcube_vla_fail_rollout_v1":
        raise ValueError(f"unexpected StackCube rollout format in {path}")
    records: list[dict[str, Any]] = []
    for episode in payload.get("episodes", []):
        timeline = list(episode.get("timeline", []))
        if not timeline:
            raise ValueError(f"episode {episode.get('seed')} has no timeline")
        llmd = [float(row["llmd"]) for row in timeline]
        # The first ACC is undefined because no prior predicted chunk exists.
        # A score of zero cannot cross the positive calibrated ACC threshold.
        acc = [0.0 if row.get("acc_ema") is None else float(row["acc_ema"]) for row in timeline]
        if not (np.isfinite(llmd).all() and np.isfinite(acc).all()):
            raise ValueError(f"episode {episode.get('seed')} has non-finite detector scores")
        records.append(
            {
                "episode_id": f"{source}_seed_{int(episode['seed']):06d}",
                "source": source,
                "seed": int(episode["seed"]),
                "success": bool(episode["success"]),
                "llmd": llmd,
                "acc": acc,
                "chunks": len(timeline),
                "steps": int(episode["steps"]),
            }
        )
    if not records:
        raise ValueError(f"no episodes in {path}")
    return payload, records


def _uniform_cdf(values: Iterable[float]) -> dict[float, float]:
    """Mid-rank empirical CDF; ties receive the same uniform rank."""

    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("rank fusion requires at least one score")
    result: dict[float, float] = {}
    start = 0
    while start < ordered.size:
        end = start + 1
        while end < ordered.size and ordered[end] == ordered[start]:
            end += 1
        result[float(ordered[start])] = float((start + 1 + end) / (2.0 * ordered.size))
        start = end
    return result


def attach_rank_or(records: list[dict[str, Any]]) -> None:
    """Paper-style LLMD+ACC rank fusion, emitted as higher-is-more-anomalous.

    VLA-FAIL maps each detector to a uniform anomaly rank and takes the
    minimum in the lower-is-more-anomalous convention.  ``1-min(rank)`` is the
    equivalent higher-is-more-anomalous score used by this generic metric code.
    """

    llmd_cdf = _uniform_cdf(score for record in records for score in record["llmd"])
    acc_cdf = _uniform_cdf(score for record in records for score in record["acc"])
    for record in records:
        record["vla_fail_rank_or"] = [
            max(llmd_cdf[float(llmd)], acc_cdf[float(acc)])
            for llmd, acc in zip(record["llmd"], record["acc"], strict=True)
        ]


def method_episodes(records: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": record["episode_id"],
            "success": record["success"],
            "scores": record[method],
        }
        for record in records
    ]


def threshold_independent(episodes: list[dict[str, Any]]) -> dict[str, float | None]:
    labels = [not bool(episode["success"]) for episode in episodes]
    scores = [float(max(episode["scores"])) for episode in episodes]
    return {
        "roc_auc": roc_auc(labels, scores),
        "aucpr": average_precision(labels, scores),
        "aucpdt": aucpdt(episodes),
    }


def _fixed(records: list[dict[str, Any]], method: str, threshold: float) -> dict[str, Any]:
    return fixed_threshold_metrics(evaluate_fixed_threshold(method_episodes(records, method), threshold=threshold))


def _source_metrics(records: list[dict[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    metrics = {
        method: {
            "fixed_threshold": _fixed(records, method, float(thresholds[method])),
            "threshold_independent": threshold_independent(method_episodes(records, method)),
        }
        for method in FIXED_THRESHOLD_METHODS
    }
    # This is the VLA-FAIL paper's rank-based fusion for threshold-independent
    # metrics.  Its deployed operating point is the separately reported
    # calibrated raw-score OR gate, not an arbitrary threshold on fused ranks.
    metrics["vla_fail_rank_or"] = {
        "fixed_threshold": None,
        "threshold_independent": threshold_independent(method_episodes(records, "vla_fail_rank_or")),
    }
    return metrics


def build_summary(
    *,
    id_payload: Mapping[str, Any],
    ood_payload: Mapping[str, Any],
    thresholds_payload: Mapping[str, Any],
    calibration_payload: Mapping[str, Any],
    id_records: list[dict[str, Any]],
    ood_records: list[dict[str, Any]],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    threshold_keys = {"llmd_threshold", "acc_threshold"}
    if not threshold_keys <= set(thresholds_payload):
        raise ValueError("threshold file lacks LLMD/ACC thresholds")
    for payload, name in ((id_payload, "id"), (ood_payload, "ood"), (calibration_payload, "calibration")):
        if payload.get("protocol", {}).get("action_horizon") != thresholds_payload.get("action_horizon"):
            raise ValueError(f"{name} action horizon mismatches threshold protocol")
        if payload.get("protocol", {}).get("execute_horizon") != thresholds_payload.get("execute_horizon"):
            raise ValueError(f"{name} execute horizon mismatches threshold protocol")

    all_records = [*id_records, *ood_records]
    attach_rank_or(all_records)
    thresholds = {
        "llmd": float(thresholds_payload["llmd_threshold"]),
        "acc": float(thresholds_payload["acc_threshold"]),
        # Rank fusion is threshold-independent here.  The calibrated deployed
        # OR gate is reported separately as ``deployed_or`` below.
        "vla_fail_rank_or": 0.5,
    }
    deployed_or = []
    for record in all_records:
        deployed_or.append(
            {
                "episode_id": record["episode_id"],
                "success": record["success"],
                "scores": [
                    max(float(llmd) / thresholds["llmd"], float(acc) / thresholds["acc"])
                    for llmd, acc in zip(record["llmd"], record["acc"], strict=True)
                ],
            }
        )

    id_only = [record for record in all_records if record["source"] == "id"]
    ood_only = [record for record in all_records if record["source"] == "ood"]
    return {
        "format": "stackcube_vla_fail_legacy_metrics_v1",
        "scope": {
            "description": "Retrospective metrics from saved StackCube t=1 passive VLA-FAIL traces. Not directly comparable with t=0 full-reference-bank LIBERO results.",
            "checkpoint": thresholds_payload.get("checkpoint"),
            "fixed_prior_seed": thresholds_payload.get("fixed_prior_seed"),
            "action_horizon": thresholds_payload.get("action_horizon"),
            "execute_horizon": thresholds_payload.get("execute_horizon"),
            "test_episodes": len(all_records),
            "id_test_episodes": len(id_only),
            "ood_test_episodes": len(ood_only),
            "rank_fusion": "Empirical per-timepoint rank transform over the retrospective ID+OOD test trace pool; higher emitted score means more anomalous.",
            "fixed_threshold_calibration": {
                "method": "existing q=0.95 policy-success calibration",
                "successful_trajectories": len([episode for episode in calibration_payload["episodes"] if episode["success"]]),
                "attempts": len(calibration_payload["episodes"]),
            },
        },
        "input_sha256": dict(input_sha256),
        "fixed_thresholds": {
            "llmd": thresholds["llmd"],
            "acc": thresholds["acc"],
            "deployed_or": "LLMD >= llmd_threshold OR ACC >= acc_threshold",
        },
        "overall": _source_metrics(all_records, thresholds),
        "id": _source_metrics(id_only, thresholds),
        "ood": _source_metrics(ood_only, thresholds),
        "deployed_or_fixed_threshold": {
            "overall": fixed_threshold_metrics(evaluate_fixed_threshold(deployed_or, threshold=1.0)),
            "id": fixed_threshold_metrics(evaluate_fixed_threshold([record for record in deployed_or if record["episode_id"].startswith("id_")], threshold=1.0)),
            "ood": fixed_threshold_metrics(evaluate_fixed_threshold([record for record in deployed_or if record["episode_id"].startswith("ood_")], threshold=1.0)),
        },
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# StackCube Legacy VLA-FAIL Metrics",
        "",
        "This is an offline reconstruction from the saved `t=1`, action-horizon-10, execute-horizon-5 traces. It is not comparable to the later LIBERO `t=0` full-reference-bank protocol.",
        "",
        "## Test Set",
        "",
        f"- ID: {summary['scope']['id_test_episodes']} trajectories",
        f"- OOD: {summary['scope']['ood_test_episodes']} trajectories",
        f"- Existing fixed thresholds: LLMD={summary['fixed_thresholds']['llmd']:.6f}, ACC={summary['fixed_thresholds']['acc']:.6f}",
        "",
        "## Overall Metrics",
        "",
        "| Detector | ROC-AUC | AUCPR | AUCPDT (lower) | Fixed Recall | Fixed FPR | Fixed Precision | Fixed BA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, metrics in summary["overall"].items():
        fixed, ti = metrics["fixed_threshold"], metrics["threshold_independent"]
        fmt = lambda value: "NA" if value is None else f"{value:.4f}"
        if fixed is None:
            fixed = {"recall": None, "fpr": None, "precision": None, "balanced_accuracy": None}
        lines.append(
            f"| {method} | {fmt(ti['roc_auc'])} | {fmt(ti['aucpr'])} | {fmt(ti['aucpdt'])} | {fmt(fixed['recall'])} | {fmt(fixed['fpr'])} | {fmt(fixed['precision'])} | {fmt(fixed['balanced_accuracy'])} |"
        )
    fixed = summary["deployed_or_fixed_threshold"]["overall"]
    fmt = lambda value: "NA" if value is None else f"{value:.4f}"
    lines.extend([
        "",
        "## Existing Deployed OR Gate",
        "",
        "| TP | FP | TN | FN | Recall | FPR | Precision | F1 | BA | Mean normalized detection time | TWA |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| " + " | ".join(fmt(fixed[key]) if isinstance(fixed[key], float) or fixed[key] is None else str(fixed[key]) for key in ("tp", "fp", "tn", "fn", "recall", "fpr", "precision", "f1", "balanced_accuracy", "mean_normalized_detection_time", "twa")) + " |",
        "",
    ])
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
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    id_payload, id_records = read_rollouts(args.id_episodes, source="id")
    ood_payload, ood_records = read_rollouts(args.ood_episodes, source="ood")
    calibration_payload = json.loads(args.calibration_episodes.read_text(encoding="utf-8"))
    thresholds_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
    summary = build_summary(
        id_payload=id_payload,
        ood_payload=ood_payload,
        thresholds_payload=thresholds_payload,
        calibration_payload=calibration_payload,
        id_records=id_records,
        ood_records=ood_records,
        input_sha256={
            "id_episodes": sha256(args.id_episodes),
            "ood_episodes": sha256(args.ood_episodes),
            "calibration_episodes": sha256(args.calibration_episodes),
            "thresholds": sha256(args.thresholds),
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
