#!/usr/bin/env python3
"""Post-hoc oracle threshold scan for airplane token-wise PCA rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from pick_single_ycb_airplane_detector_protocol import summary_for_method
from pick_single_ycb_airplane_tokenwise_pca import MAIN_METHODS, sha256
from libero_plus_failure_protocol import aucpdt, penalized_detection_time_curve


def _records(episodes: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    result = []
    for episode in episodes:
        trace = [float(point["scores"][method]) for point in episode["timeline"]]
        if not trace:
            continue
        result.append({
            "episode_id": str(episode["episode_index"]), "success": bool(episode["ever_grasped"]),
            "execute_horizon": 5, "scores": {method: trace}, "split": episode["split"],
        })
    if not result:
        raise ValueError(f"no non-empty score traces for {method}")
    return result


def _choose(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get(key) is not None]
    if not usable:
        raise ValueError(f"no threshold row exposes {key}")
    return max(
        usable,
        key=lambda row: (
            float(row[key]),
            -float(row["success_conditioned_false_alarm_rate"] if row["success_conditioned_false_alarm_rate"] is not None else 1.0),
            float(row["threshold"]),
        ),
    )


def _scan(records: list[dict[str, Any]], method: str) -> dict[str, Any]:
    candidates = sorted({max(row["scores"][method]) for row in records})
    rows = [{"threshold": float(value), **summary_for_method(records, method, float(value))} for value in candidates]
    maxima = [max(row["scores"][method]) for row in records]
    pdt_records = [
        {
            "episode_id": row["episode_id"],
            "success": row["success"],
            "scores": row["scores"][method],
        }
        for row in records
    ]
    return {
        "candidate_count": len(rows), "threshold_rows": rows,
        "best_balanced_accuracy": _choose(rows, "balanced_accuracy"),
        "best_f1": _choose(rows, "f1"),
        "auroc": summary_for_method(records, method, candidates[0])["auroc"],
        "auprc": summary_for_method(records, method, candidates[0])["auprc"],
        # Unlike trajectory classification, PDT needs every raw timestep score:
        # a non-maximum threshold can keep the same alerted episodes but alert
        # a failing episode earlier.
        "pdt_curve": penalized_detection_time_curve(pdt_records),
        "aucpdt": aucpdt(pdt_records),
        "episode_maxima": {"min": min(maxima), "max": max(maxima)},
    }


def _representatives(episodes: list[dict[str, Any]]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for split in ("id", "ood"):
        for status, expected in (("success", True), ("failure", False)):
            match = next((row for row in episodes if row["split"] == split and bool(row["ever_grasped"]) is expected), None)
            result[f"{split}_{status}"] = None if match is None else int(match["episode_index"])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite threshold scan: {args.output}")
    payload = json.loads(args.episodes.read_text(encoding="utf-8"))
    if payload.get("format") != "pick_single_ycb_airplane_tokenwise_pca_rollouts_v1":
        raise ValueError("not an airplane token-wise PCA rollout payload")
    episodes = list(payload["episodes"])
    result = {
        "format": "pick_single_ycb_airplane_tokenwise_pca_posthoc_scan_v1",
        "warning": "Post-hoc/oracle threshold scan: this same evaluation split supplies outcome labels and candidate thresholds.",
        "episodes": str(args.episodes), "episodes_sha256": sha256(args.episodes), "success_label": "ever_grasped",
        "methods": {method: _scan(_records(episodes, method), method) for method in MAIN_METHODS},
        "representative_episode_indices": _representatives(episodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({method: spec["best_balanced_accuracy"] for method, spec in result["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
