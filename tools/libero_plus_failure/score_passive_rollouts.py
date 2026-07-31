#!/usr/bin/env python3
"""Replay raw LIBERO(-Plus) traces into a frozen no-training detector table."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
from libero_plus_failure.rollout_records import read_rollout  # noqa: E402


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ASSETS = _load(TOOLS / "libero_plus_failure_assets.py", "libero_plus_failure_assets")
PROTOCOL = _load(TOOLS / "libero_plus_failure_protocol.py", "libero_plus_failure_protocol")

BASE_METHODS = ("bridge_llmd", "bridge_deep_knn", "bridge_pca_residual", "final_llmd", "acc", "stac_single")
ALL_METHODS = BASE_METHODS + ("vla_fail_final_or_acc",)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def raw_rollout_dirs(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if (root / "episode.json").is_file():
            found.append(root)
        elif root.is_dir():
            found.extend(sorted(path.parent for path in root.rglob("episode.json")))
        else:
            raise FileNotFoundError(root)
    unique = sorted(set(found))
    if not unique:
        raise ValueError("no raw rollout directories found")
    return unique


def _optional_trace(timeline: list[Mapping[str, Any]], key: str) -> list[float]:
    values = [point.get(key) for point in timeline if point.get(key) is not None]
    return [float(value) for value in values]


def score_one(path: Path, assets: Mapping[str, Any]) -> dict[str, Any]:
    episode, features = read_rollout(path)
    timeline = list(episode["timeline"])
    values: dict[str, list[float]] = {name: [] for name in ASSETS.DETECTOR_NAMES}
    for index in range(len(timeline)):
        scores = ASSETS.score_features(
            {"bridge": features["bridge"][index], "action_expert_final": features["action_expert_final"][index]}, assets
        )
        for name, value in scores.items():
            values[name].append(float(value))
    values["acc"] = _optional_trace(timeline, "acc")
    values["stac_single"] = _optional_trace(timeline, "stac_single")
    return {
        "episode_path": str(path),
        "episode_id": path.name,
        "success": bool(episode["success"]),
        "suite": episode["suite"],
        "source": episode["source"],
        "category": episode.get("category"),
        "configuration_id": episode.get("configuration_id"),
        "task_index": int(episode["task_index"]),
        "seed": int(episode["seed"]),
        "video_path": str(path / "rollout.mp4"),
        "scores": values,
        "decision_steps": [int(point["env_step"]) for point in timeline],
        "runtime_ms": {
            "policy": [float(point["policy_ms"]) for point in timeline if math.isfinite(float(point["policy_ms"]))],
            "feature": [float(point["feature_ms"]) for point in timeline if math.isfinite(float(point["feature_ms"]))],
        },
    }


def calibrate(records: list[dict[str, Any]], assets_sha: str, required_successes: int) -> dict[str, Any]:
    successes = [record for record in records if record["success"]]
    usable = [record for record in successes if all(record["scores"][name] for name in BASE_METHODS)]
    if len(usable) < required_successes:
        raise ValueError(
            "need %d successful clean rollouts with ACC/STAC overlap, found %d" % (required_successes, len(usable))
        )
    selected = usable[:required_successes]
    traces = {name: [record["scores"][name] for record in selected] for name in BASE_METHODS}
    thresholds = ASSETS.conformal_thresholds(traces, delta=0.05)
    thresholds.update(
        {
            "reference_assets_sha256": assets_sha,
            "successful_policy_rollouts": len(selected),
            "selection": [record["episode_path"] for record in selected],
            "calibration_source": "clean_policy_success_only",
        }
    )
    return thresholds


def _union_score(final_trace: list[float], acc_trace: list[float], threshold: Mapping[str, Any]) -> list[float]:
    # Both components use their fixed clean-calibration operating points.  The
    # union score crosses one exactly when VLA-FAIL's OR gate would alert.
    final_t = float(threshold["thresholds"]["final_llmd"]["threshold"])
    acc_t = float(threshold["thresholds"]["acc"]["threshold"])
    result = []
    for index, final in enumerate(final_trace):
        acc = acc_trace[index - 1] if index > 0 and index - 1 < len(acc_trace) else 0.0
        result.append(max(float(final) / final_t, float(acc) / acc_t))
    return result


def annotate(records: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    values = thresholds["thresholds"]
    for record in records:
        copied = {**record, "scores": {name: list(trace) for name, trace in record["scores"].items()}}
        copied["scores"]["vla_fail_final_or_acc"] = _union_score(
            copied["scores"]["final_llmd"], copied["scores"]["acc"], thresholds
        )
        alerts: dict[str, int | None] = {}
        for name in BASE_METHODS:
            trace = copied["scores"][name]
            alerts[name] = None if not trace else PROTOCOL.first_alert_index(trace, float(values[name]["threshold"]))
        alerts["vla_fail_final_or_acc"] = PROTOCOL.first_alert_index(copied["scores"]["vla_fail_final_or_acc"], 1.0)
        copied["first_alert"] = alerts
        result.append(copied)
    return result


def _protocol_records(records: list[dict[str, Any]], method: str, threshold: float) -> list[dict[str, Any]]:
    return [{"episode_id": record["episode_id"], "success": record["success"], "scores": record["scores"][method]} for record in records if record["scores"][method]]


def _metric_summary(records: list[dict[str, Any]], method: str, threshold: float) -> dict[str, Any]:
    rows = _protocol_records(records, method, threshold)
    if not rows:
        return {"episodes": 0}
    fixed = PROTOCOL.fixed_threshold_metrics(PROTOCOL.evaluate_fixed_threshold(rows, threshold=threshold))
    independent = PROTOCOL.threshold_independent_metrics(rows)
    result = {"episodes": len(rows), **fixed, **independent}
    for key in ("balanced_accuracy", "weighted_accuracy", "recall", "precision", "f1", "roc_auc", "average_precision", "aucpdt"):
        def metric(sample: list[Mapping[str, Any]], key: str = key) -> float:
            if key in ("roc_auc", "average_precision", "aucpdt"):
                value = PROTOCOL.threshold_independent_metrics(sample)[key]
            else:
                value = PROTOCOL.fixed_threshold_metrics(PROTOCOL.evaluate_fixed_threshold(sample, threshold=threshold))[key]
            return float("nan") if value is None else float(value)
        try:
            result[key + "_bootstrap_95_ci"] = PROTOCOL.bootstrap_interval(rows, metric, seed=20260731, samples=1000)
        except ValueError:
            result[key + "_bootstrap_95_ci"] = None
    return result


def summarize(records: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"all": {}}
    for method in ALL_METHODS:
        threshold = 1.0 if method == "vla_fail_final_or_acc" else float(thresholds["thresholds"][method]["threshold"])
        result["all"][method] = _metric_summary(records, method, threshold)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault("source=" + str(record["source"]), []).append(record)
        groups.setdefault("category=" + str(record.get("category", "clean")), []).append(record)
        groups.setdefault("task=" + str(record["task_index"]), []).append(record)
        groups.setdefault("outcome=" + ("success" if record["success"] else "failure"), []).append(record)
    result["groups"] = {}
    for name, subset in sorted(groups.items()):
        result["groups"][name] = {}
        for method in ALL_METHODS:
            threshold = 1.0 if method == "vla_fail_final_or_acc" else float(thresholds["thresholds"][method]["threshold"])
            result["groups"][name][method] = _metric_summary(subset, method, threshold)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "evaluate"), required=True)
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--required-successes", type=int, default=100)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite " + str(args.output_dir))
    assets = torch.load(args.assets, map_location="cpu", weights_only=False)
    asset_sha = sha256(args.assets)
    records = [score_one(path, assets) for path in raw_rollout_dirs(args.raw_root)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.mode == "calibrate":
        thresholds = calibrate(records, asset_sha, args.required_successes)
        (args.output_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    else:
        if args.thresholds is None:
            raise ValueError("evaluate mode requires --thresholds")
        thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if thresholds.get("reference_assets_sha256") != asset_sha:
            raise ValueError("threshold/reference asset SHA mismatch")
        annotated = annotate(records, thresholds)
        (args.output_dir / "scored_episodes.json").write_text(json.dumps(annotated, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "summary.json").write_text(json.dumps(summarize(annotated, thresholds), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
