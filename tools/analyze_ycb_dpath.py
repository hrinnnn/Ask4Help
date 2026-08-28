#!/usr/bin/env python3
"""Recompute a phase-aware TCP-position path deviation for PickSingleYCB.

This diagnostic deliberately separates path deviation from raw clock speed.  A
two-sided phase band is estimated from ID expert-to-expert DTW alignments and
then applied with a monotone, local alignment.  The output is intended for
the accompanying visual audit; it does not alter any formal detector or
training pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_xvla_erd_pose import _load_pose_root, _pose_residual, _pose_series


def _load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["episode_index"]): row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row}


def _global_dtw_map(query: dict[str, np.ndarray], reference: dict[str, np.ndarray], scale: np.ndarray) -> np.ndarray:
    """Return the monotone DTW reference index for every query index."""

    n = len(query["position"])
    m = len(reference["position"])
    cost = np.full((n, m), np.inf, dtype=np.float64)
    parent = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            cost[i, j] = float(np.linalg.norm(_pose_residual(query, reference, i, j)[:7] / scale[:7]))
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[float, int]] = []
            if i:
                candidates.append((cost[i - 1, j], 0))
            if j:
                candidates.append((cost[i, j - 1], 1))
            if i and j:
                candidates.append((cost[i - 1, j - 1], 2))
            value, step = min(candidates, key=lambda item: item[0])
            cost[i, j] += value
            parent[i, j] = step
    i, j = n - 1, m - 1
    path: list[tuple[int, int]] = []
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        step = int(parent[i, j])
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        elif step == 2:
            i -= 1
            j -= 1
        else:
            raise RuntimeError(f"invalid DTW parent at {(i, j)}: {step}")
    path.reverse()
    aligned = np.zeros(n, dtype=np.int32)
    for i in range(n):
        aligned[i] = max(j for ii, j in path if ii == i)
    return aligned


def _estimate_phase_band(
    experts: dict[int, dict[str, Any]],
    pairs: dict[int, int],
    feature_scale: np.ndarray,
    quantile: float,
) -> tuple[float, float, list[float]]:
    offsets: list[float] = []
    for seed, peer_seed in pairs.items():
        query = _pose_series(experts[seed]["arrays"], 10.0)
        reference = _pose_series(experts[peer_seed]["arrays"], 10.0)
        mapping = _global_dtw_map(query, reference, feature_scale)
        n = len(query["position"])
        m = len(reference["position"])
        for i, j in enumerate(mapping):
            offsets.append(float(j / max(1, m - 1) - i / max(1, n - 1)))
    positive = [value for value in offsets if value > 0]
    negative = [-value for value in offsets if value < 0]
    ahead = float(np.quantile(positive, quantile)) if positive else 0.0
    behind = float(np.quantile(negative, quantile)) if negative else 0.0
    return ahead, behind, offsets


def _phase_band_align(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    feature_scale: np.ndarray,
    ahead: float,
    behind: float,
    max_jump: int,
) -> np.ndarray:
    """Monotone local alignment in a normalized two-sided phase band."""

    n = len(query["position"])
    m = len(reference["position"])
    mapping = np.zeros(n, dtype=np.int32)
    previous = 0
    for i in range(n):
        progress = i / max(1, n - 1)
        lower = max(0.0, progress - behind)
        upper = min(1.0, progress + ahead)
        lo = max(previous, int(np.floor(lower * max(1, m - 1))))
        hi = min(m - 1, int(np.ceil(upper * max(1, m - 1))), previous + max_jump)
        if lo > hi:
            lo = previous
            hi = max(previous, min(m - 1, int(np.ceil(upper * max(1, m - 1)))))
        previous = min(
            range(lo, hi + 1),
            key=lambda j: float(np.linalg.norm(_pose_residual(query, reference, i, j)[:7] / feature_scale[:7])),
        )
        mapping[i] = previous
    return mapping


def _robust_scale(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - median), axis=0)
    return np.maximum(scale, np.asarray([1e-3, 1e-3, 1e-3], dtype=np.float64))


def _timeline(
    query: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    mapping: np.ndarray,
    path_scale: np.ndarray,
    horizon: int,
) -> list[float | None]:
    values: list[float | None] = []
    for i in range(horizon + 1):
        if i >= len(query["position"]):
            values.append(None)
            continue
        residual = _pose_residual(query, reference, i, int(mapping[i]))[:3]
        values.append(float(np.linalg.norm(residual / path_scale)))
    return values


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    existing_id = json.loads(args.id_analysis.read_text(encoding="utf-8"))
    existing_ood = json.loads(args.ood_analysis.read_text(encoding="utf-8"))
    id_meta = _load_jsonl(args.id_metadata)
    ood_meta = _load_jsonl(args.ood_metadata)
    experts = _load_pose_root(args.expert_root)
    id_policy = _load_pose_root(args.id_root)
    ood_policy = _load_pose_root(args.ood_root)
    expert_pose = {seed: _pose_series(item["arrays"], args.fps) for seed, item in experts.items()}
    id_pose = {seed: _pose_series(item["arrays"], args.fps) for seed, item in id_policy.items()}
    ood_pose = {seed: _pose_series(item["arrays"], args.fps) for seed, item in ood_policy.items()}
    feature_scale = np.asarray(existing_id["feature_scale"], dtype=np.float64)
    pairs = {int(row["seed"]): int(row["peer_seed"]) for row in existing_id["expert_context_pair_rows"]}
    ahead, behind, offsets = _estimate_phase_band(experts, pairs, feature_scale, args.phase_quantile)

    expert_residuals: list[np.ndarray] = []
    expert_maps: dict[int, np.ndarray] = {}
    for seed, peer_seed in pairs.items():
        mapping = _phase_band_align(expert_pose[seed], expert_pose[peer_seed], feature_scale, ahead, behind, args.max_jump)
        expert_maps[seed] = mapping
        for i, j in enumerate(mapping):
            expert_residuals.append(_pose_residual(expert_pose[seed], expert_pose[peer_seed], i, int(j))[:3])
    path_scale = _robust_scale(np.asarray(expert_residuals, dtype=np.float64))
    expert_scores = np.asarray(
        [np.linalg.norm(residual / path_scale) for residual in expert_residuals],
        dtype=np.float64,
    )

    def build_rows(
        analysis: dict[str, Any],
        metadata: dict[int, dict[str, Any]],
        poses: dict[int, dict[str, np.ndarray]],
        group: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in analysis["rows"]:
            source = metadata[int(row["episode_index"])]
            item = poses[int(row["seed"])]
            reference = expert_pose[int(row["nearest_expert_seed"])]
            mapping = _phase_band_align(item, reference, feature_scale, ahead, behind, args.max_jump)
            values = _timeline(item, reference, mapping, path_scale, args.horizon)
            rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "seed": int(row["seed"]),
                    "success": bool(source.get("success", False)),
                    "steps": int(source.get("steps", max(0, len(item["position"]) - 1))),
                    "group": group,
                    "nearest_expert_seed": int(row["nearest_expert_seed"]),
                    "distance_timeline": values,
                }
            )
        return rows

    id_success = build_rows(existing_id, id_meta, id_pose, "id_success")
    id_failure = build_rows(existing_id, id_meta, id_pose, "id_failure")
    id_success = [row for row in id_success if row["success"]]
    id_failure = [row for row in id_failure if not row["success"]]
    ood_rows = build_rows(existing_ood, ood_meta, ood_pose, "ood")

    def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        matrix = np.asarray(
            [[np.nan if value is None else value for value in row["distance_timeline"]] for row in rows],
            dtype=np.float64,
        )
        time_distribution: list[dict[str, Any]] = []
        for index, step in enumerate(range(args.horizon + 1)):
            finite = matrix[:, index][np.isfinite(matrix[:, index])] if len(matrix) else np.asarray([])
            time_distribution.append(
                {
                    "step": step,
                    "n": int(len(finite)),
                    "p25": float(np.quantile(finite, 0.25)) if len(finite) else None,
                    "median": float(np.median(finite)) if len(finite) else None,
                    "p75": float(np.quantile(finite, 0.75)) if len(finite) else None,
                    "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
                }
            )
        window = np.nanmedian(matrix[:, : args.common_window + 1], axis=1) if len(matrix) else np.asarray([])
        return {
            "episodes": len(rows),
            "time_distribution": time_distribution,
            "common_window_steps": args.common_window,
            "episode_median_quantiles": {
                "p25": float(np.quantile(window, 0.25)) if len(window) else None,
                "median": float(np.median(window)) if len(window) else None,
                "p75": float(np.quantile(window, 0.75)) if len(window) else None,
                "p95": float(np.quantile(window, 0.95)) if len(window) else None,
            },
            "rows": rows,
        }

    return {
        "format": "ycb_dpath_analysis_v1",
        "diagnostic_only": True,
        "horizon": args.horizon,
        "fps": args.fps,
        "phase_quantile": args.phase_quantile,
        "phase_band": {"reference_ahead": ahead, "reference_behind": behind},
        "phase_offset_quantiles": np.quantile(offsets, [0.025, 0.5, 0.975]).tolist(),
        "max_jump": args.max_jump,
        "expert_episodes": len(experts),
        "expert_calibration_count": len(expert_scores),
        "path_scale": path_scale.tolist(),
        "expert_path_quantiles": {
            "p50": float(np.quantile(expert_scores, 0.50)),
            "p75": float(np.quantile(expert_scores, 0.75)),
            "p95": float(np.quantile(expert_scores, 0.95)),
        },
        "groups": {
            "id_success": group_stats(id_success),
            "id_failure": group_stats(id_failure),
            "ood": group_stats(ood_rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--id-root", type=Path, required=True)
    parser.add_argument("--ood-root", type=Path, required=True)
    parser.add_argument("--id-analysis", type=Path, required=True)
    parser.add_argument("--ood-analysis", type=Path, required=True)
    parser.add_argument("--id-metadata", type=Path, required=True)
    parser.add_argument("--ood-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--phase-quantile", type=float, default=0.975)
    parser.add_argument("--max-jump", type=int, default=5)
    parser.add_argument("--common-window", type=int, default=40)
    args = parser.parse_args()
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "format": result["format"], "groups": {key: value["episodes"] for key, value in result["groups"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
