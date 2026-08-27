#!/usr/bin/env python3
"""Compute ERD-Pose curves for an arbitrary task-specific pose root.

The existing X-VLA ERD analyzer is intentionally scoped to the original
StackCube/Grab Plane horizon.  This small diagnostic adapter reuses its pose
geometry/alignment primitives while allowing longer horizons and arbitrary
task labels (YCB object variation and OpenDrawer included).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_xvla_erd_pose import (
    _causal_align,
    _context,
    _context_distance,
    _context_scale,
    _load_pose_root,
    _pairwise_expert_residuals,
    _pose_residual,
    _pose_series,
    _robust_scales,
)


def _decision_steps(horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, horizon + stride, stride, dtype=np.int32)


def _expert_scores(experts: list[dict[str, Any]], contexts: np.ndarray, context_scale: np.ndarray, feature_scale: np.ndarray, fps: float, steps: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    all_values: list[float] = []
    matrices: list[np.ndarray] = []
    for index, item in enumerate(experts):
        distances = [
            _context_distance(contexts[index], contexts[j], context_scale) if index != j else float("inf")
            for j in range(len(experts))
        ]
        peer = experts[int(np.argmin(distances))]
        query = _pose_series(item["arrays"], fps)
        reference = _pose_series(peer["arrays"], fps)
        alignment = _causal_align(query, reference, feature_scale)
        values = np.full(len(steps), np.nan, dtype=np.float64)
        for step_index, step in enumerate(steps):
            if int(step) >= len(query["position"]):
                break
            values[step_index] = float(
                np.linalg.norm(
                    _pose_residual(
                        query,
                        reference,
                        int(step),
                        int(alignment[min(int(step), len(alignment) - 1)]),
                    )
                    / feature_scale
                )
            )
            all_values.append(float(values[step_index]))
        matrices.append(values)
    return np.asarray(all_values, dtype=np.float64), matrices


def _first_persistent(values: np.ndarray, steps: np.ndarray, threshold: float, persistence: int) -> int | None:
    for index in range(0, len(values) - persistence + 1):
        window = values[index : index + persistence]
        if np.all(np.isfinite(window)) and np.all(window > threshold):
            return int(steps[index])
    return None


def analyze_group(expert_root: Path, learner_root: Path, *, horizon: int, stride: int, fps: float, persistence: int, quantiles: tuple[float, ...]) -> dict[str, Any]:
    expert_map = _load_pose_root(expert_root)
    learner_map = _load_pose_root(learner_root)
    experts = [expert_map[key] for key in sorted(expert_map)]
    learners = [learner_map[key] for key in sorted(learner_map)]
    expert_contexts = np.asarray([_context(item) for item in experts])
    context_scale = _context_scale(expert_contexts)
    residuals, pair_rows = _pairwise_expert_residuals(experts, expert_contexts, context_scale, fps)
    feature_scale = _robust_scales(residuals)
    steps = _decision_steps(horizon, stride)
    calibration_values, expert_score_rows = _expert_scores(experts, expert_contexts, context_scale, feature_scale, fps, steps)
    thresholds = {str(q): float(np.quantile(calibration_values, q)) for q in quantiles}

    learner_contexts = np.asarray([_context(item) for item in learners])
    learner_rows: list[dict[str, Any]] = []
    for item in learners:
        context = _context(item)
        distances = [_context_distance(context, ref, context_scale) for ref in expert_contexts]
        nearest_index = int(np.argmin(distances))
        reference = _pose_series(experts[nearest_index]["arrays"], fps)
        query = _pose_series(item["arrays"], fps)
        alignment = _causal_align(query, reference, feature_scale)
        values = np.full(len(steps), np.nan, dtype=np.float64)
        for step_index, step in enumerate(steps):
            if int(step) >= len(query["position"]):
                break
            values[step_index] = float(
                np.linalg.norm(
                    _pose_residual(
                        query,
                        reference,
                        int(step),
                        int(alignment[min(int(step), len(alignment) - 1)]),
                    )
                    / feature_scale
                )
            )
        alarms = {
            str(q): _first_persistent(values, steps, threshold, persistence)
            for q, threshold in ((float(q), thresholds[str(q)]) for q in quantiles)
        }
        learner_rows.append(
            {
                "episode_index": int(item["episode_index"]),
                "seed": int(item["seed"]),
                "context_distance": float(distances[nearest_index]),
                "nearest_expert_seed": int(experts[nearest_index]["seed"]),
                "steps": steps.tolist(),
                "distance_timeline": [None if not np.isfinite(value) else float(value) for value in values],
                "alarms": alarms,
            }
        )

    def stats(q: float) -> dict[str, Any]:
        key = str(q)
        alarm_steps = [row["alarms"][key] for row in learner_rows if row["alarms"][key] is not None]
        return {
            "threshold": thresholds[key],
            "observed": len(alarm_steps),
            "episodes": len(learner_rows),
            "observed_rate": len(alarm_steps) / len(learner_rows) if learner_rows else None,
            "mean_alarm_step": float(np.mean(alarm_steps)) if alarm_steps else None,
            "median_alarm_step": float(np.median(alarm_steps)) if alarm_steps else None,
            "p25_alarm_step": float(np.quantile(alarm_steps, 0.25)) if alarm_steps else None,
            "p75_alarm_step": float(np.quantile(alarm_steps, 0.75)) if alarm_steps else None,
        }

    learner_matrix = np.asarray(
        [[np.nan if value is None else value for value in row["distance_timeline"]] for row in learner_rows],
        dtype=np.float64,
    )
    time_distribution = []
    for index, step in enumerate(steps):
        values = learner_matrix[:, index]
        finite = values[np.isfinite(values)]
        time_distribution.append(
            {
                "step": int(step),
                "p25": float(np.quantile(finite, 0.25)) if len(finite) else None,
                "median": float(np.median(finite)) if len(finite) else None,
                "p75": float(np.quantile(finite, 0.75)) if len(finite) else None,
            }
        )
    return {
        "format": "cross_task_erd_analysis_v1",
        "horizon": int(horizon),
        "decision_stride": int(stride),
        "fps": float(fps),
        "persistence_decisions": int(persistence),
        "expert_root": str(expert_root),
        "learner_root": str(learner_root),
        "expert_episodes": len(experts),
        "learner_episodes": len(learners),
        "feature_scale": feature_scale.tolist(),
        "context_scale": context_scale.tolist(),
        "expert_context_pair_rows": pair_rows,
        "expert_calibration_count": int(len(calibration_values)),
        "expert_calibration_quantiles": thresholds,
        "expert_score_rows": [[None if not np.isfinite(value) else float(value) for value in row] for row in expert_score_rows],
        "summary_by_quantile": {str(q): stats(float(q)) for q in quantiles},
        "time_distribution": time_distribution,
        "rows": learner_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--learner-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.925, 0.95])
    args = parser.parse_args()
    result = analyze_group(
        args.expert_root,
        args.learner_root,
        horizon=args.horizon,
        stride=args.stride,
        fps=args.fps,
        persistence=args.persistence,
        quantiles=tuple(args.quantiles),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "task": result["expert_root"], "episodes": result["learner_episodes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
