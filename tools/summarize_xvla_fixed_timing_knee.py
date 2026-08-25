#!/usr/bin/env python3
"""Summarize fixed-step expert forks into a task-policy knee calibration.

The input root contains one directory per fixed timing anchor, each with an
episodes.jsonl and raw_archive/task_states/*.npy written by the fixed_timing
collector.  The step-0 anchor is the nominal expert reference for the same
seed; no policy update or downstream result is used here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dtw_average(left: np.ndarray, right: np.ndarray) -> float:
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = (
                (cost[i - 1, j], steps[i - 1, j]),
                (cost[i, j - 1], steps[i, j - 1]),
                (cost[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            previous_cost, previous_steps = min(candidates, key=lambda item: item[0])
            cost[i, j] = previous_cost + float(np.linalg.norm(left[i - 1] - right[j - 1]))
            steps[i, j] = previous_steps + 1
    return float(cost[n, m] / max(1, steps[n, m]))


def aligned_distance(expert: np.ndarray, nominal: np.ndarray, scale: np.ndarray) -> float:
    expert_z = expert.astype(np.float64) / scale
    nominal_z = nominal.astype(np.float64) / scale
    start = int(np.argmin(np.linalg.norm(nominal_z - expert_z[0], axis=1)))
    return _dtw_average(expert_z, nominal_z[start:])


def _load_anchor(root: Path, step: int, *, endpoint: str) -> dict[int, dict[str, Any]]:
    anchor = root / f"step_{step}"
    rows = _read_jsonl(anchor / "episodes.jsonl")
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        seed = int(row["seed"])
        state_path = Path(row["task_states"])
        if not state_path.is_absolute():
            state_path = anchor / state_path
        if not state_path.is_file():
            continue
        success = row.get(endpoint)
        if success is None and endpoint == "success":
            success = row.get("strict_success", False)
        if not bool(success) or int(row.get("expert_action_steps", 0)) <= 0:
            continue
        output[seed] = {
            "seed": seed,
            "states": np.load(state_path),
            "start": int(row["expert_start_step"]),
            "expert_actions": int(row["expert_action_steps"]),
            "steps": int(row["steps"]),
        }
    return output


def _knee_index(c: np.ndarray, d: np.ndarray) -> int:
    c_span = float(np.ptp(c))
    d_span = float(np.ptp(d))
    c_norm = (c - c.min()) / c_span if c_span > 1e-12 else np.zeros_like(c)
    d_norm = (d - d.min()) / d_span if d_span > 1e-12 else np.zeros_like(d)
    points = np.stack([c_norm, d_norm], axis=1)
    start, end = points[0], points[-1]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return int(np.argmin(c_norm + d_norm))
    distance = np.abs(
        direction[0] * (start[1] - points[:, 1])
        - (start[0] - points[:, 0]) * direction[1]
    ) / length
    return int(np.argmax(distance))


def summarize(
    root: Path,
    anchors: list[int],
    *,
    endpoint: str = "success",
    bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    pools = {step: _load_anchor(root, step, endpoint=endpoint) for step in anchors}
    common = sorted(set.intersection(*(set(pool) for pool in pools.values())))
    if not common:
        raise RuntimeError("no common successful seeds across fixed timing anchors")
    nominal = pools[anchors[0]]
    all_nominal = np.concatenate([nominal[s]["states"] for s in common], axis=0)
    scale = np.std(all_nominal, axis=0)
    scale[scale < 1e-6] = 1.0

    per_anchor: dict[str, list[dict[str, Any]]] = {}
    means_c, means_d = [], []
    for step in anchors:
        rows = []
        for s in common:
            ref = nominal[s]
            query = pools[step][s]
            distance = aligned_distance(
                query["states"][query["start"] :],
                ref["states"],
                scale,
            )
            rows.append(
                {
                    "seed": s,
                    "expert_actions": query["expert_actions"],
                    "nominal_actions": ref["expert_actions"],
                    "expert_cost": query["expert_actions"] / max(1, ref["expert_actions"]),
                    "deviation": distance,
                }
            )
        per_anchor[str(step)] = rows
        means_c.append(float(np.mean([r["expert_cost"] for r in rows])))
        means_d.append(float(np.mean([r["deviation"] for r in rows])))

    c = np.asarray(means_c, dtype=np.float64)
    d = np.asarray(means_d, dtype=np.float64)
    knee = _knee_index(c, d)
    rng = np.random.default_rng(seed)
    selected = np.zeros(len(anchors), dtype=np.int64)
    for _ in range(bootstrap):
        sample = rng.integers(0, len(common), size=len(common))
        cb, db = [], []
        for step in anchors:
            rows = per_anchor[str(step)]
            cb.append(float(np.mean([rows[i]["expert_cost"] for i in sample])))
            db.append(float(np.mean([rows[i]["deviation"] for i in sample])))
        selected[_knee_index(np.asarray(cb), np.asarray(db))] += 1
    probability = selected / max(1, bootstrap)
    knee_set = [anchors[i] for i, p in enumerate(probability) if p >= 0.10]
    if not knee_set:
        knee_set = [anchors[knee]]

    return {
        "format": "xvla_fixed_timing_knee_calibration_v1",
        "anchors": anchors,
        "common_successful_seed_count": len(common),
        "common_seeds": common,
        "nominal_anchor": anchors[0],
        "endpoint": endpoint,
        "nominal_state_scale": scale.tolist(),
        "anchor_summary": [
            {
                "step": step,
                "episodes": len(per_anchor[str(step)]),
                "mean_expert_cost": float(c[i]),
                "mean_deviation": float(d[i]),
                "knee_selection_probability": float(probability[i]),
            }
            for i, step in enumerate(anchors)
        ],
        "knee_anchor": anchors[knee],
        "knee_confidence_set": knee_set,
        "bootstrap": {"samples": bootstrap, "seed": seed, "selection_rule": "p>=0.10"},
        "per_anchor": per_anchor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", choices=("success", "strict_success", "ever_grasped"), default="success")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = summarize(
        args.root,
        args.anchors,
        endpoint=args.endpoint,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
