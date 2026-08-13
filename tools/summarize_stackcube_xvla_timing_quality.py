#!/usr/bin/env python3
"""Compute continuous DCA, EAS, and DCE for controlled takeover data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.run_xvla_stackcube_stage2_training import METHODS


STATE_SCALE = np.asarray(
    [0.10] * 15 + [0.08, 1.0, 1.0], dtype=np.float32
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_states(states: np.ndarray) -> np.ndarray:
    if states.ndim != 2 or states.shape[1] != STATE_SCALE.size:
        raise ValueError(f"unexpected task-state shape: {states.shape}")
    return states.astype(np.float64) / STATE_SCALE


def dtw_average(left: np.ndarray, right: np.ndarray) -> float:
    """Average local cost along the minimum-cost monotonic alignment path."""
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = ((cost[i - 1, j], steps[i - 1, j]),
                          (cost[i, j - 1], steps[i, j - 1]),
                          (cost[i - 1, j - 1], steps[i - 1, j - 1]))
            previous_cost, previous_steps = min(candidates, key=lambda item: item[0])
            cost[i, j] = previous_cost + np.linalg.norm(left[i - 1] - right[j - 1])
            steps[i, j] = previous_steps + 1
    return float(cost[n, m] / max(1, steps[n, m]))


def aligned_completion_distance(expert: np.ndarray, nominal: np.ndarray) -> float:
    start = int(np.argmin(np.linalg.norm(nominal - expert[0], axis=1)))
    return dtw_average(expert, nominal[start:])


def dce(alignment: float, saving: float) -> float:
    return float(2.0 * alignment * saving / (alignment + saving + 1e-12))


def pool_rows(root: Path, method: str, datasets_root: Path) -> tuple[list[dict], list[dict]]:
    collection = root / "collection_pools" / method
    training = read_jsonl(collection / "training_episodes.jsonl")
    selected = json.loads(
        (datasets_root / method / "selection_manifest.json").read_text(encoding="utf-8")
    )["selected_source_episode_indices"]
    chosen = [training[int(index)] for index in selected]
    attempts = {int(row["attempt_index"]): row for row in read_jsonl(collection / "episodes.jsonl")}
    return chosen, [attempts[int(row["raw_attempt_index"])] for row in chosen]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    nominal_training = read_jsonl(
        args.root / "collection_pools/immediate/training_episodes.jsonl"
    )
    nominal_attempts = {
        int(row["attempt_index"]): row
        for row in read_jsonl(args.root / "collection_pools/immediate/episodes.jsonl")
    }
    nominal_by_seed = {
        int(row["seed"]): nominal_attempts[int(row["raw_attempt_index"])]
        for row in nominal_training
    }

    rows = []
    raw_distances = []
    pending = []
    for method in METHODS:
        selected, attempts = pool_rows(args.root, method, args.datasets)
        for train_row, attempt in zip(selected, attempts):
            seed = int(attempt["seed"])
            nominal = nominal_by_seed[seed]
            expert_start = int(attempt["expert_start_step"])
            expert_states = normalized_states(np.load(attempt["task_states"]))[expert_start:]
            nominal_states = normalized_states(np.load(nominal["task_states"]))
            distance = aligned_completion_distance(expert_states, nominal_states)
            saving = max(
                0.0,
                1.0 - float(train_row["expert_action_steps"]) / float(nominal["steps"]),
            )
            raw_distances.append(distance)
            pending.append((method, seed, attempt, train_row, distance, saving))

    positive = np.asarray([value for value in raw_distances if value > 1e-12])
    sigma = float(np.median(positive)) if positive.size else 1.0
    for method, seed, attempt, train_row, distance, saving in pending:
        alignment = float(np.exp(-distance / max(sigma, 1e-12)))
        rows.append({
            "method": method,
            "seed": seed,
            "expert_start_step": int(attempt["expert_start_step"]),
            "expert_action_steps": int(train_row["expert_action_steps"]),
            "nominal_action_steps": int(nominal_by_seed[seed]["steps"]),
            "dtw_distance": distance,
            "dca": alignment,
            "eas": saving,
            "dce": dce(alignment, saving),
        })

    by_method = []
    for method in METHODS:
        current = [row for row in rows if row["method"] == method]
        by_method.append({
            "method": method,
            "episodes": len(current),
            "dca": float(np.mean([row["dca"] for row in current])),
            "eas": float(np.mean([row["eas"] for row in current])),
            "dce": float(np.mean([row["dce"] for row in current])),
            "expert_actions": sum(int(row["expert_action_steps"]) for row in current),
        })
    payload = {
        "format": "xvla_stackcube_timing_quality_v1",
        "state": "cube_xyz,target_xyz,tcp_xyz,tcp_minus_cube,cube_minus_target,gripper_width,grasped,on_cube",
        "state_scale": STATE_SCALE.tolist(),
        "dca_scale": sigma,
        "dca_scale_rule": "median positive DTW distance across selected timing episodes",
        "by_method": by_method,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
