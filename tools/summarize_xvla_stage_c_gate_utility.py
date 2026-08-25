#!/usr/bin/env python3
"""Summarize same-budget utility of frozen gate-selected datasets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def success_rate(summary: dict[str, Any], task: str) -> float:
    if task == "airplane":
        successes = summary.get("ever_grasped_successes")
        if successes is None:
            successes = summary.get("successes")
    else:
        successes = summary.get("successes")
    episodes = int(summary.get("episodes", -1))
    if successes is None or episodes <= 0:
        raise ValueError(f"summary lacks endpoint count: {summary}")
    return float(successes) / episodes


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("empty metric list")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def summarize(
    *,
    evaluation_root: Path,
    dataset_root: Path,
    task: str,
    methods: list[str],
    seeds: list[int],
    budget: int,
    split: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        dataset = dataset_root / task / method / "selected"
        info_path = dataset / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        info = read_json(info_path)
        if int(info.get("total_frames", -1)) != budget:
            raise RuntimeError(f"{dataset} has total_frames={info.get('total_frames')}, expected {budget}")
        values: list[float] = []
        for seed in seeds:
            summary_path = evaluation_root / task / method / f"seed_{seed}" / split / "summary.json"
            summary = read_json(summary_path)
            value = success_rate(summary, task)
            values.append(value)
            rows.append(
                {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "split": split,
                    "episodes": int(summary["episodes"]),
                    "success_rate": value,
                    "budget": budget,
                    "dataset": str(dataset.resolve()),
                }
            )
        mean, std = mean_std(values)
        rows.append(
            {
                "task": task,
                "method": method,
                "seed": "mean",
                "split": split,
                "episodes": len(values),
                "success_rate": mean,
                "seed_std": std,
                "budget": budget,
                "dataset": str(dataset.resolve()),
            }
        )
    means = [row for row in rows if row["seed"] == "mean"]
    best = max(means, key=lambda row: row["success_rate"])
    return {
        "format": "xvla_stage_c_gate_utility_v1",
        "task": task,
        "split": split,
        "methods": methods,
        "training_seeds": seeds,
        "budget": budget,
        "best_method": best["method"],
        "best_success_rate": best["success_rate"],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--split", choices=("id", "ood"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        evaluation_root=args.evaluation_root,
        dataset_root=args.dataset_root,
        task=args.task,
        methods=args.methods,
        seeds=args.seeds,
        budget=args.budget,
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
