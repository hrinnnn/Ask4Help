#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def load_metric_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).expanduser().open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"metric log is empty: {path}")
    return records


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def metric_series(
    records: list[dict[str, Any]], key: str
) -> dict[int, list[float]]:
    series: dict[int, list[float]] = defaultdict(list)
    for record in records:
        if record.get("skipped", False):
            continue
        value = _finite_number(record.get(key))
        if value is None:
            continue
        series[int(record.get("env_idx", 0))].append(value)
    return dict(sorted(series.items()))


def moving_average(values: list[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("window must be positive")
    queue: deque[float] = deque()
    total = 0.0
    smoothed = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.popleft()
        smoothed.append(total / len(queue))
    return smoothed


def aggregate_by_position(
    series: dict[int, list[float]], window: int
) -> tuple[list[float], list[float]]:
    if not series:
        return [], []
    smoothed = {env: moving_average(values, window) for env, values in series.items()}
    max_length = max(len(values) for values in smoothed.values())
    means = []
    standard_deviations = []
    for position in range(max_length):
        values = [
            env_values[position]
            for env_values in smoothed.values()
            if position < len(env_values)
        ]
        means.append(statistics.fmean(values))
        standard_deviations.append(
            statistics.pstdev(values) if len(values) > 1 else 0.0
        )
    return means, standard_deviations


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    queried = [record for record in records if not record.get("skipped", False)]
    valid = [record for record in queried if record.get("valid", False)]
    phi_values = [
        value
        for record in queried
        if (value := _finite_number(record.get("phi_next"))) is not None
    ]
    rewards = [
        value
        for record in queried
        if (value := _finite_number(record.get("shaping_reward"))) is not None
    ]

    def stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "min": None, "max": None}
        return {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "records": len(records),
        "queried_records": len(queried),
        "valid_records": len(valid),
        "valid_rate": len(valid) / len(queried) if queried else 0.0,
        "environment_count": len({int(record.get("env_idx", 0)) for record in records}),
        "phi_next": stats(phi_values),
        "shaping_reward": stats(rewards),
    }


def _plot_panel(axis, series, *, window: int, label: str, color: str) -> None:
    for env_idx, values in series.items():
        axis.plot(
            range(1, len(values) + 1),
            moving_average(values, window),
            color=color,
            alpha=0.18,
            linewidth=0.8,
            label=f"env {env_idx}" if len(series) == 1 else None,
        )
    mean, std = aggregate_by_position(series, window)
    x_values = list(range(1, len(mean) + 1))
    axis.plot(x_values, mean, color=color, linewidth=2.0, label=f"mean {label}")
    if mean:
        axis.fill_between(
            x_values,
            [value - deviation for value, deviation in zip(mean, std)],
            [value + deviation for value, deviation in zip(mean, std)],
            color=color,
            alpha=0.16,
            label="mean +/- std",
        )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")


def plot_reward_curve(
    records: list[dict[str, Any]], output: str | Path, *, window: int
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi_series = metric_series(records, "phi_next")
    reward_series = metric_series(records, "shaping_reward")
    if not phi_series and not reward_series:
        raise ValueError("metric log contains neither phi_next nor shaping_reward")

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    _plot_panel(
        axes[0], phi_series, window=window, label="Phi", color="#176B87"
    )
    axes[0].set_ylabel("Fused potential Phi")
    axes[0].set_ylim(-0.05, 1.05)

    _plot_panel(
        axes[1], reward_series, window=window, label="shaping reward", color="#C84B31"
    )
    axes[1].axhline(0.0, color="#202020", linewidth=0.8)
    axes[1].set_ylabel("PBRS shaping reward")
    axes[1].set_xlabel("GRM query index per environment")
    figure.suptitle(f"Robo-Dopamine progress and reward (moving average={window})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summarize(records), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Phi and PBRS shaping reward from a Dopamine GRM JSONL log."
    )
    parser.add_argument("--input", required=True, help="Path to grm_metrics.jsonl")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--window", type=int, default=5, help="Moving-average window")
    args = parser.parse_args()

    records = load_metric_records(args.input)
    output = plot_reward_curve(records, args.output, window=args.window)
    print(json.dumps({"output": str(output), **summarize(records)}, indent=2))


if __name__ == "__main__":
    main()
