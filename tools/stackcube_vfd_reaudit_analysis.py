#!/usr/bin/env python3
"""Analyze and plot a reproducible StackCube VFD ID/OOD audit.

The constant threshold follows FIPER's failure-detection convention: take
the requested quantile over the *maximum* VFD score of each successful ID
calibration rollout. It is intentionally different from pooling every
decision point into one global sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


METRICS = ("oneway_vfd", "twoway_vfd")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def traces(
    episodes: list[dict[str, Any]], metric: str, *, successful_only: bool = False
) -> list[tuple[int, list[float]]]:
    if metric not in METRICS:
        raise ValueError(f"unknown VFD metric: {metric}")
    output: list[tuple[int, list[float]]] = []
    for episode in episodes:
        if successful_only and not episode.get("success", False):
            continue
        trace = [
            float(chunk[metric])
            for chunk in episode.get("timeline", [])
            if metric in chunk
        ]
        if trace:
            output.append((int(episode["seed"]), trace))
    return output


def fiper_constant_threshold(
    successful_id_traces: list[tuple[int, list[float]]], quantile: float
) -> float:
    """FIPER's constant threshold: q-quantile of trajectory maxima."""
    if not successful_id_traces:
        raise ValueError("need at least one successful ID trace")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between zero and one")
    maxima = [max(trace) for _, trace in successful_id_traces]
    return float(np.quantile(maxima, quantile))


def _draw_panel(
    ax: Any,
    *,
    id_traces: list[tuple[int, list[float]]],
    ood_traces: list[tuple[int, list[float]]],
    threshold: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    for _, trace in id_traces:
        ax.plot(
            np.arange(len(trace)),
            trace,
            color="#94a3b8",
            alpha=0.28,
            linewidth=0.9,
        )
    palette = plt.get_cmap("tab10")
    for index, (seed, trace) in enumerate(ood_traces):
        ax.plot(
            np.arange(len(trace)),
            trace,
            marker="o",
            markersize=3.6,
            linewidth=1.8,
            color=palette(index % 10),
            label=f"OOD {seed}",
        )
    ax.axhline(
        threshold,
        color="#c2410c",
        linestyle="--",
        linewidth=1.5,
        label=f"FIPER q=.95 = {threshold:.3f}",
    )
    ax.set_title(title)
    ax.set_xlabel("10-step decision chunk")
    ax.set_ylabel("VFD uncertainty")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)


def plot_audit(
    id_episodes: list[dict[str, Any]],
    ood_episodes: list[dict[str, Any]],
    *,
    quantile: float,
    output: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    metrics: dict[str, dict[str, Any]] = {}
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=False)
    labels = (
        "One-way VFD: member0 to member1",
        "Two-way VFD: mean(A to B, B to A)",
    )
    for axis, metric, title in zip(axes, METRICS, labels, strict=True):
        id_success = traces(id_episodes, metric, successful_only=True)
        ood = traces(ood_episodes, metric)
        threshold = fiper_constant_threshold(id_success, quantile)
        _draw_panel(
            axis,
            id_traces=id_success,
            ood_traces=ood,
            threshold=threshold,
            title=title,
        )
        metrics[metric] = {
            "threshold": threshold,
            "id_successful_episodes": len(id_success),
            "id_chunks": sum(len(trace) for _, trace in id_success),
            "ood_episodes": len(ood),
            "ood_chunks": sum(len(trace) for _, trace in ood),
            "id_trajectory_maxima": {
                str(seed): max(trace) for seed, trace in id_success
            },
            "ood_trajectory_maxima": {
                str(seed): max(trace) for seed, trace in ood
            },
        }
    figure.suptitle(
        "StackCube step-7000 VFD re-audit: identical C=64 ID/OOD protocol",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    return {"quantile": quantile, "metrics": metrics, "figure": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-episodes", type=Path, required=True)
    parser.add_argument("--ood-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    report = plot_audit(
        read_json(args.id_episodes),
        read_json(args.ood_episodes),
        quantile=args.quantile,
        output=args.output,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
