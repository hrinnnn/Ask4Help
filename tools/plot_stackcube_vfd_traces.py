#!/usr/bin/env python3
"""Plot saved StackCube ID and OOD chunk-level VFD traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def scores(episode: dict[str, Any]) -> list[float]:
    return [float(chunk["vfd"]) for chunk in episode.get("timeline", [])]


def draw_id(ax: plt.Axes, episodes: list[dict[str, Any]], threshold: float) -> None:
    max_chunks = max(map(lambda episode: len(scores(episode)), episodes))
    aligned = np.full((len(episodes), max_chunks), np.nan)
    for row, episode in enumerate(episodes):
        trace = scores(episode)
        aligned[row, : len(trace)] = trace
        ax.plot(range(len(trace)), trace, color="#9aa5b1", alpha=0.45, linewidth=1)

    ax.plot(range(max_chunks), np.nanmedian(aligned, axis=0), color="#007f73", linewidth=2.5, label="ID median")
    ax.axhline(threshold, color="#c2410c", linestyle="--", linewidth=1.6, label=f"q=.95 threshold = {threshold:.2f}")
    ax.set_title(f"ID successful policy rollouts (n={len(episodes)})")
    ax.set_xlabel("10-step decision chunk")
    ax.set_ylabel("one-way VFD")
    ax.legend(loc="upper left", frameon=False)


def draw_ood(ax: plt.Axes, episodes: list[dict[str, Any]], threshold: float) -> None:
    palette = plt.get_cmap("tab10")
    for index, episode in enumerate(episodes):
        trace = scores(episode)
        x = np.arange(len(trace))
        color = palette(index % 10)
        ax.plot(x, trace, color=color, marker="o", linewidth=2, label=f"OOD seed {int(episode['seed'])}")
        for chunk_index, chunk in enumerate(episode.get("timeline", [])):
            if chunk.get("controller") == "expert":
                ax.scatter(chunk_index, float(chunk["vfd"]), marker="X", s=92, color="#dc2626", zorder=4)

    ax.axhline(threshold, color="#c2410c", linestyle="--", linewidth=1.6, label=f"q=.95 threshold = {threshold:.2f}")
    ax.scatter([], [], marker="X", s=92, color="#dc2626", label="expert takeover")
    ax.set_title(f"OOD 180-degree rollouts (n={len(episodes)})")
    ax.set_xlabel("10-step decision chunk")
    ax.set_ylabel("one-way VFD")
    ax.legend(loc="upper left", frameon=False, fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-traces", type=Path, required=True)
    parser.add_argument("--ood-episodes", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    id_episodes = [episode for episode in read_json(args.id_traces) if episode.get("success") and scores(episode)]
    ood_episodes = [episode for episode in read_json(args.ood_episodes) if scores(episode)]
    threshold = float(read_json(args.threshold_json)["threshold"])
    if not id_episodes or not ood_episodes:
        raise ValueError("both ID and OOD inputs need at least one VFD trace")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    draw_id(axes[0], id_episodes, threshold)
    draw_ood(axes[1], ood_episodes, threshold)
    figure.suptitle("StackCube step-7000: chunk-level one-way VFD", fontsize=15, fontweight="bold")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
