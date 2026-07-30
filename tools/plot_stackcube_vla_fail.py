#!/usr/bin/env python3
"""Render LLMD and ACC traces from StackCube VLA-FAIL rollout outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _episodes(path: Path) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "stackcube_vla_fail_rollout_v1":
        raise ValueError(f"not a StackCube VLA-FAIL result: {path}")
    return list(payload["episodes"]), payload.get("thresholds")


def _draw(axis: Any, episodes: list[dict[str, Any]], field: str, *, label: str) -> None:
    for episode in episodes:
        values = [chunk[field] for chunk in episode["timeline"] if chunk[field] is not None]
        if values:
            axis.plot(range(len(values)), values, marker="o", markersize=3, linewidth=1.3, alpha=0.8, label=f"{label} {episode['seed']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=Path, required=True, help="ID evaluation episodes.json")
    parser.add_argument("--ood", type=Path, required=True, help="OOD evaluation episodes.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    id_episodes, thresholds = _episodes(args.id)
    ood_episodes, ood_thresholds = _episodes(args.ood)
    thresholds = thresholds or ood_thresholds
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    specs = (("llmd", "LLMD", "llmd_threshold"), ("acc_ema", "ACC (EMA)", "acc_threshold"))
    for axis, (field, title, threshold_key) in zip(axes, specs, strict=True):
        _draw(axis, id_episodes, field, label="ID")
        _draw(axis, ood_episodes, field, label="OOD")
        if thresholds and threshold_key in thresholds:
            axis.axhline(float(thresholds[threshold_key]), color="#dc2626", linestyle="--", linewidth=1.5, label="conformal threshold")
        axis.set_title(title)
        axis.set_xlabel("receding-horizon decision")
        axis.set_ylabel("failure score")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    figure.suptitle("StackCube VLA-FAIL: fixed-prior LLMD and action-chunk consistency")
    figure.tight_layout(rect=(0, 0, 0.88, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
