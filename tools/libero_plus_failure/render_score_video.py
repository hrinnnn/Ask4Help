#!/usr/bin/env python3
"""Reusable rollout video renderer with detector traces and conformal bands."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


DEFAULT_METHODS = ("bridge_llmd", "bridge_deep_knn", "bridge_pca_residual", "final_llmd", "acc", "vla_fail_final_or_acc")
DISPLAY = {
    "bridge_llmd": "Bridge LLMD",
    "bridge_deep_knn": "Bridge Deep kNN",
    "bridge_pca_residual": "Bridge PCA residual",
    "final_llmd": "Action Expert final LLMD",
    "acc": "ACC",
    "vla_fail_final_or_acc": "VLA-FAIL: final LLMD OR ACC",
}


def _find_episode(scored: list[dict[str, Any]], episode_path: Path) -> dict[str, Any]:
    resolved = str(episode_path.resolve())
    for row in scored:
        if str(Path(row["episode_path"]).resolve()) == resolved:
            return row
    raise KeyError("scored episode not found: " + resolved)


def _draw_trace(canvas: np.ndarray, origin: tuple[int, int], size: tuple[int, int], title: str,
                values: list[float], threshold: float, current: int, alarm_at: int | None) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    maximum = max(max(values, default=0.0), threshold, 1e-6)
    denominator = max(len(values) - 1, 1)
    points = [
        (x0 + int(index * (width - 1) / denominator), y0 + int((1.0 - value / (maximum * 1.1)) * (height - 1)))
        for index, value in enumerate(values)
    ]
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (88, 98, 112), 1)
    threshold_y = y0 + int((1.0 - threshold / (maximum * 1.1)) * (height - 1))
    cv2.line(canvas, (x0, threshold_y), (x0 + width, threshold_y), (246, 113, 137), 1, cv2.LINE_AA)
    if len(points) > 1:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (48, 210, 188), 2, cv2.LINE_AA)
    point_index = min(current, len(points) - 1)
    if points:
        cv2.circle(canvas, points[point_index], 3, (250, 204, 21), -1, cv2.LINE_AA)
    alarm = alarm_at is not None and current >= alarm_at
    cv2.putText(canvas, title + ("  ALERT" if alarm else ""), (x0 + 5, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (246, 113, 137) if alarm else (230, 236, 244), 1)
    cv2.putText(canvas, "score %.4g  |  threshold %.4g" % (values[point_index], threshold),
                (x0 + 5, y0 + height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (205, 214, 226), 1)


def render(scored_path: Path, thresholds_path: Path, episode_path: Path, output: Path, methods: tuple[str, ...]) -> None:
    payload = json.loads(scored_path.read_text(encoding="utf-8"))
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))["thresholds"]
    episode = _find_episode(payload, episode_path)
    frames = imageio.mimread(episode["video_path"])
    if not frames:
        raise RuntimeError("input video has no frames")
    height, width = frames[0].shape[:2]
    panel_height = 32 + 66 * len(methods)
    starts = list(episode["decision_steps"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="libero-plus-failure-video-") as temporary:
        local = Path(temporary) / output.name
        writer = imageio.get_writer(local, fps=10, codec="libx264", quality=8)
        try:
            for frame_index, frame in enumerate(frames):
                decision = max((i for i, step in enumerate(starts) if step <= frame_index + 10), default=0)
                canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
                canvas[:height] = np.asarray(frame)[..., :3]
                canvas[height:] = (18, 25, 36)
                for row, method in enumerate(methods):
                    values = list(episode["scores"][method])
                    if not values:
                        continue
                    threshold = 1.0 if method == "vla_fail_final_or_acc" else float(thresholds[method]["threshold"])
                    _draw_trace(canvas, (16, height + 25 + row * 66), (width - 32, 53), DISPLAY[method], values,
                                threshold, min(decision, len(values) - 1), episode["first_alert"].get(method))
                writer.append_data(canvas)
        finally:
            writer.close()
        if not local.is_file() or local.stat().st_size == 0:
            raise RuntimeError("video encoder produced no output")
        shutil.copy2(local, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-episodes", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    args = parser.parse_args()
    render(args.scored_episodes, args.thresholds, args.episode_dir, args.output, tuple(args.methods))


if __name__ == "__main__":
    main()
