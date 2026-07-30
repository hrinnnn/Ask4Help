#!/usr/bin/env python3
"""Render an inspectable StackCube rollout video with VLA-FAIL scores.

The source episode video is preserved untouched. This tool creates a second,
annotated diagnostic MP4: the original robot view occupies the top panel while
LLMD, ACC-EMA, their conformal bands, and the current alarm are synchronized
to the executed receding-horizon decision below it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _episode_by_seed(path: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "stackcube_vla_fail_rollout_v1":
        raise ValueError(f"not a StackCube VLA-FAIL rollout file: {path}")
    episode = next((row for row in payload["episodes"] if int(row["seed"]) == seed), None)
    if episode is None:
        raise KeyError(f"seed {seed} is not present in {path}")
    thresholds = payload.get("thresholds")
    if not thresholds:
        raise ValueError("annotated score videos require calibrated thresholds")
    return episode, thresholds


def _scale_points(values: list[float], *, threshold: float, width: int, height: int) -> list[tuple[int, int]]:
    maximum = max(max(values, default=0.0), threshold, 1e-6)
    count = max(len(values) - 1, 1)
    return [
        (
            int(index * (width - 1) / count),
            int((1.0 - value / (maximum * 1.1)) * (height - 1)),
        )
        for index, value in enumerate(values)
    ]


def _draw_trace(
    image: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    values: list[float | None],
    threshold: float,
    current: int,
    title: str,
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    clean = [float(value) if value is not None else 0.0 for value in values]
    cv2.rectangle(image, (x0, y0), (x0 + width, y0 + height), (52, 65, 85), 1)
    cv2.putText(image, title, (x0 + 6, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (222, 226, 230), 1)
    maximum = max(max(clean, default=0.0), float(threshold), 1e-6)
    threshold_y = y0 + int((1.0 - threshold / (maximum * 1.1)) * (height - 1))
    cv2.line(image, (x0, threshold_y), (x0 + width, threshold_y), (67, 99, 216), 1, cv2.LINE_AA)
    points = [(x0 + x, y0 + y) for x, y in _scale_points(clean, threshold=threshold, width=width, height=height)]
    if len(points) > 1:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (80, 210, 150), 2, cv2.LINE_AA)
    if points:
        cursor = min(max(current, 0), len(points) - 1)
        cv2.line(image, (points[cursor][0], y0), (points[cursor][0], y0 + height), (240, 180, 65), 1)
        cv2.circle(image, points[cursor], 4, (240, 180, 65), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"threshold {threshold:.2f}",
        (x0 + width - 150, y0 + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (160, 178, 241),
        1,
    )


def render(*, episodes_path: Path, video_dir: Path, seed: int, output: Path, fps: int) -> None:
    import cv2
    import imageio.v2 as imageio

    episode, thresholds = _episode_by_seed(episodes_path, seed)
    source = video_dir / f"episode_{episode['episode_index']:06d}_seed_{seed:06d}.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    frames = imageio.mimread(source)
    if not frames:
        raise RuntimeError(f"video has no frames: {source}")
    height, width = frames[0].shape[:2]
    panel_height = 190
    timeline = episode["timeline"]
    llmd = [float(row["llmd"]) for row in timeline]
    acc = [row["acc_ema"] for row in timeline]
    action_steps = [int(row["env_step"]) for row in timeline]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8)
    try:
        for frame_index, frame in enumerate(frames):
            step = min(frame_index, int(episode["steps"]))
            decision = max((index for index, start in enumerate(action_steps) if start <= step), default=0)
            canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
            canvas[:height] = np.asarray(frame)[..., :3]
            canvas[height:] = (22, 29, 40)
            alarm = bool(timeline[decision]["alarm"])
            status = "ALARM" if alarm else "normal"
            color = (67, 97, 216) if alarm else (80, 210, 150)
            cv2.rectangle(canvas, (10, 10), (245, 48), (18, 25, 35), -1)
            cv2.putText(
                canvas,
                f"seed {seed} | step {step} | {status}",
                (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"outcome: {'success' if episode['success'] else 'failure'}",
                (width - 190, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            margin = 18
            trace_width = width - 2 * margin
            trace_height = 73
            _draw_trace(
                canvas,
                origin=(margin, height + 22),
                size=(trace_width, trace_height),
                values=llmd,
                threshold=float(thresholds["llmd_threshold"]),
                current=decision,
                title="LLMD (final Action Expert feature)",
            )
            _draw_trace(
                canvas,
                origin=(margin, height + 108),
                size=(trace_width, trace_height),
                values=acc,
                threshold=float(thresholds["acc_threshold"]),
                current=decision,
                title="ACC-EMA (end-effector chunk consistency)",
            )
            writer.append_data(canvas)
    finally:
        writer.close()
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    render(
        episodes_path=args.episodes,
        video_dir=args.video_dir,
        seed=args.seed,
        output=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
