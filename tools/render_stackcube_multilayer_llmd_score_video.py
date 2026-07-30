#!/usr/bin/env python3
"""Render a StackCube multi-layer LLMD rollout beside synchronized scores.

This consumes a passive multi-layer evaluator result plus its optional raw
rollout video. It never alters either source artifact: the annotated MP4 is a
separate, portable diagnostic asset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


DISPLAY_NAMES = {
    "action_expert_block_04": "Action Expert 25% (block 04)",
    "action_expert_block_08": "Action Expert 50% (block 08)",
    "action_expert_block_13": "Action Expert 75% (block 13)",
    "vlm_block_08_mean": "VLM middle (block 08)",
    "vlm_bridge_final_mean": "VLM-to-Action bridge (final)",
}


def load_episode(path: Path, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one multi-layer episode and its calibrated layer thresholds."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "stackcube_multilayer_llmd_rollout_v1":
        raise ValueError(f"not a StackCube multi-layer LLMD rollout file: {path}")
    episode_index, episode = next(
        ((index, row) for index, row in enumerate(payload["episodes"]) if int(row["seed"]) == seed),
        (None, None),
    )
    if episode is None or episode_index is None:
        raise KeyError(f"seed {seed} is not present in {path}")
    thresholds = payload.get("thresholds")
    if not thresholds or not thresholds.get("layers"):
        raise ValueError("annotated multi-layer videos require calibrated thresholds")
    return {"episode_index": episode_index, **episode}, thresholds["layers"]


def trace_specs(episode: dict[str, Any], thresholds: dict[str, Any]) -> list[tuple[str, str, list[float], float]]:
    """Build ordered score traces and validate that each has a threshold."""
    if not episode.get("timeline"):
        raise ValueError("episode has no LLMD timeline")
    names = list(episode["timeline"][0]["scores"])
    missing = [name for name in names if name not in thresholds]
    if missing:
        raise ValueError(f"missing thresholds for layers: {missing}")
    return [
        (
            name,
            DISPLAY_NAMES.get(name, name),
            [float(point["scores"][name]) for point in episode["timeline"]],
            float(thresholds[name]["threshold"]),
        )
        for name in names
    ]


def _points(values: list[float], threshold: float, width: int, height: int) -> list[tuple[int, int]]:
    maximum = max(max(values, default=0.0), threshold, 1e-6)
    denominator = max(len(values) - 1, 1)
    return [
        (
            int(index * (width - 1) / denominator),
            int((1.0 - value / (maximum * 1.1)) * (height - 1)),
        )
        for index, value in enumerate(values)
    ]


def _draw_trace(
    canvas: np.ndarray,
    *,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    values: list[float],
    threshold: float,
    current: int,
    alarm: bool,
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    maximum = max(max(values, default=0.0), threshold, 1e-6)
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (64, 77, 96), 1)
    threshold_y = y0 + int((1.0 - threshold / (maximum * 1.1)) * (height - 1))
    cv2.line(canvas, (x0, threshold_y), (x0 + width, threshold_y), (251, 113, 133), 1, cv2.LINE_AA)
    points = [(x0 + x, y0 + y) for x, y in _points(values, threshold, width, height)]
    if len(points) > 1:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (45, 212, 191), 2, cv2.LINE_AA)
    cursor = min(max(current, 0), len(points) - 1)
    if points:
        cv2.line(canvas, (points[cursor][0], y0), (points[cursor][0], y0 + height), (250, 204, 21), 1)
        cv2.circle(canvas, points[cursor], 3, (250, 204, 21), -1, cv2.LINE_AA)
    status = " ALARM" if alarm else ""
    color = (251, 113, 133) if alarm else (226, 232, 240)
    cv2.putText(canvas, f"{title}{status}", (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
    cv2.putText(
        canvas,
        f"score {values[cursor]:.1f} | threshold {threshold:.1f}",
        (x0 + width - 245, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (203, 213, 225),
        1,
    )


def render(*, episodes: Path, video_dir: Path, seed: int, output: Path, fps: int) -> None:
    import cv2
    import imageio.v2 as imageio

    episode, thresholds = load_episode(episodes, seed)
    source = video_dir / f"episode_{episode['episode_index']:06d}_seed_{seed:06d}.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    frames = imageio.mimread(source)
    if not frames:
        raise RuntimeError(f"video has no frames: {source}")
    traces = trace_specs(episode, thresholds)
    height, width = frames[0].shape[:2]
    panel_height = 34 + 68 * len(traces)
    starts = [int(point["env_step"]) for point in episode["timeline"]]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stackcube-multilayer-llmd-video-") as temporary_dir:
        local_output = Path(temporary_dir) / output.name
        writer = imageio.get_writer(local_output, fps=fps, codec="libx264", quality=8)
        try:
            for frame_index, frame in enumerate(frames):
                step = min(frame_index, int(episode["steps"]))
                decision = max((index for index, start in enumerate(starts) if start <= step), default=0)
                timeline_point = episode["timeline"][decision]
                canvas = np.zeros((height + panel_height, width, 3), dtype=np.uint8)
                canvas[:height] = np.asarray(frame)[..., :3]
                canvas[height:] = (20, 27, 38)
                any_alarm = any(timeline_point["alarms"].values())
                status = "ALARM" if any_alarm else "normal"
                color = (251, 113, 133) if any_alarm else (45, 212, 191)
                cv2.rectangle(canvas, (10, 10), (255, 47), (15, 23, 33), -1)
                cv2.putText(canvas, f"seed {seed} | step {step} | {status}", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                cv2.putText(
                    canvas,
                    f"outcome: {'success' if episode['success'] else 'failure'}",
                    (width - 205, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (235, 235, 235),
                    1,
                )
                for index, (name, title, values, threshold) in enumerate(traces):
                    _draw_trace(
                        canvas,
                        origin=(18, height + 25 + 68 * index),
                        size=(width - 36, 54),
                        title=title,
                        values=values,
                        threshold=threshold,
                        current=decision,
                        alarm=bool(timeline_point["alarms"][name]),
                    )
                writer.append_data(canvas)
        finally:
            writer.close()
        if not local_output.is_file() or local_output.stat().st_size == 0:
            raise RuntimeError(f"local video encoding did not create {local_output}")
        shutil.copy2(local_output, output)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    render(episodes=args.episodes, video_dir=args.video_dir, seed=args.seed, output=args.output, fps=args.fps)


if __name__ == "__main__":
    main()
