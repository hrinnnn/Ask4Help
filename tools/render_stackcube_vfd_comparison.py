#!/usr/bin/env python3
"""Render an ID/OOD StackCube rollout video with chunk-level VFD overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FPS = 10
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
PLOT_HEIGHT = 170


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_by_seed(episodes: list[dict[str, Any]], seed: int | None, *, prefer_peak: bool) -> dict[str, Any]:
    if seed is not None:
        return next(item for item in episodes if int(item["seed"]) == seed)
    if prefer_peak:
        return max(episodes, key=lambda item: max((float(row["vfd"]) for row in item["timeline"]), default=-float("inf")))
    return next(item for item in episodes if item.get("success") and item.get("timeline"))


def _frame(capture: cv2.VideoCapture, index: int, previous: np.ndarray | None) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, value = capture.read()
    if not ok:
        if previous is None:
            return np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
        return previous
    return cv2.resize(value, (PANEL_WIDTH, PANEL_HEIGHT), interpolation=cv2.INTER_AREA)


def _draw_plot(canvas: np.ndarray, timeline: list[dict[str, Any]], frame_index: int, threshold: float | None, title: str) -> None:
    height, width = canvas.shape[:2]
    canvas[:] = (24, 24, 24)
    scores = [float(item["vfd"]) for item in timeline]
    starts = [int(item["frame_index"]) for item in timeline]
    current = max((idx for idx, start in enumerate(starts) if start <= frame_index), default=0)
    max_value = max(scores + ([threshold] if threshold is not None else []) + [1.0]) * 1.12
    left, right, top, bottom = 52, width - 18, 28, height - 35
    cv2.rectangle(canvas, (left, top), (right, bottom), (70, 70, 70), 1)
    if threshold is not None:
        y = int(bottom - (threshold / max_value) * (bottom - top))
        cv2.line(canvas, (left, y), (right, y), (65, 90, 230), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"threshold {threshold:.2f}", (left + 4, max(top + 14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (65, 90, 230), 1, cv2.LINE_AA)
    points: list[tuple[int, int]] = []
    for index, value in enumerate(scores):
        x = int(left + (index / max(1, len(scores) - 1)) * (right - left))
        y = int(bottom - (value / max_value) * (bottom - top))
        points.append((x, y))
    if len(points) > 1:
        cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (60, 210, 100), 2, cv2.LINE_AA)
    for index, point in enumerate(points):
        color = (70, 235, 130) if index <= current else (120, 120, 120)
        cv2.circle(canvas, point, 4 if index == current else 2, color, -1, cv2.LINE_AA)
    value = scores[current] if scores else 0.0
    cv2.putText(canvas, f"{title}  VFD={value:.3f}  chunk={current}", (left, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"0", (18, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{max_value:.1f}", (10, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-traces", type=Path, required=True)
    parser.add_argument("--ood-episodes", type=Path, required=True)
    parser.add_argument("--ood-video-dir", type=Path, required=True)
    parser.add_argument("--threshold-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-seed", type=int)
    parser.add_argument("--ood-seed", type=int)
    args = parser.parse_args()

    id_episode = _episode_by_seed(_read_json(args.id_traces), args.id_seed, prefer_peak=False)
    ood_episode = _episode_by_seed(_read_json(args.ood_episodes), args.ood_seed, prefer_peak=True)
    threshold = float(_read_json(args.threshold_json)["threshold"])
    id_video = Path(id_episode["video"])
    ood_video = args.ood_video_dir / f"episode_{_read_json(args.ood_episodes).index(ood_episode):06d}_seed_{int(ood_episode['seed']):06d}.mp4"
    if not id_video.is_file() or not ood_video.is_file():
        raise FileNotFoundError(f"missing input video: id={id_video} ood={ood_video}")

    id_capture, ood_capture = cv2.VideoCapture(str(id_video)), cv2.VideoCapture(str(ood_video))
    frames = max(int(id_capture.get(cv2.CAP_PROP_FRAME_COUNT)), int(ood_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (PANEL_WIDTH * 2, PANEL_HEIGHT + PLOT_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 output")
    last_id = last_ood = None
    try:
        for index in range(frames):
            id_image = _frame(id_capture, index, last_id)
            ood_image = _frame(ood_capture, index, last_ood)
            last_id, last_ood = id_image, ood_image
            cv2.putText(id_image, f"ID policy | seed {id_episode['seed']}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ood_image, f"OOD 180 deg | seed {ood_episode['seed']}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            top = np.hstack((id_image, ood_image))
            id_plot = np.zeros((PLOT_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
            ood_plot = np.zeros((PLOT_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
            _draw_plot(id_plot, id_episode["timeline"], index, threshold, "ID")
            _draw_plot(ood_plot, ood_episode["timeline"], index, threshold, "OOD")
            writer.write(np.vstack((top, np.hstack((id_plot, ood_plot)))))
    finally:
        writer.release()
        id_capture.release()
        ood_capture.release()
    print(json.dumps({"output": str(args.output), "id_seed": id_episode["seed"], "ood_seed": ood_episode["seed"], "threshold": threshold}, indent=2))


if __name__ == "__main__":
    main()
