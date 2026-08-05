#!/usr/bin/env python3
"""Render representative airplane rollouts with raw PCA-score panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pick_single_ycb_airplane_tokenwise_pca import MAIN_METHODS


COLORS = ((72, 175, 255), (86, 212, 129), (80, 120, 255), (230, 184, 54))


def _draw_panel(frame: np.ndarray, episode: dict[str, Any], thresholds: dict[str, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    panel_h = 250
    panel = np.full((panel_h, width, 3), 22, dtype=np.uint8)
    traces = {method: [float(point["scores"][method]) for point in episode["timeline"]] for method in MAIN_METHODS}
    all_values = [value for trace in traces.values() for value in trace]
    lo, hi = min(all_values), max(all_values)
    if hi <= lo:
        hi = lo + 1.0
    cv2.putText(panel, f"{episode['split']} seed={episode['seed']} grasp={int(episode['ever_grasped'])}", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
    left, right, top, bottom = 60, width - 20, 42, panel_h - 42
    cv2.line(panel, (left, bottom), (right, bottom), (115, 115, 115), 1)
    cv2.line(panel, (left, top), (left, bottom), (115, 115, 115), 1)
    for method, color in zip(MAIN_METHODS, COLORS):
        values = traces[method]
        points = []
        for index, value in enumerate(values):
            x = left + round((right - left) * index / max(1, len(values) - 1))
            y = bottom - round((bottom - top) * (value - lo) / (hi - lo))
            points.append((x, y))
        cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        label_y = 60 + 21 * list(MAIN_METHODS).index(method)
        cv2.putText(panel, method, (left + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if method in thresholds:
            y = bottom - round((bottom - top) * (thresholds[method] - lo) / (hi - lo))
            if top <= y <= bottom:
                cv2.line(panel, (left, y), (right, y), color, 1, cv2.LINE_AA)
    latest = episode["timeline"][-1].get("topk", {}).get("bridge", {}).get("source_fraction", {})
    modal = "TopK16 bridge: " + " ".join(f"{name}={latest.get(name, 0.0):.2f}" for name in ("base_camera", "wrist_camera", "language_state"))
    cv2.putText(panel, modal, (16, panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (215, 215, 215), 1, cv2.LINE_AA)
    return np.concatenate((frame, panel), axis=0)


def _render_one(episode: dict[str, Any], output: Path, thresholds: dict[str, float]) -> None:
    capture = cv2.VideoCapture(str(episode["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source rollout video: {episode['video']}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    ok, first = capture.read()
    if not ok:
        raise RuntimeError(f"source rollout video is empty: {episode['video']}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0] + 250))
    try:
        writer.write(_draw_panel(first, episode, thresholds))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_draw_panel(frame, episode, thresholds))
    finally:
        writer.release()
        capture.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"score-video rendering failed: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite rendered videos: {args.output_dir}")
    episodes = json.loads(args.episodes.read_text(encoding="utf-8"))
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    mapping = {int(row["episode_index"]): row for row in episodes["episodes"]}
    thresholds = {method: float(spec["best_balanced_accuracy"]["threshold"]) for method, spec in scan["methods"].items()}
    args.output_dir.mkdir(parents=True)
    for label, episode_index in scan["representative_episode_indices"].items():
        if episode_index is None:
            continue
        _render_one(mapping[int(episode_index)], args.output_dir / f"{label}.mp4", thresholds)
    print(args.output_dir)


if __name__ == "__main__":
    main()
