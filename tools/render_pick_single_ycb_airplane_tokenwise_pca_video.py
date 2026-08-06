#!/usr/bin/env python3
"""Render representative airplane rollouts with raw PCA-score panels."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pick_single_ycb_airplane_tokenwise_pca import MAIN_METHODS


COLORS = ((72, 175, 255), (86, 212, 129), (80, 120, 255), (230, 184, 54))


def _first_alert(episode: dict[str, Any], method: str, threshold: float) -> dict[str, Any] | None:
    return next((point for point in episode["timeline"] if float(point["scores"][method]) >= threshold), None)


def _draw_panel(
    frame: np.ndarray,
    episode: dict[str, Any],
    thresholds: dict[str, float],
    *,
    focus_method: str,
    env_step: int,
) -> np.ndarray:
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
        thickness = 3 if method == focus_method else 1
        cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        label_y = 60 + 21 * list(MAIN_METHODS).index(method)
        cv2.putText(panel, method, (left + 8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if method in thresholds:
            y = bottom - round((bottom - top) * (thresholds[method] - lo) / (hi - lo))
            if top <= y <= bottom:
                cv2.line(panel, (left, y), (right, y), color, 1, cv2.LINE_AA)
    decision = max((index for index, point in enumerate(episode["timeline"]) if point["env_step"] <= env_step), default=0)
    current_x = left + round((right - left) * decision / max(1, len(episode["timeline"]) - 1))
    cv2.line(panel, (current_x, top), (current_x, bottom), (225, 225, 225), 1, cv2.LINE_AA)
    threshold = thresholds[focus_method]
    alert = _first_alert(episode, focus_method, threshold)
    if alert is not None:
        alert_index = episode["timeline"].index(alert)
        alert_x = left + round((right - left) * alert_index / max(1, len(episode["timeline"]) - 1))
        cv2.line(panel, (alert_x, top), (alert_x, bottom), (55, 55, 235), 2, cv2.LINE_AA)
    current = episode["timeline"][decision]
    current_score = float(current["scores"][focus_method])
    state = "ALERT" if current_score >= threshold else "policy"
    alert_text = "none" if alert is None else f"step {alert['env_step']}"
    cv2.putText(
        panel,
        f"{focus_method}: {current_score:.4g} / th={threshold:.4g} | {state} | first alert={alert_text}",
        (16, panel_h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (55, 55, 235) if state == "ALERT" else (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
    latest = episode["timeline"][-1].get("topk", {}).get("bridge", {}).get("source_fraction", {})
    modal = "TopK16 bridge: " + " ".join(f"{name}={latest.get(name, 0.0):.2f}" for name in ("base_camera", "wrist_camera", "language_state"))
    cv2.putText(panel, modal, (16, panel_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (215, 215, 215), 1, cv2.LINE_AA)
    return np.concatenate((frame, panel), axis=0)


def _transcode_h264(output: Path) -> None:
    import imageio_ffmpeg

    converted = output.with_name(f"{output.stem}.h264.mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(output), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    converted.replace(output)


def _render_one(
    episode: dict[str, Any], output: Path, thresholds: dict[str, float], focus_method: str, h264: bool
) -> None:
    capture = cv2.VideoCapture(str(episode["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source rollout video: {episode['video']}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    ok, first = capture.read()
    if not ok:
        raise RuntimeError(f"source rollout video is empty: {episode['video']}")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0] + 250))
    try:
        writer.write(_draw_panel(first, episode, thresholds, focus_method=focus_method, env_step=0))
        frame_index = 1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_draw_panel(frame, episode, thresholds, focus_method=focus_method, env_step=frame_index))
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"score-video rendering failed: {output}")
    if h264:
        _transcode_h264(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--focus-method", choices=MAIN_METHODS, default="vlm_input_pooled_pca")
    parser.add_argument("--h264", action="store_true", help="transcode outputs to browser-playable H.264/yuv420p")
    parser.add_argument(
        "--all-alerted-failures",
        action="store_true",
        help="render every failure alerted by focus-method, plus alert_times.json",
    )
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
    if args.all_alerted_failures:
        selected = [
            episode
            for episode in mapping.values()
            if not episode["ever_grasped"] and _first_alert(episode, args.focus_method, thresholds[args.focus_method]) is not None
        ]
        selected.sort(key=lambda episode: (episode["split"], _first_alert(episode, args.focus_method, thresholds[args.focus_method])["env_step"], episode["seed"]))
        index = []
        for episode in selected:
            alert = _first_alert(episode, args.focus_method, thresholds[args.focus_method])
            filename = f"{episode['split']}_failure_seed_{episode['seed']}_alert_{alert['env_step']:03d}.mp4"
            _render_one(episode, args.output_dir / filename, thresholds, args.focus_method, args.h264)
            index.append(
                {
                    "split": episode["split"],
                    "seed": episode["seed"],
                    "episode_index": episode["episode_index"],
                    "first_alert_env_step": alert["env_step"],
                    "first_alert_decision_index": alert["decision_index"],
                    "video": filename,
                }
            )
        (args.output_dir / "alert_times.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    else:
        for label, episode_index in scan["representative_episode_indices"].items():
            if episode_index is None:
                continue
            _render_one(mapping[int(episode_index)], args.output_dir / f"{label}.mp4", thresholds, args.focus_method, args.h264)
    print(args.output_dir)


if __name__ == "__main__":
    main()
