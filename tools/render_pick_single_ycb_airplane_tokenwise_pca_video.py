#!/usr/bin/env python3
"""Render representative airplane rollouts with raw PCA-score panels."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pick_single_ycb_airplane_tokenwise_pca import MAIN_METHODS


COLORS = ((72, 175, 255), (86, 212, 129), (80, 120, 255), (230, 184, 54))
ALERT_COLORS = ((55, 55, 235), (0, 165, 255), (180, 80, 190), (90, 210, 90))


def _panel_height(methods: tuple[str, ...]) -> int:
    return 58 + 92 * len(methods) + 34


def _first_alert(episode: dict[str, Any], method: str, threshold: float) -> dict[str, Any] | None:
    return next((point for point in episode["timeline"] if float(point["scores"][method]) >= threshold), None)


def _draw_panel(
    frame: np.ndarray,
    episode: dict[str, Any],
    thresholds: dict[str, float],
    *,
    focus_method: str,
    methods: tuple[str, ...],
    alert_methods: tuple[str, ...],
    env_step: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    panel_h = _panel_height(methods)
    panel = np.full((panel_h, width, 3), 22, dtype=np.uint8)
    traces = {method: [float(point["scores"][method]) for point in episode["timeline"]] for method in methods}
    cv2.putText(panel, f"{episode['split']} seed={episode['seed']} grasp={int(episode['ever_grasped'])}", (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
    left, right = 60, width - 20
    row_top = 38
    row_h = 76
    alert_positions: dict[str, int] = {}
    for method, color in zip(methods, COLORS):
        values = traces[method]
        index = methods.index(method)
        top = row_top + index * 92
        bottom = top + row_h
        threshold = float(thresholds[method])
        lo = min(min(values), threshold)
        hi = max(max(values), threshold)
        padding = max((hi - lo) * 0.10, 1e-8)
        lo, hi = lo - padding, hi + padding
        cv2.line(panel, (left, bottom), (right, bottom), (115, 115, 115), 1)
        cv2.line(panel, (left, top), (left, bottom), (115, 115, 115), 1)
        points = []
        for index, value in enumerate(values):
            x = left + round((right - left) * index / max(1, len(values) - 1))
            y = bottom - round((bottom - top) * (value - lo) / (hi - lo))
            points.append((x, y))
        thickness = 3 if method == focus_method else 1
        cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
        cv2.putText(panel, f"{method}: threshold={threshold:.4g}", (left + 8, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        threshold_y = bottom - round((bottom - top) * (threshold - lo) / (hi - lo))
        cv2.line(panel, (left, threshold_y), (right, threshold_y), color, 1, cv2.LINE_AA)
    decision = max((index for index, point in enumerate(episode["timeline"]) if point["env_step"] <= env_step), default=0)
    alerts = {method: _first_alert(episode, method, thresholds[method]) for method in alert_methods}
    for method, color in zip(alert_methods, ALERT_COLORS):
        alert = alerts[method]
        if alert is None:
            continue
        alert_index = episode["timeline"].index(alert)
        alert_x = left + round((right - left) * alert_index / max(1, len(episode["timeline"]) - 1))
        alert_positions[method] = alert_x
    for index, method in enumerate(methods):
        top = row_top + index * 92
        bottom = top + row_h
        current_x = left + round((right - left) * decision / max(1, len(episode["timeline"]) - 1))
        cv2.line(panel, (current_x, top), (current_x, bottom), (225, 225, 225), 1, cv2.LINE_AA)
        for alert_method, color in zip(alert_methods, ALERT_COLORS):
            if alert_method in alert_positions:
                cv2.line(panel, (alert_positions[alert_method], top), (alert_positions[alert_method], bottom), color, 2, cv2.LINE_AA)
    current = episode["timeline"][decision]
    current_score = float(current["scores"][focus_method])
    state = "ALERT" if current_score >= thresholds[focus_method] else "policy"
    alert_text = " | ".join(
        f"{method.replace('_pooled_pca', '').replace('_', ' ')}="
        f"{'none' if alert is None else alert['env_step']}"
        for method, alert in alerts.items()
    )
    cv2.putText(
        panel,
        f"{focus_method}: {current_score:.4g} / th={thresholds[focus_method]:.4g} | {state} | first: {alert_text}",
        (16, panel_h - 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (55, 55, 235) if state == "ALERT" else (215, 215, 215),
        1,
        cv2.LINE_AA,
    )
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
    episode: dict[str, Any], output: Path, thresholds: dict[str, float], focus_method: str,
    methods: tuple[str, ...], alert_methods: tuple[str, ...], h264: bool,
) -> None:
    capture = cv2.VideoCapture(str(episode["video"]))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source rollout video: {episode['video']}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
    ok, first = capture.read()
    if not ok:
        raise RuntimeError(f"source rollout video is empty: {episode['video']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="airplane-token-pca-video-") as temporary:
        local_output = Path(temporary) / output.name
        writer = cv2.VideoWriter(
            str(local_output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (first.shape[1], first.shape[0] + _panel_height(methods)),
        )
        try:
            writer.write(_draw_panel(first, episode, thresholds, focus_method=focus_method, methods=methods, alert_methods=alert_methods, env_step=0))
            frame_index = 1
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(_draw_panel(frame, episode, thresholds, focus_method=focus_method, methods=methods, alert_methods=alert_methods, env_step=frame_index))
                frame_index += 1
        finally:
            writer.release()
            capture.release()
        if not local_output.is_file() or local_output.stat().st_size == 0:
            raise RuntimeError(f"score-video rendering failed: {output}")
        if h264:
            _transcode_h264(local_output)
        shutil.copyfile(local_output, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--focus-method", default="vlm_input_pooled_pca")
    parser.add_argument("--alert-methods", nargs="+", default=None)
    parser.add_argument("--h264", action="store_true", help="transcode outputs to browser-playable H.264/yuv420p")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--all-alerted-failures",
        action="store_true",
        help="render every failure alerted by focus-method, plus alert_times.json",
    )
    selection.add_argument("--all-failures", action="store_true", help="render every failure, including detector misses")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite rendered videos: {args.output_dir}")
    episodes = json.loads(args.episodes.read_text(encoding="utf-8"))
    scan = json.loads(args.scan.read_text(encoding="utf-8"))
    mapping = {int(row["episode_index"]): row for row in episodes["episodes"]}
    thresholds = {method: float(spec["best_balanced_accuracy"]["threshold"]) for method, spec in scan["methods"].items()}
    methods = tuple(args.methods or MAIN_METHODS)
    if args.focus_method not in methods or not set(args.alert_methods or [args.focus_method]).issubset(methods):
        raise ValueError("focus and alert methods must be selected for rendering")
    if set(methods) - set(thresholds):
        raise ValueError("every rendered method needs a threshold in --scan")
    alert_methods = tuple(args.alert_methods or [args.focus_method])
    args.output_dir.mkdir(parents=True)
    if args.all_alerted_failures or args.all_failures:
        selected = [
            episode
            for episode in mapping.values()
            if not episode["ever_grasped"]
            and (args.all_failures or _first_alert(episode, args.focus_method, thresholds[args.focus_method]) is not None)
        ]
        selected.sort(key=lambda episode: (episode["split"], episode["seed"]))
        index = []
        for episode in selected:
            alerts = {method: _first_alert(episode, method, thresholds[method]) for method in alert_methods}
            label = "_".join(
                f"{method.replace('_pooled_pca', '').replace('_', '-')}-{('none' if alert is None else str(alert['env_step']).zfill(3))}"
                for method, alert in alerts.items()
            )
            filename = f"{episode['split']}_failure_seed_{episode['seed']}_{label}.mp4"
            _render_one(episode, args.output_dir / filename, thresholds, args.focus_method, methods, alert_methods, args.h264)
            index.append(
                {
                    "split": episode["split"],
                    "seed": episode["seed"],
                    "episode_index": episode["episode_index"],
                    "first_alerts": {
                        method: None if alert is None else {
                            "env_step": alert["env_step"], "decision_index": alert["decision_index"],
                        }
                        for method, alert in alerts.items()
                    },
                    "video": filename,
                }
            )
        (args.output_dir / "alert_times.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    else:
        for label, episode_index in scan["representative_episode_indices"].items():
            if episode_index is None:
                continue
            _render_one(
                mapping[int(episode_index)], args.output_dir / f"{label}.mp4", thresholds,
                args.focus_method, methods, alert_methods, args.h264,
            )
    print(args.output_dir)


if __name__ == "__main__":
    main()
