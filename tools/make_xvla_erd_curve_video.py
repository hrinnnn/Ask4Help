#!/usr/bin/env python3
"""Animate one expert residual curve against one OOD-fail ERD curve.

The expert curve is computed against its nearest-context expert peer using the
same robust scales and causal alignment as the diagnostic ERD analysis.  The
fail curve is read from the already audited OOD summary.  This is a visual
diagnostic and never modifies the formal X-VLA pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import analyze_xvla_erd_pose as erd


WIDTH, HEIGHT, FPS = 1280, 720, 10
PLOT_X0, PLOT_Y0, PLOT_W, PLOT_H = 78, 150, 1120, 470
BG = (18, 22, 30)
PANEL = (29, 35, 46)
GRID = (77, 88, 104)
TEXT = (235, 239, 245)
MUTED = (166, 177, 192)
EXPERT = (88, 196, 139)
FAIL = (242, 106, 93)
THRESHOLD = (248, 191, 72)
Q925 = (139, 149, 166)
Q975 = (180, 121, 219)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/SFNS.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_SMALL = _font(19)
FONT_LABEL = _font(24)
FONT_TITLE = _font(36)
FONT_CARD = _font(46)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, value, font=font, fill=fill)


def _dash(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], fill: tuple[int, int, int], width: int = 2, dash: int = 9) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        return
    for start in np.arange(0.0, length, dash * 2):
        end = min(start + dash, length)
        a, b = start / length, end / length
        draw.line(((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)), fill=fill, width=width)


def _expert_curve(expert_root: Path, selected_seed: int, horizon: int, fps: float) -> tuple[np.ndarray, int]:
    expert_map = erd._load_pose_root(expert_root)
    expert = [expert_map[key] for key in sorted(expert_map)]
    contexts = np.asarray([erd._context(item) for item in expert])
    context_scale = erd._context_scale(contexts)
    residuals, _ = erd._pairwise_expert_residuals(expert, contexts, context_scale, fps)
    feature_scale = erd._robust_scales(residuals)
    selected = next(item for item in expert if int(item["seed"]) == int(selected_seed))
    selected_context = erd._context(selected)
    distances = [
        erd._context_distance(selected_context, erd._context(item), context_scale)
        if int(item["seed"]) != int(selected_seed)
        else float("inf")
        for item in expert
    ]
    reference = expert[int(np.argmin(distances))]
    query = erd._pose_series(selected["arrays"], fps)
    reference_series = erd._pose_series(reference["arrays"], fps)
    match_indices = erd._causal_align(query, reference_series, feature_scale)
    steps = erd.DECISION_STEPS[erd.DECISION_STEPS <= horizon]
    values = np.full(len(steps), np.nan, dtype=np.float64)
    for index, step in enumerate(steps):
        if int(step) >= len(query["position"]):
            break
        values[index] = float(
            np.linalg.norm(
                erd._pose_residual(
                    query,
                    reference_series,
                    int(step),
                    int(match_indices[min(int(step), len(match_indices) - 1)]),
                )
                / feature_scale
            )
        )
    return values, len(query["position"])


def _fail_curve(summary: dict[str, Any], horizon: int) -> tuple[np.ndarray, int]:
    row = summary["rows"][0]
    values = np.asarray(row["distance_at_decision_steps"], dtype=np.float64)
    stride = int(summary.get("learner", {}).get("decision_stride", 5))
    steps = erd.DECISION_STEPS[erd.DECISION_STEPS <= horizon]
    return values[: len(steps)], int(row["erd_alarm_step"])


def _task_data(summary_path: Path, expert_root: Path, expert_seed: int, horizon: int = 50) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cal = np.asarray(summary["threshold"]["calibration_values"], dtype=np.float64)
    thresholds = {q: float(np.quantile(cal, q)) for q in (0.925, 0.95, 0.975)}
    expert_values, expert_length = _expert_curve(expert_root, expert_seed, horizon, 10.0)
    fail_values, fail_alarm = _fail_curve(summary, horizon)
    return {
        "task": summary.get("task", "task"),
        "expert_seed": expert_seed,
        "fail_seed": int(summary["rows"][0]["seed"]),
        "expert_values": expert_values,
        "expert_length": expert_length,
        "fail_values": fail_values,
        "fail_alarm": fail_alarm,
        "thresholds": thresholds,
        "horizon": horizon,
    }


def _plot_frame(data: dict[str, Any], frame: int, task_label: str) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    current_step = min(frame, data["horizon"])
    _text(draw, (42, 34), f"ERD-Pose score curves  |  {task_label}", FONT_TITLE)
    _text(draw, (42, 82), f"green expert demo (seed {data['expert_seed']})   red OOD fail (seed {data['fail_seed']})   |   q=.95 = {data['thresholds'][0.95]:.3f}", FONT_SMALL, MUTED)
    draw.rectangle((PLOT_X0, PLOT_Y0, PLOT_X0 + PLOT_W, PLOT_Y0 + PLOT_H), fill=PANEL, outline=GRID, width=1)

    values_all = np.concatenate((data["fail_values"], data["expert_values"][np.isfinite(data["expert_values"])]))
    y_max = max(float(np.max(values_all)) if len(values_all) else 10.0, data["thresholds"][0.975])
    y_max = max(10.0, np.ceil(y_max / 10.0) * 10.0)

    def x(step: float) -> float:
        return PLOT_X0 + float(step) / data["horizon"] * PLOT_W

    def y(value: float) -> float:
        return PLOT_Y0 + PLOT_H - float(value) / y_max * PLOT_H

    for tick in np.linspace(0, y_max, 5):
        py = y(float(tick))
        draw.line((PLOT_X0, py, PLOT_X0 + PLOT_W, py), fill=GRID, width=1)
        _text(draw, (PLOT_X0 - 50, int(py - 10)), f"{tick:g}", FONT_SMALL, MUTED)
    for tick in (0, 10, 20, 30, 40, 50):
        px = x(tick)
        draw.line((px, PLOT_Y0, px, PLOT_Y0 + PLOT_H), fill=GRID, width=1)
        _text(draw, (int(px - 10), PLOT_Y0 + PLOT_H + 10), str(tick), FONT_SMALL, MUTED)

    for q, color, label in ((0.925, Q925, "q.925"), (0.95, THRESHOLD, "q.95"), (0.975, Q975, "q.975")):
        py = y(data["thresholds"][q])
        _dash(draw, (PLOT_X0, py), (PLOT_X0 + PLOT_W, py), color, 2)
        _text(draw, (PLOT_X0 + PLOT_W - 74, int(py - 21)), label, FONT_SMALL, color)

    decision_steps = np.arange(0, data["horizon"] + 1, 5, dtype=np.int64)
    visible = decision_steps <= current_step
    fail_points = [(x(float(step)), y(float(value))) for step, value in zip(decision_steps[visible], data["fail_values"][visible])]
    _line(draw, fail_points, FAIL, 4)
    expert_valid = np.isfinite(data["expert_values"]) & visible
    expert_points = [(x(float(step)), y(float(value))) for step, value in zip(decision_steps[expert_valid], data["expert_values"][expert_valid])]
    _line(draw, expert_points, EXPERT, 4)
    # Hold the last expert score with a dashed line only after the recorded demo ends.
    if current_step > data["expert_length"] and expert_points:
        last_step, last_y = expert_points[-1]
        _dash(draw, (last_step, last_y), (x(current_step), last_y), EXPERT, 2)

    cursor = x(current_step)
    draw.line((cursor, PLOT_Y0, cursor, PLOT_Y0 + PLOT_H), fill=TEXT, width=1)
    index = min(int(round(current_step / 5)), len(data["fail_values"]) - 1)
    _text(draw, (int(cursor + 8), PLOT_Y0 + 16), f"t={current_step}", FONT_SMALL, TEXT)
    _text(draw, (PLOT_X0, PLOT_Y0 - 36), "D(t) (robust normalized pose deviation)", FONT_LABEL, TEXT)
    _text(draw, (PLOT_X0 + 490, PLOT_Y0 - 32), "green expert residual   |   red fail residual   |   dashed = threshold", FONT_SMALL, MUTED)
    _text(draw, (PLOT_X0, PLOT_Y0 + PLOT_H + 43), "environment step (10 Hz)", FONT_SMALL, MUTED)
    _text(draw, (PLOT_X0 + 700, PLOT_Y0 + PLOT_H + 43), f"q=.95 fail onset = step {data['fail_alarm']}  |  confirmation ≈ step {data['fail_alarm'] + 5}", FONT_SMALL, THRESHOLD)
    if current_step >= data["fail_alarm"]:
        _text(draw, (PLOT_X0 + 16, PLOT_Y0 + 18), f"persistent crossing starts at step {data['fail_alarm']}", FONT_LABEL, FAIL)
    if data["expert_length"] <= data["horizon"]:
        _text(draw, (PLOT_X0 + 16, PLOT_Y0 + PLOT_H - 28), f"expert recorded through step {data['expert_length'] - 1}; continuation is held for display", FONT_SMALL, MUTED)
    return canvas


def _line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill: tuple[int, int, int], width: int) -> None:
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")
    for point in points:
        radius = 5
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill=fill)


def _title_frame(title: str, subtitle: str, data: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (75, 175), title, FONT_CARD)
    _text(draw, (78, 250), subtitle, FONT_LABEL, MUTED)
    _text(draw, (78, 330), f"q=.95 threshold = {data['thresholds'][0.95]:.3f}   |   fail onset = step {data['fail_alarm']}   |   confirmation ≈ step {data['fail_alarm'] + 5}", FONT_LABEL, THRESHOLD)
    _text(draw, (78, 405), "green = expert residual    red = OOD-fail residual", FONT_LABEL, TEXT)
    return canvas


def _end_frame(stack: dict[str, Any], air: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (75, 180), "Reading the curves", FONT_CARD)
    _text(draw, (78, 270), "The red curve is the OOD fail trajectory's ERD score.", FONT_LABEL, FAIL)
    _text(draw, (78, 320), "The green curve is an expert demo's residual to its nearest expert peer.", FONT_LABEL, EXPERT)
    _text(draw, (78, 390), "q=.95 is calibrated from expert residuals; it is not tuned on the fail curve.", FONT_LABEL, MUTED)
    return canvas


def _write_video(frames: Iterable[Image.Image], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.convert("RGB").tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output}")


def build(args: argparse.Namespace) -> None:
    stack = _task_data(args.stack_summary, args.stack_expert_root, 150000)
    air = _task_data(args.air_summary, args.air_expert_root, 160016)

    def frames() -> Iterable[Image.Image]:
        for _ in range(FPS):
            yield _title_frame("Expert vs OOD-fail ERD curves", "Stack Cube", stack)
        for frame in range(stack["horizon"] + 1):
            yield _plot_frame(stack, frame, "Stack Cube")
        for _ in range(FPS):
            yield _title_frame("Expert vs OOD-fail ERD curves", "Grab Plane", air)
        for frame in range(air["horizon"] + 1):
            yield _plot_frame(air, frame, "Grab Plane")
        for _ in range(FPS):
            yield _end_frame(stack, air)

    _write_video(frames(), args.output)
    print(json.dumps({"output": str(args.output), "fps": FPS, "width": WIDTH, "height": HEIGHT}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-summary", type=Path, required=True)
    parser.add_argument("--stack-expert-root", type=Path, required=True)
    parser.add_argument("--air-summary", type=Path, required=True)
    parser.add_argument("--air-expert-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
