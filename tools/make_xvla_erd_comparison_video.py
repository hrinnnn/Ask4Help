#!/usr/bin/env python3
"""Make a compact, diagnostic-only ERD-Pose comparison video.

The video pairs one held-out OOD fail rollout with its nearest-context expert
reference and overlays the per-episode ERD score together with the 50-rollout
distribution.  It is intentionally a presentation artifact; it does not
modify or relaunch the formal X-VLA pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH = 1280
HEIGHT = 720
FPS = 10
PLOT_X0 = 58
PLOT_Y0 = 438
PLOT_W = 840
PLOT_H = 238
HIST_X0 = 945
HIST_Y0 = 438
HIST_W = 286
HIST_H = 238

BG = (18, 22, 30)
PANEL = (29, 35, 46)
GRID = (77, 88, 104)
TEXT = (235, 239, 245)
MUTED = (166, 177, 192)
EXPERT = (88, 196, 139)
FAIL = (242, 106, 93)
MEDIAN = (88, 166, 245)
THRESHOLD = (248, 191, 72)
Q925 = (139, 149, 166)
Q975 = (180, 121, 219)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_SMALL = _font(18)
FONT_LABEL = _font(21)
FONT_TITLE = _font(30, bold=True)
FONT_CARD = _font(40, bold=True)


def _probe(path: Path) -> tuple[int, int, int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames",
        "-of",
        "csv=p=0:s=,",
        str(path),
    ]
    output = subprocess.check_output(command, text=True).strip().split(",")
    width, height = int(output[0]), int(output[1])
    frames = int(output[2]) if len(output) > 2 and output[2].isdigit() else 0
    return width, height, frames


def _read_frames(path: Path) -> list[Image.Image]:
    width, height, _ = _probe(path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    frame_size = width * height * 3
    frames: list[Image.Image] = []
    while True:
        chunk = process.stdout.read(frame_size)
        if len(chunk) != frame_size:
            break
        frames.append(Image.frombytes("RGB", (width, height), chunk))
    process.wait()
    if not frames:
        raise RuntimeError(f"no video frames decoded from {path}")
    return frames


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_data(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["rows"]
    stride = int(summary.get("learner", {}).get("decision_stride", 5))
    steps = np.arange(0, 151, stride, dtype=np.int64)
    matrix = np.asarray([row["distance_at_decision_steps"] for row in rows], dtype=np.float64)
    threshold_values = np.asarray(summary["threshold"]["calibration_values"], dtype=np.float64)
    q_values = {q: float(np.quantile(threshold_values, q)) for q in (0.925, 0.95, 0.975)}
    selected = rows[0]
    selected_index = 0
    data = {
        "task": summary.get("task", "task"),
        "rows": rows,
        "steps": steps[: matrix.shape[1]],
        "matrix": matrix,
        "selected": selected,
        "selected_index": selected_index,
        "selected_values": matrix[selected_index],
        "expert_calibration": threshold_values,
        "thresholds": q_values,
        "q95": q_values[0.95],
    }
    return data


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font: ImageFont.FreeTypeFont, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, value, font=font, fill=fill)


def _line(draw: ImageDraw.ImageDraw, points: Iterable[tuple[float, float]], fill: tuple[int, int, int], width: int = 2) -> None:
    points = list(points)
    if len(points) >= 2:
        draw.line(points, fill=fill, width=width, joint="curve")


def _dash_line(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], fill: tuple[int, int, int], width: int = 2, dash: int = 8) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        return
    for start in np.arange(0.0, length, dash * 2):
        end = min(start + dash, length)
        r0, r1 = start / length, end / length
        draw.line(
            [(x0 + (x1 - x0) * r0, y0 + (y1 - y0) * r0), (x0 + (x1 - x0) * r1, y0 + (y1 - y0) * r1)],
            fill=fill,
            width=width,
        )


def _score_x(step: float) -> float:
    return PLOT_X0 + float(step) / 150.0 * PLOT_W


def _draw_plot(canvas: Image.Image, data: dict[str, Any], frame: int) -> None:
    draw = ImageDraw.Draw(canvas)
    matrix = data["matrix"]
    steps = data["steps"]
    selected_values = data["selected_values"]
    p25, median, p75 = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0)
    y_max = max(float(np.max(matrix)), float(np.max(data["expert_calibration"])), float(np.max(selected_values)), float(data["q95"]))
    y_max = max(10.0, np.ceil(y_max / 10.0) * 10.0)

    def sy(value: float) -> float:
        return PLOT_Y0 + PLOT_H - float(value) / y_max * PLOT_H

    # Plot frame and light grid.
    draw.rectangle((PLOT_X0, PLOT_Y0, PLOT_X0 + PLOT_W, PLOT_Y0 + PLOT_H), fill=PANEL, outline=GRID, width=1)
    for y_tick in np.linspace(0, y_max, 5):
        y = sy(float(y_tick))
        draw.line((PLOT_X0, y, PLOT_X0 + PLOT_W, y), fill=GRID, width=1)
        _text(draw, (PLOT_X0 - 45, int(y - 10)), f"{y_tick:g}", FONT_SMALL, MUTED)
    for x_tick in (0, 25, 50, 75, 100, 125, 150):
        x = _score_x(x_tick)
        draw.line((x, PLOT_Y0, x, PLOT_Y0 + PLOT_H), fill=GRID, width=1)
        _text(draw, (int(x - 10), PLOT_Y0 + PLOT_H + 8), str(x_tick), FONT_SMALL, MUTED)

    upper = [(float(_score_x(step)), float(sy(value))) for step, value in zip(steps, p75)]
    lower = [(float(_score_x(step)), float(sy(value))) for step, value in zip(steps[::-1], p25[::-1])]
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(upper + lower, fill=(*MEDIAN, 50))
    canvas.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(canvas)
    _line(draw, [(float(_score_x(s)), float(sy(v))) for s, v in zip(steps, median)], MEDIAN, 3)
    _line(draw, [(float(_score_x(s)), float(sy(v))) for s, v in zip(steps, selected_values)], FAIL, 3)

    for q, color, label in ((0.925, Q925, "q.925"), (0.95, THRESHOLD, "q.95"), (0.975, Q975, "q.975")):
        y = sy(data["thresholds"][q])
        _dash_line(draw, (PLOT_X0, y), (PLOT_X0 + PLOT_W, y), color, 2, 7)
        _text(draw, (PLOT_X0 + PLOT_W - 70, int(y - 18)), label, FONT_SMALL, color)

    current_step = min(int(frame), 149)
    cursor_x = _score_x(current_step)
    draw.line((cursor_x, PLOT_Y0, cursor_x, PLOT_Y0 + PLOT_H), fill=TEXT, width=1)
    current_index = min(int(round(current_step / 5)), len(selected_values) - 1)
    current_score = float(selected_values[current_index])
    _text(draw, (PLOT_X0, PLOT_Y0 - 28), "ERD score D(t): 50 OOD fail trajectories", FONT_LABEL, TEXT)
    _text(draw, (PLOT_X0 + 440, PLOT_Y0 - 26), "blue median  |  blue band P25–P75  |  red selected fail", FONT_SMALL, MUTED)
    _text(draw, (PLOT_X0, PLOT_Y0 + PLOT_H + 31), "environment step", FONT_SMALL, MUTED)
    _text(draw, (PLOT_X0 - 2, PLOT_Y0 - 48), f"D(t), max={y_max:g}", FONT_SMALL, MUTED)
    _text(draw, (int(cursor_x + 6), PLOT_Y0 + 8), f"t={current_step}  D={current_score:.1f}", FONT_SMALL, TEXT)

    # Histogram: expert calibration residuals versus the fail-score snapshot.
    hist_values = matrix[:, current_index]
    h_max = y_max
    bins = np.linspace(0.0, h_max, 13)
    expert_hist, _ = np.histogram(data["expert_calibration"], bins=bins)
    fail_hist, _ = np.histogram(hist_values, bins=bins)
    max_count = max(int(np.max(expert_hist)), int(np.max(fail_hist)), 1)
    draw.rectangle((HIST_X0, HIST_Y0, HIST_X0 + HIST_W, HIST_Y0 + HIST_H), fill=PANEL, outline=GRID, width=1)
    bar_base = HIST_Y0 + HIST_H - 4
    max_bar_height = HIST_H - 88  # reserve the title/legend band above the bars
    for index in range(len(bins) - 1):
        x_left = HIST_X0 + (bins[index] / h_max) * HIST_W
        x_right = HIST_X0 + (bins[index + 1] / h_max) * HIST_W
        expert_h = expert_hist[index] / max_count * max_bar_height
        fail_h = fail_hist[index] / max_count * max_bar_height
        draw.rectangle((int(x_left), int(bar_base - expert_h), int((x_left + x_right) / 2), bar_base), fill=EXPERT)
        draw.rectangle((int((x_left + x_right) / 2), int(bar_base - fail_h), int(x_right), bar_base), fill=FAIL)
    _text(draw, (HIST_X0 + 10, HIST_Y0 + 8), "score distribution", FONT_LABEL, TEXT)
    _text(draw, (HIST_X0 + 10, HIST_Y0 + 34), "green expert calibration", FONT_SMALL, EXPERT)
    _text(draw, (HIST_X0 + 10, HIST_Y0 + 54), f"red OOD fail @ t={current_step}", FONT_SMALL, FAIL)
    _text(draw, (HIST_X0 + 10, HIST_Y0 + HIST_H + 8), "score", FONT_SMALL, MUTED)
    for x_tick in (0, h_max / 2, h_max):
        x = HIST_X0 + (x_tick / h_max) * HIST_W
        _text(draw, (int(x - 8), HIST_Y0 + HIST_H + 8), f"{x_tick:g}", FONT_SMALL, MUTED)


def _compose_frame(expert_frame: Image.Image, fail_frame: Image.Image, data: dict[str, Any], frame: int, task_label: str, expert_seed: int, fail_seed: int) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (32, 18), f"ERD-Pose diagnostic  |  {task_label}", FONT_TITLE, TEXT)
    _text(draw, (32, 55), f"step {min(frame, 149)} / 150  |  threshold q=.95 = {data['q95']:.3f}  |  persistence = 2 decision points", FONT_SMALL, MUTED)

    video_size = (610, 300)
    left = ImageOps.fit(expert_frame.convert("RGB"), video_size, method=Image.Resampling.BILINEAR)
    right = ImageOps.fit(fail_frame.convert("RGB"), video_size, method=Image.Resampling.BILINEAR)
    canvas.paste(left, (30, 88))
    canvas.paste(right, (640, 88))
    draw = ImageDraw.Draw(canvas)
    _text(draw, (38, 96), f"EXPERT reference  (seed {expert_seed})", FONT_LABEL, EXPERT)
    _text(draw, (648, 96), f"OOD fail  (seed {fail_seed})", FONT_LABEL, FAIL)
    if len(_read_video_frames_cache.get("expert", [])) < frame + 1:
        _text(draw, (38, 364), "expert video ended; final frame held", FONT_SMALL, MUTED)
    _draw_plot(canvas, data, frame)
    return canvas


_read_video_frames_cache: dict[str, list[Image.Image]] = {}


def _title_frame(title: str, subtitle: str, data: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (70, 170), title, FONT_CARD, TEXT)
    _text(draw, (72, 235), subtitle, FONT_LABEL, MUTED)
    _text(draw, (72, 310), f"q=.95 threshold: {data['q95']:.3f}    |    median alarm: {np.median([r['erd_alarm_step'] for r in data['rows'] if r.get('erd_alarm_step') is not None]):g} steps", FONT_LABEL, THRESHOLD)
    _text(draw, (72, 365), "green = expert reference    red = OOD fail    blue = fail-score distribution", FONT_LABEL, MEDIAN)
    return canvas


def _end_frame(stack: dict[str, Any], air: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (70, 155), "Diagnostic threshold sweep", FONT_CARD, TEXT)
    _text(draw, (72, 235), "Current defensible operating point: expert-calibrated q=.95", FONT_LABEL, THRESHOLD)
    _text(draw, (72, 300), f"StackCube: D threshold {stack['q95']:.3f}, median crossing step 20", FONT_LABEL, TEXT)
    _text(draw, (72, 345), f"Grab Plane: D threshold {air['q95']:.3f}, median crossing step 15", FONT_LABEL, TEXT)
    _text(draw, (72, 415), "This is a pose-deviation timing reference, not a proven Success-Rate optimum.", FONT_LABEL, MUTED)
    _text(draw, (72, 465), "Use held-out calibration and controlled takeover to make a formal claim.", FONT_LABEL, MUTED)
    return canvas


def _write_video(frames: Iterable[Image.Image], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(frame.convert("RGB").tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output}")


def _segment(data: dict[str, Any], expert_path: Path, fail_path: Path, task_label: str, expert_seed: int, fail_seed: int) -> Iterable[Image.Image]:
    expert_frames = _read_frames(expert_path)
    fail_frames = _read_frames(fail_path)
    _read_video_frames_cache["expert"] = expert_frames
    _read_video_frames_cache["fail"] = fail_frames
    for frame in range(min(len(fail_frames), 150)):
        expert_frame = expert_frames[min(frame, len(expert_frames) - 1)]
        yield _compose_frame(expert_frame, fail_frames[frame], data, frame, task_label, expert_seed, fail_seed)


def build(args: argparse.Namespace) -> None:
    stack = _task_data(_load_summary(args.stack_summary))
    air = _task_data(_load_summary(args.air_summary))

    def frames() -> Iterable[Image.Image]:
        for _ in range(FPS):
            yield _title_frame("ERD-Pose comparison", "Stack Cube: matched-context expert vs held-out OOD fail", stack)
        yield from _segment(stack, args.stack_expert, args.stack_fail, "Stack Cube", 150000, 151000)
        for _ in range(FPS):
            yield _title_frame("ERD-Pose comparison", "Grab Plane: matched-context expert vs held-out OOD fail", air)
        yield from _segment(air, args.air_expert, args.air_fail, "Grab Plane", 160016, 161000)
        for _ in range(2 * FPS):
            yield _end_frame(stack, air)

    _write_video(frames(), args.output)
    print(json.dumps({"output": str(args.output), "fps": FPS, "width": WIDTH, "height": HEIGHT}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack-summary", type=Path, required=True)
    parser.add_argument("--air-summary", type=Path, required=True)
    parser.add_argument("--stack-expert", type=Path, required=True)
    parser.add_argument("--stack-fail", type=Path, required=True)
    parser.add_argument("--air-expert", type=Path, required=True)
    parser.add_argument("--air-fail", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
