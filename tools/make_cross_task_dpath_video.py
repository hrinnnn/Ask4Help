#!/usr/bin/env python3
"""Render a task-specific phase-aware D_path video from cross-task evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from make_ycb_dpath_video import (
    BG,
    BLUE,
    F_CARD,
    F_LABEL,
    F_SMALL,
    F_TITLE,
    FPS,
    GREEN,
    GRID,
    H,
    MUTED,
    ORANGE,
    PANEL,
    RED,
    TEXT,
    W,
    YELLOW,
    dashed,
    fit,
    line,
    read_frames,
    text,
)


COLORS = {"blue": BLUE, "orange": ORANGE, "red": RED, "green": GREEN, "yellow": YELLOW}


def select_task(payload: dict[str, Any], name: str) -> dict[str, Any]:
    for task in payload["tasks"]:
        if task["task"] == name:
            return task
    raise KeyError(f"task not found: {name}")


def curve(task: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return task["groups"][spec["group"]][spec["category"]]


def title_frame(task: dict[str, Any], config: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (75, 155), config["title"], F_CARD)
    text(draw, (78, 225), config["subtitle"], F_LABEL, MUTED)
    text(draw, (78, 305), f"D_path threshold = {task['threshold']:.2f}", F_LABEL, GREEN)
    y = 355
    for spec in config["series"]:
        data = curve(task, spec)
        median = data.get("episode_median_0_40")
        if median is None:
            continue
        text(draw, (78, y), f"{spec['label']} median (0–40) = {median:.2f}", F_LABEL, COLORS[spec["color"]])
        y += 46
    text(draw, (78, 560), config["status"], F_LABEL, MUTED)
    return canvas


def end_frame(config: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (75, 155), "Readout", F_CARD)
    y = 250
    for item in config["readout"]:
        color = COLORS.get(item.get("color", "green"), TEXT)
        text(draw, (78, y), item["text"], F_LABEL, color)
        y += 58
    return canvas


def draw_plot(canvas: Image.Image, draw: ImageDraw.ImageDraw, task: dict[str, Any], config: dict[str, Any], frame: int, total: int) -> None:
    x0, y0, width, height = 90, 345, 1115, 295
    draw.rectangle((x0, y0, x0 + width, y0 + height), fill=PANEL, outline=GRID, width=1)
    horizon = int(config["horizon"])
    raw_current = frame / max(1, total - 1) * horizon
    current = min(horizon, int(round(raw_current / 5.0) * 5))
    series_data = [(spec, curve(task, spec)) for spec in config["series"]]
    y_max = 0.0
    for _, data in series_data:
        for row in data["time_distribution"]:
            if row["p95"] is not None:
                y_max = max(y_max, float(row["p95"]))
    y_max = max(10.0, float(np.ceil(y_max / 20.0) * 20.0))

    def sx(step: float) -> float:
        return x0 + float(step) / max(1, horizon) * width

    def sy(value: float) -> float:
        return y0 + height - float(value) / y_max * height

    for tick in np.linspace(0, y_max, 5):
        py = sy(float(tick))
        draw.line((x0, py, x0 + width, py), fill=GRID, width=1)
        text(draw, (x0 - 55, int(py - 9)), f"{tick:g}", F_SMALL, MUTED)
    for tick in np.linspace(0, horizon, 5):
        px = sx(float(tick))
        draw.line((px, y0, px, y0 + height), fill=GRID, width=1)
        text(draw, (int(px - 12), y0 + height + 7), f"{tick:g}", F_SMALL, MUTED)

    for spec, data in series_data:
        color = COLORS[spec["color"]]
        visible = [row for row in data["time_distribution"] if row["step"] <= current and row["median"] is not None]
        points = [(sx(row["step"]), sy(float(row["median"]))) for row in visible]
        line(draw, points, color, width=3)
        upper = [(sx(row["step"]), sy(float(row["p75"]))) for row in visible if row["p75"] is not None]
        lower = [(sx(row["step"]), sy(float(row["p25"]))) for row in visible[::-1] if row["p25"] is not None]
        if len(upper) > 1 and len(lower) > 1:
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).polygon(upper + lower, fill=(*color, 44))
            canvas.paste(overlay, (0, 0), overlay)

    threshold = float(task["threshold"])
    dashed(draw, (x0, sy(threshold)), (x0 + width, sy(threshold)), GREEN, width=2)
    text(draw, (x0 + width - 180, int(sy(threshold) - 21)), f"calibrated tau {threshold:.2f}", F_SMALL, GREEN)
    text(draw, (x0, y0 - 32), "phase-aware TCP-position D_path(t): median with P25–P75 bands", F_LABEL)
    legend = "  |  ".join(spec["label"] for spec in config["series"])
    text(draw, (x0 + 650, y0 - 29), legend, F_SMALL, MUTED)
    cursor = sx(current)
    draw.line((cursor, y0, cursor, y0 + height), fill=TEXT, width=1)
    text(draw, (int(min(cursor + 6, x0 + width - 75)), y0 + 10), f"t={current}", F_SMALL, TEXT)

    zoom_max = float(config.get("zoom_max", min(30.0, y_max)))
    ix, iy, iw, ih = x0 + 770, y0 + 31, 325, 135
    draw.rectangle((ix, iy, ix + iw, iy + ih), fill=(23, 28, 38), outline=GRID, width=1)

    def zx(step: float) -> float:
        return ix + float(step) / max(1, horizon) * iw

    def zy(value: float) -> float:
        return iy + ih - min(zoom_max, float(value)) / zoom_max * ih

    for spec, data in series_data:
        color = COLORS[spec["color"]]
        visible = [row for row in data["time_distribution"] if row["step"] <= current and row["median"] is not None]
        line(draw, [(zx(row["step"]), zy(float(row["median"]))) for row in visible], color, width=2)
    text(draw, (ix + 8, iy + 6), f"zoom: D_path ≤ {zoom_max:g}", F_SMALL, TEXT)

    risk = []
    for spec, data in series_data:
        row = next((row for row in data["time_distribution"] if int(row["step"]) == current), None)
        risk.append(f"{spec['label']} {0 if row is None else row['n']}")
    text(draw, (x0, y0 + height + 36), "risk set n(t): " + "  |  ".join(risk), F_SMALL, MUTED)
    text(draw, (x0 + 590, y0 + height + 36), config["footer"], F_SMALL, YELLOW)


def render(args: argparse.Namespace) -> None:
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    task = select_task(payload, config["task"])
    video_frames = {panel["key"]: read_frames(Path(panel["video"])) for panel in config["panels"]}
    frames: list[Image.Image] = [title_frame(task, config)] * int(config.get("title_seconds", 3) * FPS)
    segment_frames = int(config.get("segment_seconds", 20) * FPS)
    panel_count = len(config["panels"])
    if panel_count == 4:
        panel_width, panel_height, gap = 300, 145, 15
    elif panel_count == 3:
        panel_width, panel_height, gap = 390, 185, 25
    else:
        panel_width, panel_height, gap = 500, 185, 30
    total_width = panel_count * panel_width + (panel_count - 1) * gap
    left = (W - total_width) // 2
    for frame_index in range(segment_frames):
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        text(draw, (28, 18), config["title"], F_TITLE)
        text(draw, (30, 58), config["subtitle"], F_SMALL, MUTED)
        for index, panel in enumerate(config["panels"]):
            x = left + index * (panel_width + gap)
            y = 91
            source = video_frames[panel["key"]]
            canvas.paste(fit(source[min(frame_index, len(source) - 1)], (panel_width, panel_height)), (x, y))
            text(draw, (x + 8, y + 8), panel["label"], F_SMALL, COLORS[panel["color"]])
        draw_plot(canvas, draw, task, config, frame_index, segment_frames)
        frames.append(canvas)
    frames.extend([end_frame(config)] * int(config.get("end_seconds", 3) * FPS))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.asarray(frame.convert("RGB"), dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(json.dumps({"output": str(args.output), "frames": len(frames), "duration": len(frames) / FPS}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
