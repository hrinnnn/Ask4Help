#!/usr/bin/env python3
"""Render the PickSingleYCB phase-aware path-deviation visual audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


W, H, FPS = 1280, 720, 10
BG = (18, 22, 30)
PANEL = (29, 35, 46)
GRID = (77, 88, 104)
TEXT = (235, 239, 245)
MUTED = (166, 177, 192)
GREEN = (88, 196, 139)
BLUE = (88, 166, 245)
ORANGE = (247, 166, 72)
RED = (242, 106, 93)
YELLOW = (248, 191, 72)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_SMALL = font(17)
F_LABEL = font(21)
F_TITLE = font(31)
F_CARD = font(42)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, f: ImageFont.FreeTypeFont = F_SMALL, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, value, font=f, fill=fill)


def line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: tuple[int, int, int], width: int = 3) -> None:
    if len(points) < 2:
        return
    draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def dashed(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], color: tuple[int, int, int], width: int = 2, dash: int = 8) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        return
    for start in np.arange(0.0, length, dash * 2):
        end = min(start + dash, length)
        a, b = start / length, end / length
        draw.line(
            ((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)),
            fill=color,
            width=width,
        )


def read_frames(path: Path) -> list[Image.Image]:
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=,", str(path)],
        text=True,
    ).strip().split(",")
    width, height = int(probe[0]), int(probe[1])
    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    size = width * height * 3
    frames: list[Image.Image] = []
    while True:
        chunk = process.stdout.read(size)
        if len(chunk) != size:
            break
        frames.append(Image.frombytes("RGB", (width, height), chunk))
    process.wait()
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    return frames


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def draw_curve_panel(canvas: Image.Image, draw: ImageDraw.ImageDraw, analysis: dict[str, Any], frame: int, total_frames: int) -> None:
    plot_x, plot_y, plot_w, plot_h = 90, 345, 1115, 295
    draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), fill=PANEL, outline=GRID, width=1)
    horizon = int(analysis["horizon"])
    current_step = min(horizon, int(round(frame / max(1, total_frames - 1) * horizon)))
    step_rows = {int(row["step"]): row for row in analysis["groups"]["id_success"]["time_distribution"]}
    colors = {"id_success": BLUE, "id_failure": ORANGE, "ood": RED}
    labels = {"id_success": "ID-success", "id_failure": "ID-failure", "ood": "OOD"}
    max_value = 0.0
    for group in colors:
        for row in analysis["groups"][group]["time_distribution"]:
            if row["p95"] is not None:
                max_value = max(max_value, float(row["p95"]))
    max_value = max(10.0, float(np.ceil(max_value / 20.0) * 20.0))

    def sx(step: float) -> float:
        return plot_x + float(step) / max(1, horizon) * plot_w

    def sy(value: float) -> float:
        return plot_y + plot_h - float(value) / max_value * plot_h

    for tick in np.linspace(0, max_value, 5):
        py = sy(float(tick))
        draw.line((plot_x, py, plot_x + plot_w, py), fill=GRID, width=1)
        text(draw, (plot_x - 55, int(py - 9)), f"{tick:g}", F_SMALL, MUTED)
    for tick in (0, 50, 100, 150, 200):
        px = sx(tick)
        draw.line((px, plot_y, px, plot_y + plot_h), fill=GRID, width=1)
        text(draw, (int(px - 12), plot_y + plot_h + 7), str(tick), F_SMALL, MUTED)

    for group, color in colors.items():
        rows = analysis["groups"][group]["time_distribution"]
        visible = [row for row in rows if row["step"] <= current_step and row["median"] is not None]
        points = [(sx(row["step"]), sy(float(row["median"]))) for row in visible]
        line(draw, points, color, width=3)
        upper = [(sx(row["step"]), sy(float(row["p75"]))) for row in visible if row["p75"] is not None]
        lower = [(sx(row["step"]), sy(float(row["p25"]))) for row in visible[::-1] if row["p25"] is not None]
        if len(upper) >= 2 and len(lower) >= 2:
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).polygon(upper + lower, fill=(*color, 44))
            canvas.paste(overlay, (0, 0), overlay)

    threshold = float(analysis["expert_path_quantiles"]["p95"])
    dashed(draw, (plot_x, sy(threshold)), (plot_x + plot_w, sy(threshold)), GREEN, width=2)
    text(draw, (plot_x + plot_w - 170, int(sy(threshold) - 21)), f"expert q95 {threshold:.2f}", F_SMALL, GREEN)
    text(draw, (plot_x, plot_y - 32), "phase-aware TCP-position D_path(t): median with P25–P75 bands", F_LABEL)
    text(draw, (plot_x + 750, plot_y - 29), "blue success ID  |  orange failure ID  |  red OOD", F_SMALL, MUTED)

    cursor = sx(current_step)
    draw.line((cursor, plot_y, cursor, plot_y + plot_h), fill=TEXT, width=1)
    text(draw, (int(min(cursor + 6, plot_x + plot_w - 70)), plot_y + 11), f"t={current_step}", F_SMALL, TEXT)

    # A small early-window zoom keeps the successful-ID distribution visible
    # despite the much larger failure values.
    inset_x, inset_y, inset_w, inset_h = plot_x + 770, plot_y + 32, 325, 135
    draw.rectangle((inset_x, inset_y, inset_x + inset_w, inset_y + inset_h), fill=(23, 28, 38), outline=GRID, width=1)
    zoom_max = 25.0

    def zx(step: float) -> float:
        return inset_x + float(step) / max(1, horizon) * inset_w

    def zy(value: float) -> float:
        return inset_y + inset_h - min(zoom_max, float(value)) / zoom_max * inset_h

    for tick in (0, 10, 20):
        py = zy(tick)
        draw.line((inset_x, py, inset_x + inset_w, py), fill=GRID, width=1)
        text(draw, (inset_x - 27, int(py - 8)), str(tick), F_SMALL, MUTED)
    for group, color in colors.items():
        rows = analysis["groups"][group]["time_distribution"]
        visible = [row for row in rows if row["step"] <= current_step and row["median"] is not None]
        points = [(zx(row["step"]), zy(float(row["median"]))) for row in visible]
        line(draw, points, color, width=2)
    text(draw, (inset_x + 8, inset_y + 6), "zoom: D_path ≤ 25", F_SMALL, TEXT)

    n_success = step_rows.get(current_step, {}).get("n", 0)
    n_failure = next((row["n"] for row in analysis["groups"]["id_failure"]["time_distribution"] if int(row["step"]) == current_step), 0)
    n_ood = next((row["n"] for row in analysis["groups"]["ood"]["time_distribution"] if int(row["step"]) == current_step), 0)
    text(draw, (plot_x, plot_y + plot_h + 36), f"risk set n(t): success ID {n_success}  |  failure ID {n_failure}  |  OOD {n_ood}", F_SMALL, MUTED)
    summary = analysis["groups"]
    text(
        draw,
        (plot_x + 430, plot_y + plot_h + 36),
        f"first 0–40 episode median: success {summary['id_success']['episode_median_quantiles']['median']:.2f}  |  failure {summary['id_failure']['episode_median_quantiles']['median']:.2f}  |  OOD {summary['ood']['episode_median_quantiles']['median']:.2f}",
        F_SMALL,
        YELLOW,
    )


def title_frame(analysis: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (75, 165), "PickSingleYCB | phase-aware path deviation", F_CARD)
    text(draw, (78, 238), "Existing trajectories, re-aligned with an ID-derived two-sided phase band", F_LABEL, MUTED)
    q95 = analysis["expert_path_quantiles"]["p95"]
    s = analysis["groups"]["id_success"]["episode_median_quantiles"]["median"]
    f = analysis["groups"]["id_failure"]["episode_median_quantiles"]["median"]
    o = analysis["groups"]["ood"]["episode_median_quantiles"]["median"]
    text(draw, (78, 320), f"expert path q95 = {q95:.2f}", F_LABEL, GREEN)
    text(draw, (78, 368), f"ID-success median (0–40) = {s:.2f}", F_LABEL, BLUE)
    text(draw, (78, 416), f"ID-failure median (0–40) = {f:.2f}", F_LABEL, ORANGE)
    text(draw, (78, 464), f"OOD median (0–40) = {o:.2f}", F_LABEL, RED)
    text(draw, (78, 555), "Diagnostic visualization only; no training or formal gate was changed.", F_LABEL, MUTED)
    return canvas


def end_frame(analysis: dict[str, Any]) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (75, 160), "Readout", F_CARD)
    text(draw, (78, 255), "Successful ID paths stay close to the expert reference.", F_LABEL, BLUE)
    text(draw, (78, 310), "ID failures and OOD paths move to a clearly separated region.", F_LABEL, RED)
    text(draw, (78, 365), "The late ID curve is failure-only after successful episodes terminate.", F_LABEL, ORANGE)
    text(draw, (78, 465), "Primary view: D_path; phase lag and velocity remain auxiliary diagnostics.", F_LABEL, MUTED)
    return canvas


def render(args: argparse.Namespace) -> None:
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    videos = {key: read_frames(Path(value)) for key, value in args.videos.items()}
    frames: list[Image.Image] = [title_frame(analysis)] * int(args.title_seconds * FPS)
    total_segment = int(args.segment_seconds * FPS)
    panel_specs = [
        ("expert", "ID expert", GREEN),
        ("id_success", "ID-success", BLUE),
        ("id_failure", "ID-failure", ORANGE),
        ("ood", "OOD policy", RED),
    ]
    for frame_index in range(total_segment):
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        current_step = min(int(analysis["horizon"]), int(round(frame_index / max(1, total_segment - 1) * int(analysis["horizon"]))))
        text(draw, (28, 18), "PickSingleYCB | phase-aware D_path", F_TITLE)
        text(draw, (30, 58), "ID expert vs successful/failed ID and OOD policy trajectories", F_SMALL, MUTED)
        for index, (key, label, color) in enumerate(panel_specs):
            x = 20 + index * 315
            y = 95
            source = videos[key]
            canvas.paste(fit(source[min(frame_index, len(source) - 1)], (300, 145)), (x, y))
            text(draw, (x + 7, y + 7), label, F_SMALL, color)
        draw_curve_panel(canvas, draw, analysis, frame_index, total_segment)
        frames.append(canvas)
    frames.extend([end_frame(analysis)] * int(args.end_seconds * FPS))

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
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with return code {return_code}")
    print(json.dumps({"output": str(args.output), "frames": len(frames), "duration_seconds": len(frames) / FPS}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title-seconds", type=float, default=3.0)
    parser.add_argument("--segment-seconds", type=float, default=20.0)
    parser.add_argument("--end-seconds", type=float, default=3.0)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--id-success", type=Path, required=True)
    parser.add_argument("--id-failure", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    args = parser.parse_args()
    args.videos = {"expert": args.expert, "id_success": args.id_success, "id_failure": args.id_failure, "ood": args.ood}
    render(args)


if __name__ == "__main__":
    main()
