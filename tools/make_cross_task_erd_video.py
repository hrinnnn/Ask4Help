#!/usr/bin/env python3
"""Render a single visual audit video for the three selected tasks.

Each segment shows an expert reference, an ID policy rollout and an OOD policy
rollout above the corresponding ERD score distributions.  OpenDrawer is split
into its three registered OOD factors so that the reference/calibration scope
is explicit.  The output is diagnostic-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

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
RED = (242, 106, 93)
YELLOW = (248, 191, 72)
PURPLE = (180, 121, 219)
GRAY = (139, 149, 166)

PLOT = (70, 395, 1140, 270)


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in ("/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/SFNS.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


F_SMALL = font(18)
F_LABEL = font(22)
F_TITLE = font(32)
F_CARD = font(44)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, f: ImageFont.FreeTypeFont = F_SMALL, fill: tuple[int, int, int] = TEXT) -> None:
    draw.text(xy, value, font=f, fill=fill)


def dashed(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], color: tuple[int, int, int], width: int = 2, dash: int = 9) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length == 0:
        return
    for start in np.arange(0.0, length, dash * 2):
        end = min(start + dash, length)
        a, b = start / length, end / length
        draw.line(((x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)), fill=color, width=width)


def line(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], color: tuple[int, int, int], width: int = 3) -> None:
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")
    for x, y in points:
        r = 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def probe(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=,", str(path)], text=True).strip().split(",")
    return int(out[0]), int(out[1])


def read_frames(path: Path) -> list[Image.Image]:
    width, height = probe(path)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    size = width * height * 3
    frames: list[Image.Image] = []
    while True:
        chunk = proc.stdout.read(size)
        if len(chunk) != size:
            break
        frames.append(Image.frombytes("RGB", (width, height), chunk))
    proc.wait()
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    return frames


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def curve_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    steps = np.asarray([row["step"] for row in analysis["time_distribution"]], dtype=np.float64)
    p25 = np.asarray([np.nan if row["p25"] is None else row["p25"] for row in analysis["time_distribution"]], dtype=np.float64)
    median = np.asarray([np.nan if row["median"] is None else row["median"] for row in analysis["time_distribution"]], dtype=np.float64)
    p75 = np.asarray([np.nan if row["p75"] is None else row["p75"] for row in analysis["time_distribution"]], dtype=np.float64)
    expert = np.asarray([[np.nan if value is None else value for value in row] for row in analysis["expert_score_rows"]], dtype=np.float64)
    expert_steps = np.asarray([i * int(analysis["decision_stride"]) for i in range(expert.shape[1])], dtype=np.float64)

    def safe_quantile(q: float) -> np.ndarray:
        result = np.full(expert.shape[1], np.nan, dtype=np.float64)
        for index in range(expert.shape[1]):
            finite = expert[:, index][np.isfinite(expert[:, index])]
            if len(finite):
                result[index] = float(np.quantile(finite, q))
        return result

    return {
        "steps": steps,
        "p25": p25,
        "median": median,
        "p75": p75,
        "expert_steps": expert_steps,
        "expert_p25": safe_quantile(0.25),
        "expert_median": safe_quantile(0.50),
        "expert_p75": safe_quantile(0.75),
        "thresholds": {float(k): float(v) for k, v in analysis["expert_calibration_quantiles"].items()},
        "summary": analysis["summary_by_quantile"],
    }


def fit_frame(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.BILINEAR)


def draw_segment(config: dict[str, Any], frame: int, frames: dict[str, list[Image.Image]], curves: dict[str, dict[str, Any]]) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    title = config["title"]
    horizon = int(config["horizon"])
    current_step = min(horizon, int(round(frame / 120 * horizon)))
    text(draw, (30, 20), f"Cross-task ERD-Pose evidence  |  {title}", F_TITLE)
    text(draw, (30, 61), config["subtitle"], F_SMALL, MUTED)
    text(draw, (30, 83), "diagnostic-only  |  green expert  |  blue ID policy  |  red OOD policy", F_SMALL, MUTED)

    panel_size = (390, 260)
    panel_y = 115
    for x, key, label, color in ((25, "expert_video", config["expert_label"], GREEN), (445, "id_video", config["id_label"], BLUE), (865, "ood_video", config["ood_label"], RED)):
        source = frames[key]
        image = source[min(frame, len(source) - 1)]
        canvas.paste(fit_frame(image, panel_size), (x, panel_y))
        text(draw, (x + 8, panel_y + 8), label, F_SMALL, color)

    plot_x, plot_y, plot_w, plot_h = PLOT
    draw.rectangle((plot_x, plot_y, plot_x + plot_w, plot_y + plot_h), fill=PANEL, outline=GRID, width=1)
    all_curves = list(curves.values())
    max_values: list[float] = []
    for curve in all_curves:
        for key in ("p75", "expert_p75"):
            finite = curve[key][np.isfinite(curve[key])]
            if len(finite):
                max_values.append(float(np.max(finite)))
        max_values.extend(float(v) for v in curve["thresholds"].values())
    y_max = max(max_values or [10.0])
    y_max = max(10.0, float(np.ceil(y_max / 10.0) * 10.0))

    def sx(step: float) -> float:
        return plot_x + float(step) / max(1, horizon) * plot_w

    def sy(value: float) -> float:
        return plot_y + plot_h - float(value) / y_max * plot_h

    for tick in np.linspace(0, y_max, 5):
        py = sy(float(tick))
        draw.line((plot_x, py, plot_x + plot_w, py), fill=GRID, width=1)
        text(draw, (plot_x - 48, int(py - 10)), f"{tick:g}", F_SMALL, MUTED)
    x_ticks = [0, horizon / 4, horizon / 2, horizon * 3 / 4, horizon]
    for tick in x_ticks:
        px = sx(tick)
        draw.line((px, plot_y, px, plot_y + plot_h), fill=GRID, width=1)
        text(draw, (int(px - 12), plot_y + plot_h + 8), f"{tick:g}", F_SMALL, MUTED)

    def draw_distribution(curve: dict[str, Any], color: tuple[int, int, int], label: str) -> None:
        visible = curve["steps"] <= current_step
        points = [(sx(step), sy(value)) for step, value, ok in zip(curve["steps"], curve["median"], visible) if ok and np.isfinite(value)]
        line(draw, points, color, 3)
        band = [(sx(step), sy(value)) for step, value, ok in zip(curve["steps"], curve["p75"], visible) if ok and np.isfinite(value)]
        low = [(sx(step), sy(value)) for step, value, ok in zip(curve["steps"][::-1], curve["p25"][::-1], visible[::-1]) if ok and np.isfinite(value)]
        if len(band) >= 2 and len(low) >= 2:
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).polygon(band + low, fill=(*color, 45))
            canvas.paste(overlay, (0, 0), overlay)
        text(draw, (plot_x + 12 + 180 * (0 if label == "ID" else 1), plot_y + plot_h - 31), f"{label} median", F_SMALL, color)

    # Expert reference is taken from the selected expert set associated with the
    # segment; ID/OOD curves are learner distributions.
    expert_curve = curves["expert"]
    visible_expert = expert_curve["expert_steps"] <= current_step
    line(draw, [(sx(step), sy(value)) for step, value, ok in zip(expert_curve["expert_steps"], expert_curve["expert_median"], visible_expert) if ok and np.isfinite(value)], GREEN, 3)
    draw_distribution(curves["id"], BLUE, "ID")
    draw_distribution(curves["ood"], RED, "OOD")
    text(draw, (plot_x, plot_y - 31), "ERD score D(t): median curves with P25–P75 bands", F_LABEL)
    text(draw, (plot_x + 515, plot_y - 27), "green expert  |  blue ID  |  red OOD", F_SMALL, MUTED)

    # Thresholds are shown for the reference used by each learner curve. For
    # split-specific OpenDrawer segments they are intentionally distinct.
    for name, curve in (("ID", curves["id"]), ("OOD", curves["ood"])):
        threshold_map = curve["thresholds"]
        for q, color in ((0.925, YELLOW), (0.95, GRAY)):
            if q not in threshold_map:
                continue
            py = sy(threshold_map[q])
            dashed(draw, (plot_x, py), (plot_x + plot_w, py), color, 2, 8)
            label_x = plot_x + plot_w - 190 if name == "ID" else plot_x + plot_w - 90
            label_y = int(py - 20 - (14 if q == 0.925 else 0))
            text(draw, (label_x, label_y), f"{name} q={q:g}", F_SMALL, color)

    cursor = sx(current_step)
    draw.line((cursor, plot_y, cursor, plot_y + plot_h), fill=TEXT, width=1)
    text(draw, (int(cursor + 6), plot_y + 13), f"t={current_step}", F_SMALL, TEXT)
    text(draw, (plot_x, plot_y + plot_h + 42), "environment step", F_SMALL, MUTED)
    id95 = curves["id"]["summary"].get("0.95", {})
    ood95 = curves["ood"]["summary"].get("0.95", {})
    text(draw, (plot_x + 280, plot_y + plot_h + 42), f"q=.95 onset: ID median {id95.get('median_alarm_step')}  |  OOD median {ood95.get('median_alarm_step')}", F_SMALL, YELLOW)
    text(draw, (30, 690), config["legacy"], F_SMALL, MUTED)
    return canvas


def title_frame(config: dict[str, Any], curves: dict[str, dict[str, Any]]) -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (75, 170), config["title"], F_CARD)
    text(draw, (78, 245), config["subtitle"], F_LABEL, MUTED)
    id95 = curves["id"]["summary"].get("0.95", {})
    ood95 = curves["ood"]["summary"].get("0.95", {})
    text(draw, (78, 330), f"ID q=.95 threshold {curves['id']['thresholds'].get(0.95, float('nan')):.3f}, median onset {id95.get('median_alarm_step')}", F_LABEL, BLUE)
    text(draw, (78, 375), f"OOD q=.95 threshold {curves['ood']['thresholds'].get(0.95, float('nan')):.3f}, median onset {ood95.get('median_alarm_step')}", F_LABEL, RED)
    text(draw, (78, 450), "This video visualizes existing trajectories; it does not change any formal training pipeline.", F_LABEL, MUTED)
    return canvas


def end_frame() -> Image.Image:
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (75, 175), "Cross-task readout", F_CARD)
    text(draw, (78, 265), "q=.925 is shown as a common calibration quantile, not a common raw distance.", F_LABEL, YELLOW)
    text(draw, (78, 325), "StackCube Stage 2: reference and policy replay are available; ID uses OOD expert context.", F_LABEL, MUTED)
    text(draw, (78, 370), "YCB: ID expert reference; OOD object policy trajectories replayed with the same Panda geometry.", F_LABEL, MUTED)
    text(draw, (78, 415), "OpenDrawer: split-specific expert suffix references; interpret as diagnostic until full-prefix references are available.", F_LABEL, MUTED)
    return canvas


def write_video(config_path: Path, output: Path) -> None:
    config = load(config_path)
    segments = config["segments"]
    prepared: list[tuple[dict[str, Any], dict[str, dict[str, Any]]]] = []
    for segment in segments:
        expert_analysis = load(Path(segment["expert_analysis"]))
        id_analysis = load(Path(segment["id_analysis"]))
        ood_analysis = load(Path(segment["ood_analysis"]))
        curves = {"expert": curve_summary(expert_analysis), "id": curve_summary(id_analysis), "ood": curve_summary(ood_analysis)}
        prepared.append((segment, curves))

    def frames() -> Iterable[Image.Image]:
        for segment, curves in prepared:
            # Decode one task segment at a time so the three 400-step
            # OpenDrawer clips do not remain resident while later segments are
            # rendered.
            videos = {key: read_frames(Path(segment[key])) for key in ("expert_video", "id_video", "ood_video")}
            for _ in range(FPS):
                yield title_frame(segment, curves)
            for frame in range(121):
                yield draw_segment(segment, frame, videos, curves)
        for _ in range(2 * FPS):
            yield end_frame()

    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(output)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames():
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {output}")
    print(json.dumps({"output": str(output), "segments": len(segments), "fps": FPS, "frames": len(segments) * (FPS + 121) + 2 * FPS}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_video(args.config, args.output)


if __name__ == "__main__":
    main()
