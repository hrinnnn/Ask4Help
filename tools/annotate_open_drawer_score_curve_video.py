#!/usr/bin/env python3
"""Annotate OpenDrawer rollouts with per-layer score curves and alarm times."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


SHORT_LABELS = {
    "vlm_input_visual_mean": "VLM input",
    "vlm_bridge_final_mean": "VLM bridge",
    "vlm_block_08_mean": "VLM B08",
    "action_expert_block_08": "Action B08",
    "action_expert_final": "Action final",
    "vlm_input_visual_tokens": "VLM input tok",
    "vlm_bridge_visual_tokens": "VLM bridge tok",
    "vlm_block_08_visual_tokens": "VLM B08 tok",
    "action_expert_block_08": "Action B08",
    "action_expert_final": "Action final",
}


def _base_name(name: str) -> str:
    return name.split("__", 1)[0]


def _short_label(name: str) -> str:
    return SHORT_LABELS.get(_base_name(name), _base_name(name)[:18])


def _ordered_names(timeline: list[dict[str, Any]]) -> list[str]:
    names = list(timeline[0]["scores"])
    order = {
        "vlm_input_visual_mean": 0,
        "vlm_input_visual_tokens": 0,
        "vlm_bridge_final_mean": 1,
        "vlm_bridge_visual_tokens": 1,
        "vlm_block_08_mean": 2,
        "vlm_block_08_visual_tokens": 2,
        "action_expert_block_08": 3,
        "action_expert_final": 4,
    }
    return sorted(names, key=lambda name: (order.get(_base_name(name), 99), name))


def _dimensions(video: Path) -> tuple[int, int]:
    text = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        text=True,
    ).strip()
    width, height = (int(value) for value in text.split(","))
    return width, height


def _fps(video: Path) -> float:
    value = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        text=True,
    ).strip()
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def _font(path: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _threshold_values(thresholds: dict[str, Any], quantile: str, name: str) -> list[float]:
    value = thresholds.get(quantile, {}).get(name)
    if isinstance(value, dict):
        return [float(item) for item in value.values() if isinstance(item, (int, float)) and math.isfinite(float(item))]
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    return []


def _threshold_for_point(thresholds: dict[str, Any], quantile: str, name: str, point: dict[str, Any] | None) -> float | None:
    value = thresholds.get(quantile, {}).get(name)
    if isinstance(value, dict):
        phase = (point or {}).get("selected_phase", {}).get(_base_name(name))
        value = value.get(str(phase)) if phase is not None else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines) or "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds-json", type=Path, help="summary JSON used when the timeline omits thresholds")
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    parser.add_argument("--header-height", type=int, default=144)
    parser.add_argument("--panel-width", type=int, default=560)
    args = parser.parse_args()

    payload = json.loads(args.timeline.read_text())
    if not payload.get("thresholds") and args.thresholds_json:
        threshold_payload = json.loads(args.thresholds_json.read_text())
        payload["thresholds"] = threshold_payload.get("thresholds", threshold_payload)
    timeline = payload["timeline"]
    names = _ordered_names(timeline)
    thresholds = payload.get("thresholds", {})
    width, height = _dimensions(args.video)
    fps = _fps(args.video)
    header = int(args.header_height)
    panel_width = int(args.panel_width)
    if header < 80 or panel_width < 360:
        raise ValueError("header-height or panel-width is too small for readable annotations")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    max_step = max(int(point["env_step"]) for point in timeline)
    max_step = max(max_step, int(payload.get("first_drawer_opened_env_step") or 0), 1)
    canvas_width = width + panel_width
    canvas_height = height + header
    row_height = max(55, (canvas_height - 52) // max(1, len(names)))
    title_font = _font(args.font, 15)
    label_font = _font(args.font, 14)
    small_font = _font(args.font, 12)
    alarm_font = _font(args.font, 16)

    score_values = {name: [float(point["scores"][name]) for point in timeline] for name in names}
    first_alarm = payload.get("first_alarm_env_step", {})
    boundary = payload.get("first_drawer_opened_env_step")
    split = payload.get("split", "rollout")
    seed = payload.get("seed", "?")

    probe = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-i", str(args.video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    writer = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{canvas_width}x{canvas_height}", "-r", str(fps), "-i", "-", "-an",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    if probe.stdout is None or writer.stdin is None:
        raise RuntimeError("failed to open video pipes")

    frame_bytes = width * height * 3
    frame_index = 0
    while True:
        raw = probe.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        current_step = frame_index
        current_point = None
        for point in timeline:
            if int(point["env_step"]) <= current_step:
                current_point = point
            else:
                break

        canvas = Image.new("RGB", (canvas_width, canvas_height), (18, 24, 38))
        scene = Image.frombytes("RGB", (width, height), raw)
        canvas.paste(scene, (0, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), f"OpenDrawer {split} seed={seed} | current env_step={current_step}", fill=(245, 245, 245), font=title_font)
        draw.text((8, 31), f"drawer_opened_env_step={boundary if boundary is not None else '-'} | score curves are decision-time values", fill=(215, 220, 230), font=small_font)
        alarm_now = []
        if current_point is not None:
            for q in ("0.8", "0.95"):
                alarm_now.extend(
                    f"q={q} {_short_label(name)}"
                    for name, active in current_point["alarms"].get(q, {}).items()
                    if active
                )
        alarm_text = _wrap_text(
            draw,
            "ALARM NOW: " + (", ".join(alarm_now) if alarm_now else "none"),
            alarm_font,
            width - 16,
        )
        draw.multiline_text(
            (8, 55),
            alarm_text,
            fill=(255, 112, 112) if alarm_now else (180, 190, 205),
            font=alarm_font,
            spacing=2,
        )

        panel_x = width
        draw.rectangle((panel_x, 0, canvas_width, canvas_height), fill=(17, 24, 39))
        draw.text((panel_x + 12, 7), "PCA score curves and alarm times", fill=(245, 245, 245), font=title_font)
        draw.text((panel_x + 12, 28), "white=score  blue=q.80  orange=q.95  red=current", fill=(195, 205, 220), font=small_font)

        for row, name in enumerate(names):
            y_top = 48 + row * row_height
            y_bottom = y_top + row_height - 18
            x_left = panel_x + 105
            x_right = canvas_width - 14
            values = score_values[name]
            candidates = list(values)
            candidates.extend(_threshold_values(thresholds, "0.8", name))
            candidates.extend(_threshold_values(thresholds, "0.95", name))
            candidates = _finite(candidates)
            low = min([0.0] + candidates)
            high = max([1.0] + candidates)
            margin = max(1e-6, (high - low) * 0.12)
            low -= margin * 0.15
            high += margin

            draw.text((panel_x + 10, y_top + 2), _short_label(name), fill=(230, 235, 245), font=label_font)
            q80_step = first_alarm.get("0.8", {}).get(name)
            q95_step = first_alarm.get("0.95", {}).get(name)
            draw.text((panel_x + 10, y_top + 22), f"q80={q80_step if q80_step is not None else '-'} q95={q95_step if q95_step is not None else '-'}", fill=(190, 200, 215), font=small_font)
            draw.rectangle((x_left, y_top, x_right, y_bottom), outline=(68, 82, 105), width=1)
            for fraction in (0.0, 0.5, 1.0):
                y = int(y_bottom - fraction * (y_bottom - y_top))
                draw.line((x_left, y, x_right, y), fill=(38, 50, 70), width=1)

            def x_for(step: float) -> int:
                return int(x_left + (step / max_step) * (x_right - x_left))

            def y_for(value: float) -> int:
                return int(y_bottom - ((value - low) / max(1e-9, high - low)) * (y_bottom - y_top))

            q80 = _threshold_for_point(thresholds, "0.8", name, current_point)
            q95 = _threshold_for_point(thresholds, "0.95", name, current_point)
            if q80 is not None:
                y = y_for(q80)
                draw.line((x_left, y, x_right, y), fill=(65, 155, 255), width=1)
            if q95 is not None:
                y = y_for(q95)
                draw.line((x_left, y, x_right, y), fill=(255, 170, 65), width=1)
            points = [(x_for(int(point["env_step"])), y_for(float(point["scores"][name]))) for point in timeline]
            if len(points) > 1:
                draw.line(points, fill=(240, 245, 250), width=2)
            for alarm_step, color in ((q80_step, (255, 80, 80)), (q95_step, (255, 170, 65))):
                if alarm_step is not None:
                    point = next((point for point in timeline if int(point["env_step"]) == int(alarm_step)), None)
                    if point is not None:
                        cx, cy = x_for(int(alarm_step)), y_for(float(point["scores"][name]))
                        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color, outline=(255, 255, 255))
                        draw.line((cx, y_top, cx, y_bottom), fill=color, width=1)
            current_x = x_for(current_step)
            draw.line((current_x, y_top, current_x, y_bottom), fill=(240, 70, 70), width=1)
            if current_point is not None:
                current_y = y_for(float(current_point["scores"][name]))
                draw.ellipse((current_x - 3, current_y - 3, current_x + 3, current_y + 3), fill=(255, 255, 255))

        writer.stdin.write(canvas.tobytes())
        frame_index += 1

    writer.stdin.close()
    writer.wait()
    probe.wait()
    if writer.returncode != 0:
        raise SystemExit(f"ffmpeg encoder failed: {writer.returncode}")
    print(json.dumps({"output": str(args.output), "frames": frame_index, "detectors": names}, sort_keys=True))


if __name__ == "__main__":
    main()
