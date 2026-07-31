#!/usr/bin/env python3
"""Render a StackCube gated-DAgger rollout with controller provenance.

The collector keeps a machine-readable controller timeline.  This tool turns
one archived rollout into a reviewable video: every frame shows the active
controller and a bottom timeline preserves the exact policy/expert boundary.
It deliberately reads the collector's recorded timeline instead of inferring
the boundary again during rendering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


POLICY_COLOR = (28, 172, 196)
EXPERT_COLOR = (244, 143, 44)
SUCCESS_COLOR = (72, 190, 112)
TEXT_COLOR = (242, 244, 248)
BACKGROUND = (21, 25, 32)


def _load_episode(path: Path, episode_index: int | None) -> dict[str, Any]:
    """Load one collector row from either JSON or JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty episode metadata file: {path}")
    if text.startswith("{") and "\n{" not in text:
        payload = json.loads(text)
        if "episodes" in payload:
            rows = payload["episodes"]
        else:
            return payload
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if episode_index is None:
        if len(rows) != 1:
            raise ValueError("--episode-index is required when metadata contains multiple episodes")
        return rows[0]
    for row in rows:
        if int(row["episode_index"]) == episode_index:
            return row
    raise ValueError(f"episode_index={episode_index} not found in {path}")


def _timeline_point(row: dict[str, Any], action_step: int) -> dict[str, Any]:
    timeline = row.get("timeline", [])
    if not timeline:
        return {"controller": "policy", "env_step": 0}
    return max(
        (point for point in timeline if int(point["env_step"]) <= action_step),
        key=lambda point: int(point["env_step"]),
        default=timeline[0],
    )


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, *, fill: tuple[int, int, int]) -> None:
    draw.text(xy, value, fill=fill, font=ImageFont.load_default())


def build_annotated_frame(
    frame: np.ndarray,
    row: dict[str, Any],
    *,
    frame_index: int,
    title: str,
) -> np.ndarray:
    """Overlay one frame without modifying the raw source recording."""
    base = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    width, height = base.size
    panel_height = 92
    canvas = Image.new("RGB", (width, height + panel_height), BACKGROUND)
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)

    action_step = min(max(0, frame_index - 1), max(0, int(row.get("steps", 1)) - 1))
    point = _timeline_point(row, action_step)
    controller = str(point.get("controller", "policy"))
    color = EXPERT_COLOR if controller == "expert" else POLICY_COLOR
    controller_name = "EXPERT" if controller == "expert" else "POLICY"
    steps = max(1, int(row.get("steps", 1)))
    method = str(row.get("method", ""))
    success = bool(row.get("success", False))

    _text(draw, (8, height + 7), title, fill=TEXT_COLOR)
    _text(draw, (8, height + 25), f"t={action_step:03d}/{steps:03d}  controller={controller_name}", fill=color)
    if method == "bridge_knn":
        score = point.get("score")
        threshold = point.get("threshold")
        if score is not None and threshold is not None:
            state = "ALARM" if bool(point.get("alarm")) else "monitoring"
            _text(draw, (8, height + 43), f"Bridge kNN={float(score):.5f}  threshold={float(threshold):.5f}  {state}", fill=TEXT_COLOR)
    elif method == "late_success":
        gate = int(row.get("expert_start_step") or 0)
        status = "human observed incomplete task -> expert recovery" if controller == "expert" else "human monitoring policy"
        _text(draw, (8, height + 43), f"human gate at t={gate:03d}: {status}", fill=TEXT_COLOR)

    bar_y = height + 69
    bar_x = 8
    bar_width = max(1, width - 16)
    draw.rounded_rectangle((bar_x, bar_y, bar_x + bar_width, bar_y + 12), radius=2, fill=(73, 79, 90))
    for point_index, segment in enumerate(row.get("timeline", [])):
        start = int(segment["env_step"])
        end = int(row["timeline"][point_index + 1]["env_step"]) if point_index + 1 < len(row.get("timeline", [])) else steps
        left = bar_x + round(bar_width * start / steps)
        right = bar_x + round(bar_width * end / steps)
        segment_color = EXPERT_COLOR if segment.get("controller") == "expert" else POLICY_COLOR
        draw.rectangle((left, bar_y, max(left + 1, right), bar_y + 12), fill=segment_color)
    gate_step = row.get("expert_start_step")
    if gate_step is not None:
        gate_x = bar_x + round(bar_width * int(gate_step) / steps)
        draw.line((gate_x, bar_y - 3, gate_x, bar_y + 15), fill=TEXT_COLOR, width=1)
    current_x = bar_x + round(bar_width * action_step / steps)
    draw.line((current_x, bar_y - 3, current_x, bar_y + 15), fill=(255, 235, 98), width=2)
    _text(draw, (width - 82, height + 7), "SUCCESS" if success else "NOT SUCCESS", fill=SUCCESS_COLOR if success else EXPERT_COLOR)
    return np.asarray(canvas)


def render_episode(*, row: dict[str, Any], output: Path, title: str, fps: int) -> dict[str, Any]:
    source = Path(row["video_path"])
    if not source.is_file():
        raise FileNotFoundError(f"raw collector video is missing: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # imageio's v3 FFMPEG wrapper differs across our persisted runtimes.  The
    # stable legacy writer works with both the current H20 image and macOS.
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", macro_block_size=16)
    frames = 0
    try:
        for frame_index, frame in enumerate(imageio.get_reader(str(source))):
            writer.append_data(build_annotated_frame(frame, row, frame_index=frame_index, title=title))
            frames += 1
    finally:
        writer.close()
    return {
        "source_video": str(source),
        "output_video": str(output),
        "frames": frames,
        "fps": fps,
        "method": row.get("method"),
        "seed": row.get("seed"),
        "expert_start_step": row.get("expert_start_step"),
        "success": row.get("success"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True, help="Collector episodes.jsonl or one JSON row")
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    row = _load_episode(args.episodes, args.episode_index)
    summary = render_episode(row=row, output=args.output, title=args.title, fps=args.fps)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
