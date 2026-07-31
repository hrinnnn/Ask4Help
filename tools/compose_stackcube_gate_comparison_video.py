#!/usr/bin/env python3
"""Compose two annotated StackCube gate videos into one durable comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np


DIVIDER = 12


def build_comparison_frame(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Join equally high annotated frames with a visible neutral divider."""
    if left.ndim != 3 or right.ndim != 3 or left.shape[2] != 3 or right.shape[2] != 3:
        raise ValueError("comparison frames must be RGB arrays")
    height = max(left.shape[0], right.shape[0])
    width = left.shape[1] + DIVIDER + right.shape[1]
    canvas = np.full((height, width, 3), 21, dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    right_x = left.shape[1] + DIVIDER
    canvas[: right.shape[0], right_x : right_x + right.shape[1]] = right
    return canvas


def compose(*, left_video: Path, right_video: Path, output: Path, fps: int) -> int:
    if not left_video.is_file() or not right_video.is_file():
        raise FileNotFoundError("both annotated input videos must exist")
    left_frames = [np.asarray(frame, dtype=np.uint8) for frame in imageio.get_reader(str(left_video))]
    right_frames = [np.asarray(frame, dtype=np.uint8) for frame in imageio.get_reader(str(right_video))]
    if not left_frames or not right_frames:
        raise ValueError("cannot compose an empty video")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", macro_block_size=16)
    try:
        for index in range(max(len(left_frames), len(right_frames))):
            left = left_frames[min(index, len(left_frames) - 1)]
            right = right_frames[min(index, len(right_frames) - 1)]
            writer.append_data(build_comparison_frame(left, right))
    finally:
        writer.close()
    return max(len(left_frames), len(right_frames))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    frames = compose(left_video=args.left_video, right_video=args.right_video, output=args.output, fps=args.fps)
    print(f"frames={frames} output={args.output}")


if __name__ == "__main__":
    main()
