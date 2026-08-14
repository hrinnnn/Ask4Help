#!/usr/bin/env python3
"""Run one OpenDrawer split and mark it complete only after artifact validation."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python),
        "-u",
        str(args.evaluator),
        "--checkpoint",
        str(args.checkpoint),
        "--pi05-base",
        str(args.pi05_base),
        "--norm-stats",
        str(args.norm_stats),
        "--output-dir",
        str(args.output_dir),
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--split",
        args.split,
        "--execute-horizon",
        str(args.execute_horizon),
        "--max-episode-steps",
        str(args.max_episode_steps),
    ]
    completed = subprocess.run(command)
    summary_path = args.output_dir / "summary.json"
    valid = False
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
            valid = summary.get("episodes") == args.episodes and len(
                glob.glob(str(args.output_dir / "videos" / "*.mp4"))
            ) == args.episodes
        except json.JSONDecodeError:
            valid = False
    if not valid:
        raise SystemExit(completed.returncode or 1)
    (args.output_dir / "EVAL_COMPLETE").write_text(
        "evaluation summary and videos verified\n"
    )


if __name__ == "__main__":
    main()
