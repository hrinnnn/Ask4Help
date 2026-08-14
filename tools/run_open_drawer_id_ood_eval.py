#!/usr/bin/env python3
"""Run the frozen OpenDrawer policy evaluation across all four fixed splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SPLITS = (
    ("id", 76000),
    ("handle_ood", 77000),
    ("grasp_ood", 78000),
    ("goal_ood", 79000),
)


def _write_state(path: Path, *, stage: str, status: str, detail: str = "") -> None:
    payload = {
        "task": "OpenDrawerRetrievePlace",
        "stage": stage,
        "status": status,
        "detail": detail,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _complete(output_dir: Path, episodes: int) -> bool:
    summary_path = output_dir / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return False
    videos = list((output_dir / "videos").glob("*.mp4"))
    return summary.get("episodes") == episodes and len(videos) == episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    state_path = args.output_root / "pipeline_state.json"
    common = [
        str(args.python),
        "-u",
        str(args.evaluator),
        "--checkpoint",
        str(args.checkpoint),
        "--pi05-base",
        str(args.pi05_base),
        "--norm-stats",
        str(args.norm_stats),
        "--episodes",
        str(args.episodes),
        "--execute-horizon",
        str(args.execute_horizon),
        "--max-episode-steps",
        str(args.max_episode_steps),
    ]
    for split, seed in SPLITS:
        output_dir = args.output_root / split
        complete_marker = output_dir / "EVAL_COMPLETE"
        if complete_marker.exists() and _complete(output_dir, args.episodes):
            continue
        if output_dir.exists() and any(output_dir.iterdir()):
            _write_state(
                state_path,
                stage=split,
                status="blocked_partial_output",
                detail=f"existing incomplete output: {output_dir}",
            )
            raise SystemExit(2)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_state(state_path, stage=split, status="running")
        log_path = args.output_root / "logs" / f"{split}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = common + ["--output-dir", str(output_dir), "--seed", str(seed), "--split", split]
        env = os.environ.copy()
        with log_path.open("w") as log_file:
            completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
        if not _complete(output_dir, args.episodes):
            _write_state(
                state_path,
                stage=split,
                status="failed_incomplete",
                detail=f"evaluator returncode={completed.returncode}; artifacts incomplete",
            )
            raise SystemExit(completed.returncode or 1)
        complete_marker.write_text("evaluation summary and videos verified\n")
        _write_state(state_path, stage=split, status="completed")
    _write_state(state_path, stage="all_splits", status="completed")
    (args.output_root / "EVAL_COMPLETE").write_text("all four split summaries and videos verified\n")
    print("OPEN_DRAWER_ID_OOD_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
