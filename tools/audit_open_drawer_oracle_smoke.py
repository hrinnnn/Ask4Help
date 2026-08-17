#!/usr/bin/env python3
"""Audit a trajectory-recording OpenDrawer Oracle smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    split = args.root / "id"
    summary_path = args.root / "summary.json"
    split_summary_path = split / "summary.json"
    episodes_path = split / "episodes.jsonl"
    summary = json.loads(split_summary_path.read_text()) if split_summary_path.is_file() else {}
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()] if episodes_path.is_file() else []
    videos = sorted((split / "videos").glob("*.mp4")) if (split / "videos").is_dir() else []
    if summary.get("attempts") != args.expected or summary.get("successes") != args.expected:
        errors.append(f"oracle_success:{summary.get('successes')}/{summary.get('attempts')}")
    if len(rows) != args.expected:
        errors.append(f"episode_rows:{len(rows)}!={args.expected}")
    if len(videos) != args.expected:
        errors.append(f"videos:{len(videos)}!={args.expected}")
    for row in rows:
        stages: dict[str, Any] = row.get("oracle", {}).get("stages", row)
        row_split = row.get("split", stages.get("split"))
        if row_split != "id" or stages.get("split") != "id":
            errors.append(f"non_id_row:{row.get('seed')}")
        if stages.get("success") is not True:
            errors.append(f"failed_stage_row:{row.get('seed')}")
        if not stages or not all(isinstance(value, (int, float)) and value >= 0 for key, value in stages.items() if key.endswith("_steps")):
            errors.append(f"invalid_stage_steps:{row.get('seed')}")

    ffprobe = shutil.which("ffprobe")
    video_decode_failures: list[str] = []
    if ffprobe is None:
        errors.append("ffprobe_missing")
    else:
        for video in videos:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 or not result.stdout.strip():
                video_decode_failures.append(video.name)
        if video_decode_failures:
            errors.extend(f"video_decode:{name}" for name in video_decode_failures)

    trajectory_files = sorted((split / "videos").glob("*.h5")) if (split / "videos").is_dir() else []
    trajectory_count = 0
    action_lengths: list[int] = []
    state_lengths: list[int] = []
    try:
        import h5py

        for trajectory_file in trajectory_files:
            with h5py.File(trajectory_file, "r") as handle:
                groups = [handle[name] for name in handle.keys() if name.startswith("traj_")]
                trajectory_count += len(groups)
                for group in groups:
                    if "actions" not in group:
                        errors.append(f"trajectory_actions_missing:{trajectory_file.name}:{group.name}")
                        continue
                    action_lengths.append(int(group["actions"].shape[0]))
                    state_datasets = []
                    if "env_states" in group:
                        def visit(_name: str, obj: Any) -> None:
                            if hasattr(obj, "shape") and obj.shape:
                                state_datasets.append(int(obj.shape[0]))
                        group["env_states"].visititems(visit)
                    if state_datasets:
                        state_lengths.append(max(state_datasets))
        if trajectory_count != args.expected:
            errors.append(f"trajectory_groups:{trajectory_count}!={args.expected}")
        if action_lengths and state_lengths and any(state != action + 1 for action, state in zip(action_lengths, state_lengths)):
            errors.append("state_action_boundary_mismatch")
    except ImportError:
        errors.append("h5py_missing")

    report = {
        "format": "open_drawer_oracle_smoke_audit_v1",
        "root": str(args.root),
        "summary": summary,
        "episodes": len(rows),
        "videos": len(videos),
        "video_decode_failures": video_decode_failures,
        "trajectory_files": [str(path) for path in trajectory_files],
        "trajectory_groups": trajectory_count,
        "action_lengths": action_lengths,
        "state_lengths": state_lengths,
        "errors": sorted(set(errors)),
        "pass": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
