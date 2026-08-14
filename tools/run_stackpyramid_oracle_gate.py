#!/usr/bin/env python3
"""Restart-tolerant four-split StackPyramid oracle gate controller."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


SPLITS = ("id", "stage1_ood", "stage2_ood", "stage3_ood")


def write_state(path: Path, state: dict) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--min-successes", type=int, default=19)
    parser.add_argument("--gpu", type=str, default="1")
    args = parser.parse_args()
    if args.episodes <= 0 or not 0 < args.min_successes <= args.episodes:
        raise ValueError("invalid oracle gate counts")
    if args.local_root.exists() or args.persistent_root.exists():
        raise FileExistsError("oracle gate requires new local and persistent roots")

    args.local_root.mkdir(parents=True)
    state_path = args.local_root / "pipeline_state.json"
    state = {
        "format": "stackpyramid_oracle_gate_controller_v1",
        "stage": "pending",
        "splits": {split: "pending" for split in SPLITS},
        "episodes_per_split": args.episodes,
        "minimum_successes_per_split": args.min_successes,
        "persistent_root": str(args.persistent_root),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_state(state_path, state)

    for index, split in enumerate(SPLITS):
        state["stage"] = split
        state["splits"][split] = "running"
        write_state(state_path, state)
        stage_root = args.local_root / split
        log_path = args.local_root / f"{split}.log"
        command = [
            sys.executable,
            str(args.repo_root / "tools/collect_stackpyramid_oracle.py"),
            "--output",
            str(stage_root),
            "--split",
            split,
            "--episodes",
            str(args.episodes),
            "--start-seed",
            str(16000 + index * 1000),
            "--sim-backend",
            "cpu",
            "--render-backend",
            "gpu",
        ]
        env = dict(**__import__("os").environ)
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["PYTHONPATH"] = str(args.repo_root)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        summary_path = stage_root / f"oracle_summary_{split}.json"
        if result.returncode != 0 or not summary_path.exists():
            state["splits"][split] = "failed"
            state["failure"] = {"split": split, "returncode": result.returncode, "log": str(log_path)}
            write_state(state_path, state)
            raise RuntimeError(f"oracle collection failed for {split}; see {log_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["episodes"] != args.episodes or summary["strict_successes"] < args.min_successes:
            state["splits"][split] = "gate_failed"
            state["failure"] = {"split": split, "summary": summary}
            write_state(state_path, state)
            raise RuntimeError(f"oracle gate failed for {split}: {summary['strict_successes']}/{summary['episodes']}")
        motion_dir = stage_root / summary["env_id"] / "motionplanning"
        if not list(motion_dir.glob("*.h5")) or not list(motion_dir.glob("*.mp4")):
            state["splits"][split] = "artifact_failed"
            state["failure"] = {"split": split, "motion_dir": str(motion_dir)}
            write_state(state_path, state)
            raise RuntimeError(f"oracle artifacts missing for {split}: {motion_dir}")

        persistent_stage = args.persistent_root / split
        persistent_stage.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage_root, persistent_stage)
        shutil.copy2(log_path, args.persistent_root / f"{split}.log")
        state["splits"][split] = "passed"
        state.setdefault("summaries", {})[split] = summary
        write_state(state_path, state)

    args.persistent_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(state_path, args.persistent_root / "pipeline_state.json")
    (args.persistent_root / "ORACLE_GATE_COMPLETE").write_text(
        "All four StackPyramid splits passed the strict oracle gate.\n", encoding="utf-8"
    )
    state["stage"] = "complete"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_state(state_path, state)
    shutil.copy2(state_path, args.persistent_root / "pipeline_state.json")


if __name__ == "__main__":
    main()
