#!/usr/bin/env python3
"""Collect the formal StackPyramid ID and stage-localized OOD expert data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SPLITS = ("id", "stage1_ood", "stage2_ood", "stage3_ood")
TARGETS = {"id": 128, "stage1_ood": 100, "stage2_ood": 100, "stage3_ood": 100}
SEEDS = {"id": 20000, "stage1_ood": 21000, "stage2_ood": 22000, "stage3_ood": 23000}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def summary_path(stage_root: Path, split: str, env_id: str) -> Path:
    return stage_root / env_id / "motionplanning" / ".." / f"oracle_summary_{split}.json"


def run_stage(
    *,
    split: str,
    target_successes: int,
    start_seed: int,
    attempts: int,
    local_root: Path,
    persistent_root: Path,
    repo_root: Path,
    python: str,
    gpu: str,
    state: dict,
    state_path: Path,
) -> dict:
    stage_root = local_root / split
    log_path = local_root / f"{split}.log"
    if stage_root.exists() or log_path.exists():
        raise FileExistsError(f"formal collection stage already exists: {split}")
    stage_root.mkdir(parents=True)
    state["stage"] = split
    state["splits"][split] = "running"
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(state_path, state)

    command = [
        python,
        str(repo_root / "tools/collect_stackpyramid_oracle.py"),
        "--output",
        str(stage_root),
        "--split",
        split,
        "--episodes",
        str(attempts),
        "--start-seed",
        str(start_seed),
        "--sim-backend",
        "cpu",
        "--render-backend",
        "gpu",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = str(repo_root)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)

    summary = stage_root / f"oracle_summary_{split}.json"
    if result.returncode != 0 or not summary.exists():
        state["splits"][split] = "failed"
        state["failure"] = {"split": split, "returncode": result.returncode, "log": str(log_path)}
        write_json(state_path, state)
        raise RuntimeError(f"formal collection failed for {split}; see {log_path}")

    payload = json.loads(summary.read_text(encoding="utf-8"))
    motion_dir = stage_root / payload["env_id"] / "motionplanning"
    h5_count = len(list(motion_dir.glob("*.h5")))
    video_count = len(list(motion_dir.glob("*.mp4")))
    accepted = int(payload["strict_successes"])
    if payload["episodes"] != attempts or accepted < target_successes or h5_count < 1 or video_count < attempts:
        state["splits"][split] = "artifact_failed"
        state["failure"] = {
            "split": split,
            "summary": payload,
            "h5_count": h5_count,
            "video_count": video_count,
            "target_successes": target_successes,
        }
        write_json(state_path, state)
        raise RuntimeError(f"formal collection validation failed for {split}: {accepted}/{attempts}")

    persistent_stage = persistent_root / split
    persistent_root.mkdir(parents=True, exist_ok=True)
    if persistent_stage.exists():
        raise FileExistsError(f"refusing to overwrite persistent stage: {persistent_stage}")
    shutil.copytree(stage_root, persistent_stage)
    shutil.copy2(log_path, persistent_root / log_path.name)
    result_row = {
        "split": split,
        "target_successes": target_successes,
        "raw_attempts": int(payload["episodes"]),
        "strict_successes": accepted,
        "success_rate": float(payload["success_rate"]),
        "h5_count": h5_count,
        "video_count": video_count,
        "local_root": str(stage_root),
        "persistent_root": str(persistent_stage),
        "start_seed": start_seed,
    }
    state["splits"][split] = "passed"
    state.setdefault("results", {})[split] = result_row
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(state_path, state)
    write_json(persistent_root / "collection_state.json", state)
    return result_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--attempt-margin", type=int, default=10)
    args = parser.parse_args()
    if args.attempt_margin < 0:
        raise ValueError("attempt margin must be non-negative")
    if args.local_root.exists() or args.persistent_root.exists():
        raise FileExistsError("formal collection requires new roots")

    args.local_root.mkdir(parents=True)
    state_path = args.local_root / "collection_state.json"
    state = {
        "format": "stackpyramid_formal_collection_controller_v1",
        "stage": "pending",
        "splits": {split: "pending" for split in SPLITS},
        "targets": TARGETS,
        "seed_manifest": SEEDS,
        "attempt_margin": args.attempt_margin,
        "persistent_root": str(args.persistent_root),
        "task_spec": str(args.repo_root / "configs/stackpyramid_task_spec_v1.json"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(state_path, state)

    for split in SPLITS:
        target = TARGETS[split]
        run_stage(
            split=split,
            target_successes=target,
            start_seed=SEEDS[split],
            attempts=target + args.attempt_margin,
            local_root=args.local_root,
            persistent_root=args.persistent_root,
            repo_root=args.repo_root,
            python=args.python,
            gpu=args.gpu,
            state=state,
            state_path=state_path,
        )

    args.persistent_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.repo_root / "configs/stackpyramid_task_spec_v1.json", args.persistent_root / "task_spec.json")
    shutil.copy2(state_path, args.persistent_root / "collection_state.json")
    state["stage"] = "complete"
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state["updated_at"] = state["completed_at"]
    write_json(state_path, state)
    write_json(args.persistent_root / "collection_state.json", state)
    (args.persistent_root / "FORMAL_COLLECTION_COMPLETE").write_text(
        "Formal StackPyramid ID and stage-localized OOD expert collection passed.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
