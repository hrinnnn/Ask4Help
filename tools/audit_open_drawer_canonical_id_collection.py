#!/usr/bin/env python3
"""Audit the canonical OpenDrawer ID collection and merged ID dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RLinf"))

from rlinf.envs.maniskill.open_drawer_retrieve_place_spec import (  # noqa: E402
    TASK_INSTRUCTION,
    validate_reset_metadata,
)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_collection(root: Path, expected: int) -> dict:
    rows = jsonl(root / "episodes.jsonl")
    errors: list[str] = []
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != expected:
        errors.append(f"accepted_episodes={len(rows)} expected={expected}")
    if len(set(seeds)) != len(seeds):
        errors.append("duplicate_seed")
    videos = sorted((root / "videos").glob("*.mp4"))
    if len(videos) != expected:
        errors.append(f"videos={len(videos)} expected={expected}")
    raw_attempts = sorted((root / "raw_attempts").glob("attempt_*"))
    if len(raw_attempts) < len(rows):
        errors.append("raw_attempts_fewer_than_accepted")
    episode_rows = []
    for row in rows:
        episode_dir = root / "episodes" / f"episode_{int(row['episode_index']):06d}"
        action_path = episode_dir / "actions.npy"
        state_path = episode_dir / "states.npy"
        metadata_path = episode_dir / "reset_metadata.json"
        stages_path = episode_dir / "oracle_stages.json"
        if not all(path.is_file() for path in (action_path, state_path, metadata_path, stages_path)):
            errors.append(f"missing_evidence:{row.get('episode_index')}")
            continue
        actions = np.load(action_path)
        states = np.load(state_path)
        metadata = json.loads(metadata_path.read_text())
        stages = json.loads(stages_path.read_text())
        if actions.ndim != 2 or actions.shape[1] != 8:
            errors.append(f"action_shape:{row.get('episode_index')}:{actions.shape}")
        if states.shape != (len(actions) + 1, 9):
            errors.append(f"state_shape:{row.get('episode_index')}:{states.shape}")
        if len(actions) >= 400:
            errors.append(f"truncated_or_max_length:{row.get('episode_index')}")
        if metadata.get("instruction") != TASK_INSTRUCTION:
            errors.append(f"prompt_mismatch:{row.get('episode_index')}")
        errors.extend(f"episode_{row.get('episode_index')}:{error}" for error in validate_reset_metadata(metadata, split="id"))
        if not stages.get("success", False):
            errors.append(f"oracle_stage_success_false:{row.get('episode_index')}")
        episode_rows.append({"episode": int(row["episode_index"]), "actions": int(len(actions)), "states": int(len(states))})
    return {
        "episodes": len(rows),
        "videos": len(videos),
        "raw_attempts": len(raw_attempts),
        "seed_start": min(seeds) if seeds else None,
        "seed_end": max(seeds) if seeds else None,
        "total_actions": sum(item["actions"] for item in episode_rows),
        "tail_anchors": 9 * len(episode_rows),
        "action_horizon": 10,
        "execute_horizon": 5,
        "max_episode_steps": 400,
        "errors": sorted(set(errors)),
        "pass": not errors,
        "episode_rows": episode_rows,
    }


def dataset_episode_count(path: Path) -> int:
    return len(jsonl(path / "meta" / "episodes.jsonl"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--merged-dataset", type=Path, required=True)
    parser.add_argument("--expected-new", type=int, default=256)
    parser.add_argument("--expected-merged", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collection = audit_collection(args.collection, args.expected_new)
    merged_count = dataset_episode_count(args.merged_dataset) if (args.merged_dataset / "meta/episodes.jsonl").is_file() else 0
    errors = list(collection["errors"])
    if merged_count != args.expected_merged:
        errors.append(f"merged_episodes={merged_count} expected={args.expected_merged}")
    report = {"format": "open_drawer_canonical_id_collection_audit_v1", "task_instruction": TASK_INSTRUCTION, "collection": collection, "merged_dataset_episodes": merged_count, "errors": sorted(set(errors)), "pass": not errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
