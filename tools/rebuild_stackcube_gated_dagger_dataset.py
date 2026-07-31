#!/usr/bin/env python3
"""Rebuild an accepted StackCube DAgger dataset without truncating suffixes.

The legacy collector retained raw actions and controller labels, but its
training materialization rounded successful expert suffixes down to a
10-action multiple.  This tool deterministically replays the stored action
sequence from its recorded reset seed and writes the complete expert suffix to
a fresh LeRobot dataset.  It never modifies the source archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RLINF_ROOT = ROOT / "RLinf"
sys.path[:0] = [str(ROOT), str(RLINF_ROOT)]

from rlinf.envs.maniskill.stack_cube_variants import STACK_CUBE_TASK, reset_metadata  # noqa: E402
from toolkits.lerobot.collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _build_frames,
    _create_dataset,
    _extract_record,
    _select_camera,
)
from toolkits.lerobot.collect_maniskill_plug_lerobot_joint import write_episode_video_durably  # noqa: E402
from tools.collect_stackcube_gated_dagger import (  # noqa: E402
    CHUNK_LABEL_HORIZON,
    _bool,
    _build_env,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_complete_successful_suffixes(
    episodes: list[dict[str, Any]], training_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve legacy training rows to their full successful expert suffixes."""
    by_raw_episode = {int(row["episode_index"]): row for row in episodes}
    selected: list[dict[str, Any]] = []
    for training_row in training_rows:
        raw_index = int(training_row["raw_episode_index"])
        row = by_raw_episode[raw_index]
        start = row.get("expert_start_step")
        action_count = int(row.get("expert_action_steps", 0))
        if not bool(row.get("success")) or start is None or action_count < CHUNK_LABEL_HORIZON:
            raise ValueError(
                f"legacy training row {raw_index} is not a successful full expert suffix"
            )
        selected.append(row)
    if not selected:
        raise ValueError("source contains no accepted successful expert suffixes")
    return selected


def _raw_action_paths(source_dir: Path, row: dict[str, Any]) -> tuple[Path, Path]:
    stem = source_dir / "raw_archive" / "actions" / (
        f"episode_{int(row['episode_index']):06d}_seed_{int(row['seed']):06d}"
    )
    return Path(f"{stem}.npy"), Path(f"{stem}.sources.json")


def _replay_full_episode(env: Any, row: dict[str, Any], actions: np.ndarray) -> tuple[list[Any], bool]:
    raw_obs, _info = env.reset(seed=int(row["seed"]))
    reset_metadata(env, split=str(row["split"]))
    records = [_extract_record(raw_obs)]
    success = False
    terminated = truncated = False
    for action in actions:
        raw_obs, _reward, terminated, truncated, info = env.step(
            torch.as_tensor(action, device=env.unwrapped.device).unsqueeze(0)
        )
        records.append(_extract_record(raw_obs))
        success = _bool(info.get("success", False))
        if success or _bool(terminated) or _bool(truncated):
            break
    if len(records) != len(actions) + 1:
        raise RuntimeError(
            f"replay ended after {len(records) - 1} actions but raw archive has {len(actions)}"
        )
    return records, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() or args.repo_id.exists():
        raise FileExistsError("output-dir and repo-id must be fresh paths")
    episode_rows = _read_jsonl(args.source_dir / "episodes.jsonl")
    legacy_training_rows = _read_jsonl(args.source_dir / "training_episodes.jsonl")
    selected = select_complete_successful_suffixes(episode_rows, legacy_training_rows)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    envs = {split: _build_env(100, task="stack", split=split) for split in ("id", "ood")}
    dataset = None
    rebuilt_rows: list[dict[str, Any]] = []
    try:
        for dataset_episode_index, row in enumerate(selected):
            action_path, source_path = _raw_action_paths(args.source_dir, row)
            actions = np.load(action_path).astype(np.float32, copy=False)
            sources = json.loads(source_path.read_text(encoding="utf-8"))
            if len(actions) != len(sources) or len(actions) != int(row["steps"]):
                raise ValueError(f"raw sidecar length mismatch for episode {row['episode_index']}")
            start = int(row["expert_start_step"])
            suffix_actions = actions[start:]
            if len(suffix_actions) != int(row["expert_action_steps"]):
                raise ValueError(f"expert suffix length mismatch for episode {row['episode_index']}")
            if len(suffix_actions) < CHUNK_LABEL_HORIZON or any(
                source != "expert" for source in sources[start:]
            ):
                raise ValueError(f"expert latch is not a complete terminal suffix for {row['episode_index']}")

            records, replay_success = _replay_full_episode(envs[str(row["split"])], row, actions)
            if replay_success != bool(row["success"]):
                raise RuntimeError(f"deterministic replay success mismatch for episode {row['episode_index']}")
            suffix_records = records[start:]
            label_frames = _build_frames(
                records=suffix_records,
                actions=list(suffix_actions),
                task=STACK_CUBE_TASK,
                main_camera=_select_camera(records[0].obs, "", MAIN_CAMERA_CANDIDATES, "main"),
                wrist_camera=_select_camera(records[0].obs, "", WRIST_CAMERA_CANDIDATES, "wrist"),
            )
            if len(label_frames) != len(suffix_actions):
                raise RuntimeError("LeRobot materialization lost an action frame")
            if dataset is None:
                dataset = _create_dataset(
                    repo_id=str(args.repo_id),
                    image_shape=tuple(label_frames[0]["image"].shape),
                    wrist_image_shape=tuple(label_frames[0]["wrist_image"].shape),
                    fps=10,
                    image_writer_threads=4,
                    image_writer_processes=0,
                )
            for frame in label_frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            video_path = write_episode_video_durably(
                label_frames,
                video_dir=args.output_dir / "replayed_suffix_videos",
                episode_index=dataset_episode_index,
                seed=int(row["seed"]),
                fps=10,
            )
            rebuilt_rows.append(
                {
                    "dataset_episode_index": dataset_episode_index,
                    "raw_episode_index": int(row["episode_index"]),
                    "seed": int(row["seed"]),
                    "split": row["split"],
                    "expert_start_step": start,
                    "full_expert_action_steps": len(suffix_actions),
                    "valid_10_step_anchors": len(suffix_actions) - CHUNK_LABEL_HORIZON + 1,
                    "raw_actions_sha256": _sha256(action_path),
                    "source_labels_sha256": _sha256(source_path),
                    "replay_success": replay_success,
                    "video_path": str(video_path),
                }
            )
            _write_jsonl(args.output_dir / "training_episodes.jsonl", rebuilt_rows)
    finally:
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        for env in envs.values():
            env.close()

    total_actions = sum(row["full_expert_action_steps"] for row in rebuilt_rows)
    total_anchors = sum(row["valid_10_step_anchors"] for row in rebuilt_rows)
    _write_json(
        args.output_dir / "rebuild_summary.json",
        {
            "source_dir": str(args.source_dir),
            "dataset": str(args.repo_id),
            "accepted_successful_expert_trajectories": len(rebuilt_rows),
            "full_expert_action_steps": total_actions,
            "valid_10_step_anchors": total_anchors,
            "terminal_actions_preserved": True,
            "source_episodes_sha256": _sha256(args.source_dir / "episodes.jsonl"),
            "source_training_rows_sha256": _sha256(
                args.source_dir / "training_episodes.jsonl"
            ),
        },
    )


if __name__ == "__main__":
    main()
