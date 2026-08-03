"""Shared, immutable-on-completion helpers for a full LIBERO-10 ID bank."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_HORIZON = 10
BRIDGE_SHAPE = (1, 2048)
FINAL_SHAPE = (10, 1024)
FORMAT = "libero10_all_observation_feature_bank_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def episode_output_dir(root: Path, worker_index: int) -> Path:
    return root / "shards" / ("worker_%02d" % worker_index)


def episode_paths(root: Path, worker_index: int, episode_index: int) -> tuple[Path, Path]:
    directory = episode_output_dir(root, worker_index)
    stem = "episode_%06d" % episode_index
    return directory / (stem + ".npz"), directory / (stem + ".json")


def shard_paths(root: Path) -> list[Path]:
    return sorted(root.glob("shards/worker_*/episode_*.npz"))


def read_episode_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def complete_episode_shard(feature_path: Path, metadata_path: Path) -> bool:
    if not feature_path.is_file() or feature_path.stat().st_size == 0 or not metadata_path.is_file():
        return False
    try:
        metadata = read_episode_metadata(metadata_path)
        with np.load(feature_path, allow_pickle=False) as payload:
            bridge = payload["bridge"]
            final = payload["action_expert_final"]
        frames = int(metadata["frame_count"])
        return (
            metadata.get("format") == FORMAT
            and bridge.shape == (frames,) + BRIDGE_SHAPE
            and final.shape == (frames,) + FINAL_SHAPE
            and np.isfinite(bridge).all()
            and np.isfinite(final).all()
            and len(metadata.get("records", [])) == frames
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def all_records(metadata: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in metadata:
        records.extend(item["records"])
    return records


def validate_record_sequence(records: list[dict[str, Any]], *, expected_frames: int | None = None) -> dict[str, Any]:
    seen: set[tuple[int, int]] = set()
    task_ids: set[int] = set()
    terminal_tails = 0
    for record in records:
        episode_id = int(record["episode_index"])
        frame_id = int(record["frame_id"])
        key = (episode_id, frame_id)
        if key in seen:
            raise ValueError("duplicate all-observation record %r" % (key,))
        seen.add(key)
        task_ids.add(int(record["task_index"]))
        action_indices = [int(value) for value in record["action_indices"]]
        action_is_pad = [bool(value) for value in record["action_is_pad"]]
        if len(action_indices) != ACTION_HORIZON or len(action_is_pad) != ACTION_HORIZON:
            raise ValueError("record does not carry a full native action horizon")
        if int(record["tail_padding_count"]) != sum(action_is_pad):
            raise ValueError("tail padding count disagrees with action mask")
        if action_indices != sorted(action_indices):
            raise ValueError("clamped action indices must be monotone")
        terminal_tails += int(bool(record["tail_padding_count"]))
    if expected_frames is not None and len(records) != expected_frames:
        raise ValueError("bank frame count %d != expected %d" % (len(records), expected_frames))
    if task_ids != set(range(10)):
        raise ValueError("bank does not cover exactly LIBERO-10 task ids 0..9")
    return {"records": len(records), "tasks": sorted(task_ids), "terminal_tail_records": terminal_tails}
