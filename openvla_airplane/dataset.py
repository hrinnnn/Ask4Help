"""LeRobot-to-OpenVLA bridge for the controlled airplane dataset.

The adapter keeps the official OpenVLA prompt and action-token contract while
reading the existing LeRobot parquet episodes. Only the base camera is passed
to OpenVLA; wrist images remain in the source archive for auditability.
"""

from __future__ import annotations

import io
import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

AIRPLANE_INSTRUCTION = "pick up the toy airplane and move it to the green goal"
IGNORE_INDEX = -100


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, dict):
        for key in ("bytes", "data", "encoded"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _read_jsonl(path: Path) -> List[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class FrameRef:
    path: Path
    row: int
    episode_index: int
    frame_index: int


def index_lerobot_episodes(data_dir: Path) -> List[FrameRef]:
    meta_dir = data_dir / "meta"
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    refs: List[FrameRef] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk = episode_index // 1000
        path = data_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        length = int(episode["length"])
        refs.extend(FrameRef(path, row, episode_index, row) for row in range(length))
    return refs


def validate_lerobot_dataset(data_dir: Path, expected_episodes: int = 98, expected_frames: int = 9109) -> dict:
    info = json.loads((data_dir / "meta" / "info.json").read_text())
    episodes = _read_jsonl(data_dir / "meta" / "episodes.jsonl")
    actual_frames = sum(int(item["length"]) for item in episodes)
    if int(info["total_episodes"]) != expected_episodes or len(episodes) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, got {len(episodes)}")
    if actual_frames != expected_frames or int(info["total_frames"]) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {actual_frames}")
    if info["features"]["actions"]["shape"] != [8]:
        raise ValueError("The airplane experiment requires an 8D action space")
    if info["features"]["image"]["shape"] != [384, 384, 3]:
        raise ValueError("Unexpected base camera shape")
    return {
        "episodes": expected_episodes,
        "frames": expected_frames,
        "action_dim": 8,
        "camera": "image",
        "instruction": AIRPLANE_INSTRUCTION,
    }


def compute_action_stats(data_dir: Path) -> dict:
    """Compute the official q01/q99 action bounds from ID expert actions only."""
    import pyarrow.parquet as pq

    paths = dict.fromkeys(ref.path for ref in index_lerobot_episodes(data_dir))
    episode_actions = []
    for path in paths:
        table = pq.read_table(path, columns=["actions"])
        episode_actions.append(np.asarray(table["actions"].to_pylist(), dtype=np.float32))
    array = np.concatenate(episode_actions, axis=0)
    q01, q99 = np.quantile(array, [0.01, 0.99], axis=0)
    return {
        "q01": q01.astype(np.float32).tolist(),
        "q99": q99.astype(np.float32).tolist(),
        "num_transitions": int(array.shape[0]),
        "num_trajectories": len(_read_jsonl(data_dir / "meta" / "episodes.jsonl")),
        "action_dim": int(array.shape[1]),
        "source": "98 ID expert trajectories only",
    }


def normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    low = np.asarray(stats["q01"], dtype=np.float32)
    high = np.asarray(stats["q99"], dtype=np.float32)
    scale = np.where(high > low, high - low, 1.0)
    return np.clip(2.0 * (action - low) / scale - 1.0, -1.0, 1.0)


class AirplaneDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        validate_lerobot_dataset(self.data_dir)
        self.refs = index_lerobot_episodes(self.data_dir)

    @staticmethod
    @lru_cache(maxsize=98)
    def _read_episode(path: Path):
        import pyarrow.parquet as pq

        return pq.read_table(path, columns=["image", "actions"])

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict:
        ref = self.refs[index]
        table = self._read_episode(ref.path)
        return {
            "image": _decode_image(table["image"][ref.row].as_py()),
            "action": np.asarray(table["actions"][ref.row].as_py(), dtype=np.float32),
            "episode_index": ref.episode_index,
            "frame_index": ref.frame_index,
        }


def build_example(
    sample: dict,
    tokenizer,
    action_tokenizer,
    image_transform,
    action_stats: dict,
    prompt_builder_fn,
) -> dict:
    normalized = normalize_action(sample["action"], action_stats)
    action_text = action_tokenizer(normalized)
    tokenized_action = tokenizer(action_text, add_special_tokens=False).input_ids
    if len(tokenized_action) != normalized.shape[0]:
        raise ValueError(f"8D action did not round-trip to 8 tokens: {tokenized_action}")

    prompt_builder = prompt_builder_fn("openvla")
    prompt_builder.add_turn("human", f"What action should the robot take to {AIRPLANE_INSTRUCTION}?")
    prompt_builder.add_turn("gpt", action_text)
    input_ids = tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
    labels = list(input_ids)
    labels[: -(len(tokenized_action) + 1)] = [IGNORE_INDEX] * (len(labels) - len(tokenized_action) - 1)
    return {
        "pixel_values": image_transform(sample["image"]),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "action": sample["action"],
        "normalized_action": normalized,
        "episode_index": sample["episode_index"],
        "frame_index": sample["frame_index"],
    }


def action_token_round_trip(action: Sequence[float], tokenizer, action_tokenizer) -> dict:
    original = np.asarray(action, dtype=np.float32)
    text = action_tokenizer(original)
    ids = tokenizer(text, add_special_tokens=False).input_ids
    decoded = action_tokenizer.decode_token_ids_to_actions(np.asarray(ids, dtype=np.int64))
    if len(ids) != len(original):
        raise ValueError(f"Expected {len(original)} tokens, got {len(ids)}")
    return {
        "action_dim": int(len(original)),
        "token_ids": [int(value) for value in ids],
        "decoded": np.asarray(decoded, dtype=np.float32).tolist(),
        "max_abs_error": float(np.max(np.abs(decoded - original))),
    }
