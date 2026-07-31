#!/usr/bin/env python3
"""Selectively download and cache the light 1,000-anchor LIBERO-10 bank."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from libero_plus_failure_protocol import evenly_spaced_indices, select_expert_anchors  # noqa: E402


HF_MIRROR = "https://hf-mirror.com/datasets/physical-intelligence/libero/resolve/main/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / ("chunk-%03d" % (episode_index // 1000)) / ("episode_%06d.parquet" % episode_index)


def ensure_download(dataset_root: Path, episode_index: int) -> Path:
    destination = episode_path(dataset_root, episode_index)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative = "data/chunk-%03d/episode_%06d.parquet" % (episode_index // 1000, episode_index)
    partial = destination.with_suffix(".partial")
    urllib.request.urlretrieve(HF_MIRROR + relative + "?download=true", partial)
    partial.replace(destination)
    return destination


def decode_image(value: dict[str, Any]) -> np.ndarray:
    if value.get("bytes") is None:
        raise ValueError("official parquet image is not embedded bytes")
    with Image.open(io.BytesIO(value["bytes"])) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def make_observation(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    base = image_tools.convert_to_uint8(image_tools.resize_with_pad(decode_image(row["image"]), 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(decode_image(row["wrist_image"]), 224, 224))
    return {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": np.asarray(row["state"], dtype=np.float32),
        "prompt": prompt,
    }


def select_records(meta_root: Path, *, seed: int) -> list[dict[str, Any]]:
    tasks = read_jsonl(meta_root / "tasks.jsonl")
    task_to_index = {str(row["task"]): int(row["task_index"]) for row in tasks}
    episodes = read_jsonl(meta_root / "episodes.jsonl")
    experts = []
    for episode in episodes:
        task = str(episode["tasks"][0])
        task_index = task_to_index[task]
        if task_index >= 10:
            continue
        length = int(episode["length"])
        # A training anchor requires an observation plus a full native action
        # target.  The tail cannot cross the episode boundary.
        valid = length - 10
        if valid < 10:
            continue
        experts.append(
            {
                "task_id": str(task_index),
                "episode_id": str(episode["episode_index"]),
                "anchor_ids": evenly_spaced_indices(valid, 10),
                "task": task,
                "length": length,
            }
        )
    selected = select_expert_anchors(experts, demos_per_task=10, anchors_per_demo=10, seed=seed)
    original = {str(row["episode_id"]): row for row in experts}
    for row in selected:
        source = original[str(row["episode_id"])]
        row.update({"task": source["task"], "episode_length": source["length"], "success_source": "official_expert_demo"})
    if len(selected) != 1000 or {row["task_id"] for row in selected} != {str(index) for index in range(10)}:
        raise RuntimeError("selection did not produce exactly 10 tasks x 10 demos x 10 anchors")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite " + str(args.output_dir))
    meta = args.dataset_root / "meta"
    if not (meta / "episodes.jsonl").is_file() or not (meta / "tasks.jsonl").is_file():
        raise FileNotFoundError("download official meta/episodes.jsonl and meta/tasks.jsonl first")
    selected = select_records(meta, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection_path = args.output_dir / "expert_selection_manifest.json"
    selection_path.write_text(json.dumps({"format": "libero_plus_expert_selection_v1", "seed": args.seed, "rows": selected}, indent=2) + "\n", encoding="utf-8")
    client = WebsocketClientPolicy(args.host, args.port)
    bridge, final, actions = [], [], []
    for index, selected_anchor in enumerate(selected):
        episode_index = int(selected_anchor["episode_id"])
        path = ensure_download(args.dataset_root, episode_index)
        rows = pq.read_table(path).to_pylist()
        anchor = int(selected_anchor["anchor_id"])
        if anchor + 10 > len(rows):
            raise ValueError("selected anchor has no full target: " + str(selected_anchor))
        response = client.infer(make_observation(rows[anchor], str(selected_anchor["task"])))
        features = response.get("failure_features", {})
        current_bridge = np.asarray(features["bridge"], dtype=np.float32)
        current_final = np.asarray(features["action_expert_final"], dtype=np.float32)
        current_actions = np.asarray(response["actions"], dtype=np.float32)
        if current_bridge.shape != (1, 2048) or current_final.shape != (10, 1024) or current_actions.shape != (10, 7):
            raise ValueError("instrumented pi05 response violates frozen feature/action contract")
        bridge.append(current_bridge)
        final.append(current_final)
        actions.append(current_actions)
        if (index + 1) % 25 == 0 or index + 1 == len(selected):
            print("cached %d/%d anchors" % (index + 1, len(selected)), flush=True)
    payload = {
        "format": "libero_plus_expert_feature_cache_v1",
        "selection_sha256": sha256(selection_path),
        "selected_anchors": selected,
        "features": {"bridge": torch.from_numpy(np.stack(bridge)), "action_expert_final": torch.from_numpy(np.stack(final))},
        "action_chunks": torch.from_numpy(np.stack(actions)),
        "checkpoint_protocol": "pi05_libero_internal_feature_probe_v1",
    }
    cache_path = args.output_dir / "expert_feature_cache.pt"
    torch.save(payload, cache_path)
    (args.output_dir / "feature_cache_manifest.json").write_text(
        json.dumps({"cache": str(cache_path), "cache_sha256": sha256(cache_path), "anchors": len(selected), "selection_sha256": payload["selection_sha256"]}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
