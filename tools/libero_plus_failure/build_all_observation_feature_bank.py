#!/usr/bin/env python3
"""Build one deterministic, resume-safe shard of the full LIBERO-10 ID bank.

Every source observation is retained.  The action indices/mask record the
official fixed-horizon padding convention for terminal observations; detector
features themselves are always extracted from the current observation only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from libero_plus_failure.full_reference_bank import (  # noqa: E402
    ACTION_HORIZON,
    BRIDGE_SHAPE,
    FINAL_SHAPE,
    FORMAT,
    complete_episode_shard,
    episode_output_dir,
    episode_paths,
    sha256,
)

_LIGHT_PATH = Path(__file__).with_name("build_expert_feature_bank.py")
_SPEC = importlib.util.spec_from_file_location("libero_light_reference", _LIGHT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_LIGHT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LIGHT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def all_libero10_episodes(meta_root: Path) -> list[dict[str, Any]]:
    tasks = {str(row["task"]): int(row["task_index"]) for row in read_jsonl(meta_root / "tasks.jsonl")}
    result = []
    for row in read_jsonl(meta_root / "episodes.jsonl"):
        task = str(row["tasks"][0])
        task_index = tasks[task]
        if task_index < 10:
            result.append({
                "episode_index": int(row["episode_index"]), "task_index": task_index,
                "task": task, "length": int(row["length"]),
            })
    result.sort(key=lambda value: int(value["episode_index"]))
    if not result or {int(value["task_index"]) for value in result} != set(range(10)):
        raise ValueError("official metadata does not contain all LIBERO-10 tasks")
    return result


def padded_action_record(*, episode: dict[str, Any], frame_id: int) -> dict[str, Any]:
    length = int(episode["length"])
    if not 0 <= frame_id < length:
        raise ValueError("frame id is outside its episode")
    raw_indices = [frame_id + offset for offset in range(ACTION_HORIZON)]
    return {
        "task_index": int(episode["task_index"]),
        "task": str(episode["task"]),
        "episode_index": int(episode["episode_index"]),
        "frame_id": frame_id,
        "episode_length": length,
        "action_indices": [min(length - 1, value) for value in raw_indices],
        "action_is_pad": [value >= length for value in raw_indices],
        "tail_padding_count": sum(value >= length for value in raw_indices),
        "success_source": "official_expert_demo",
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_npz_atomic(path: Path, *, bridge: list[np.ndarray], final: list[np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, bridge=np.stack(bridge).astype(np.float32), action_expert_final=np.stack(final).astype(np.float32))
    temporary.replace(path)


def build_episode(*, client: Any, dataset_root: Path, output_root: Path, worker_index: int,
                  worker_count: int, episode: dict[str, Any], checkpoint_sha256: str, probe_sha256: str) -> bool:
    episode_id = int(episode["episode_index"])
    if episode_id % worker_count != worker_index:
        return False
    feature_path, metadata_path = episode_paths(output_root, worker_index, episode_id)
    if complete_episode_shard(feature_path, metadata_path):
        return False
    if feature_path.exists() or metadata_path.exists():
        raise RuntimeError("incomplete shard exists; refuse to overwrite %s" % feature_path)
    source = _LIGHT.ensure_download(dataset_root, episode_id)
    rows = pq.read_table(source).to_pylist()
    if len(rows) != int(episode["length"]):
        raise ValueError("metadata/parquet length mismatch for episode %d" % episode_id)
    bridge: list[np.ndarray] = []
    final: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for frame_id, row in enumerate(rows):
        observation = _LIGHT.make_observation(row, str(episode["task"]))
        observation["failure_probe/feature_only"] = True
        response = client.infer(observation)
        if "actions" in response:
            raise RuntimeError("feature-only RPC unexpectedly sampled an action chunk")
        features = response.get("failure_features", {})
        current_bridge = np.asarray(features.get("bridge"), dtype=np.float32)
        current_final = np.asarray(features.get("action_expert_final"), dtype=np.float32)
        if current_bridge.shape != BRIDGE_SHAPE or current_final.shape != FINAL_SHAPE:
            raise ValueError("invalid feature shape at episode %d frame %d" % (episode_id, frame_id))
        bridge.append(current_bridge)
        final.append(current_final)
        records.append(padded_action_record(episode=episode, frame_id=frame_id))
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    _write_npz_atomic(feature_path, bridge=bridge, final=final)
    _write_json_atomic(metadata_path, {
        "format": FORMAT, "episode": episode, "frame_count": len(rows), "records": records,
        "feature_path": str(feature_path), "feature_sha256": sha256(feature_path),
        "checkpoint_sha256": checkpoint_sha256, "probe_sha256": probe_sha256,
    })
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-source", type=Path, default=Path(__file__).with_name("serve_pi05_internal_features.py"))
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must lie in [0, worker-count)")
    episodes = all_libero10_episodes(args.dataset_root / "meta")
    root_manifest = args.output_root / "reference_bank_request.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    request = {
        "format": FORMAT, "dataset_root": str(args.dataset_root), "episodes": len(episodes),
        "expected_observations": sum(int(item["length"]) for item in episodes), "action_horizon": ACTION_HORIZON,
        "flow_timestep": 0.0, "feature_only": True, "checkpoint_sha256": sha256(args.checkpoint),
        "probe_sha256": sha256(args.probe_source), "worker_count": args.worker_count,
    }
    # Worker zero owns creation of the immutable request.  Other workers wait
    # instead of racing on one shared *.partial file.
    if args.worker_index == 0 and not root_manifest.exists():
        _write_json_atomic(root_manifest, request)
    if args.worker_index != 0:
        for _ in range(60):
            if root_manifest.exists():
                break
            time.sleep(1)
        else:
            raise TimeoutError("worker zero did not create the full-bank request manifest")
    if json.loads(root_manifest.read_text(encoding="utf-8")) != request:
        raise ValueError("output root belongs to a different immutable full-bank request")
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    client = WebsocketClientPolicy(args.host, args.port)
    completed = 0
    for episode in episodes:
        if build_episode(client=client, dataset_root=args.dataset_root, output_root=args.output_root,
                         worker_index=args.worker_index, worker_count=args.worker_count, episode=episode,
                         checkpoint_sha256=request["checkpoint_sha256"], probe_sha256=request["probe_sha256"]):
            completed += 1
            print("worker %d cached %d episodes" % (args.worker_index, completed), flush=True)
    _write_json_atomic(episode_output_dir(args.output_root, args.worker_index) / "worker_complete.json", {
        "worker_index": args.worker_index, "worker_count": args.worker_count, "completed_new": completed,
        "assigned_episode_ids": [int(item["episode_index"]) for item in episodes if int(item["episode_index"]) % args.worker_count == args.worker_index],
    })


if __name__ == "__main__":
    main()
