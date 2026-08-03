#!/usr/bin/env python3
"""Validate and seal a completed full LIBERO-10 feature bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from full_reference_bank import all_records, complete_episode_shard, read_episode_metadata, shard_paths, sha256, validate_record_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    request_path = args.output_root / "reference_bank_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    paths = shard_paths(args.output_root)
    if not all(complete_episode_shard(path, path.with_suffix(".json")) for path in paths):
        raise ValueError("feature bank contains an incomplete or non-finite shard")
    metadata = [read_episode_metadata(path.with_suffix(".json")) for path in paths]
    summary = validate_record_sequence(all_records(metadata), expected_frames=int(request["expected_observations"]))
    if len({int(item["episode"]["episode_index"]) for item in metadata}) != int(request["episodes"]):
        raise ValueError("missing or duplicate episode shards")
    output = {
        "format": "libero10_all_observation_feature_bank_validation_v1", **summary,
        "episodes": len(metadata), "request_sha256": sha256(request_path),
        "shards": [{"path": str(path), "sha256": sha256(path)} for path in paths],
    }
    target = args.output_root / "validation.json"
    if target.exists():
        raise FileExistsError("validation already exists")
    target.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
