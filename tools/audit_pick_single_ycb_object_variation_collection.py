#!/usr/bin/env python3
"""Audit ID collection evidence and temporal-anchor accounting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.collection_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]
    accepted = [row for row in rows if row.get("accepted") is True]
    episodes_meta = args.dataset_dir / "meta" / "episodes.jsonl"
    meta_rows = [json.loads(line) for line in episodes_meta.read_text().splitlines() if line.strip()]
    lengths = [int(row["length"]) for row in meta_rows]
    total_anchors = sum(lengths)
    tail_anchor_count = 9 * len(lengths)
    valid_timesteps = sum(max(0, 10 * length - 45) for length in lengths)
    videos = list((args.collection_dir / "videos").glob("*.mp4"))
    model_ids = {
        row["reset_metadata"]["object_model_id"]
        for row in accepted
        if isinstance(row.get("reset_metadata"), dict)
    }
    report = {
        "format": "pick_single_ycb_object_variation_id_data_audit_v1",
        "collection_dir": str(args.collection_dir),
        "dataset_dir": str(args.dataset_dir),
        "raw_attempts": len(rows),
        "accepted_episodes": len(accepted),
        "dataset_meta_episodes": len(meta_rows),
        "video_count": len(videos),
        "object_model_ids": sorted(model_ids),
        "expected_id_model_id": "005_tomato_soup_can",
        "total_observations_as_anchors": total_anchors,
        "tail_anchor_count": tail_anchor_count,
        "action_horizon": 10,
        "valid_action_timesteps": valid_timesteps,
        "valid_action_ratio": valid_timesteps / max(1, total_anchors * 10),
        "final_anchor_is_retained": True,
        "final_anchor_valid_timesteps": 1,
        "temporal_mask_contract": "repeat final real action only for shape; action_valid_mask excludes padded positions from loss",
        "passed": len(accepted) == 128 and len(meta_rows) == 128 and len(videos) == 128 and model_ids == {"005_tomato_soup_can"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker = args.output.parent / ("ID_DATA_AUDIT_PASSED" if report["passed"] else "ID_DATA_AUDIT_FAILED")
    marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

