#!/usr/bin/env python3
"""Materialize successful Panda ID planner episodes as an X-VLA dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary_path = args.raw_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [row for row in summary.get("rows", []) if bool(row.get("success"))]
    if len(rows) < args.target_episodes:
        raise RuntimeError(
            f"only {len(rows)} strict successful episodes available; need {args.target_episodes}"
        )
    rows = rows[: args.target_episodes]
    args.output.mkdir(parents=True)
    data_dir = args.output / "data"
    video_dir = args.output / "videos"
    metadata_dir = args.output / "metadata"
    for path in (data_dir, video_dir, metadata_dir):
        path.mkdir()
    selected = []
    for new_index, row in enumerate(rows):
        source_h5 = Path(row["data_path"])
        source_video = Path(row["video_path"])
        source_metadata = Path(row["metadata_path"])
        for source in (source_h5, source_video, source_metadata):
            if not source.exists() or source.stat().st_size <= 0:
                raise RuntimeError(f"missing source evidence: {source}")
        stem = f"episode_{new_index:06d}"
        target_h5 = data_dir / f"{stem}.h5"
        target_video = video_dir / f"{stem}.mp4"
        target_metadata = metadata_dir / f"{stem}.json"
        shutil.copy2(source_h5, target_h5)
        shutil.copy2(source_video, target_video)
        metadata = json.loads(source_metadata.read_text(encoding="utf-8"))
        metadata.update(
            {
                "dataset_episode_index": new_index,
                "source_raw_root": str(args.raw_root),
                "source_raw_episode_index": row.get("episode_index"),
                "data_path": str(target_h5),
                "video_path": str(target_video),
                "metadata_path": str(target_metadata),
            }
        )
        target_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        selected.append(metadata)
    manifest = {
        "format": "xvla_panda_vegetable_basket_id_dataset_v1",
        "source_raw_root": str(args.raw_root),
        "episodes": len(selected),
        "id_only": True,
        "ood_included": False,
        "success_definition": "released object inside basket region, above target plane, static and not grasped",
        "rows": selected,
    }
    (args.output / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in selected) + "\n", encoding="utf-8"
    )
    (args.output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "DATASET_MATERIALIZED").write_text("complete\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "episodes": len(selected)}, indent=2))


if __name__ == "__main__":
    main()

