#!/usr/bin/env python3
"""Render score-curve PCA annotations for the fixed OpenDrawer video set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


SOURCE_ID = Path(
    "/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_id_control_retry1/videos"
)
SOURCE_OOD = Path(
    "/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_retry4/grasp_ood_smoke/videos"
)


def _video_for_index(index: int) -> Path:
    if index < 3:
        return SOURCE_ID / f"episode_{index:06d}_seed_{910000 + index}.mp4"
    return SOURCE_OOD / f"episode_{index - 3:06d}_seed_{920000 + index - 3}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--annotator", type=Path, required=True)
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    variants = [
        {
            "name": "pooled_expert",
            "summary": args.root / "rescore_pooled_retry3/summary.json",
            "asset": "expert_id",
            "timeline_dir": args.root / "annotated_timelines/pooled_expert",
            "pooled": True,
        },
        {
            "name": "pooled_policy_raw_31",
            "summary": args.root / "rescore_pooled_policy_raw_replay_31_retry2/summary.json",
            "asset": "policy_success_id",
            "timeline_dir": args.root / "annotated_timelines/pooled_policy_raw_31",
            "pooled": True,
        },
        {
            "name": "preopen_safe_31",
            "summary": args.root / "rescore_preopen_safe_policy_31_v1/summary.json",
            "asset": "policy_preopen_safe",
            "timeline_dir": args.root / "annotated_timelines/preopen_safe_31",
            "pooled": True,
        },
        {
            "name": "tokenwise_expert",
            "summary": args.root / "rescore_tokenwise_expert_id_v1/summary.json",
            "timeline_dir": args.root / "rescore_tokenwise_expert_id_v1/episodes",
            "pooled": False,
        },
        {
            "name": "tokenwise_policy_raw_replay_31",
            "summary": args.root / "rescore_tokenwise_policy_replay_31_v1/summary.json",
            "timeline_dir": args.root / "rescore_tokenwise_policy_replay_31_v1/episodes",
            "pooled": False,
        },
        {
            "name": "phase_aligned",
            "summary": args.root / "rescore_phase_aligned_policy_id_v1/summary.json",
            "timeline_dir": args.root / "rescore_phase_aligned_policy_id_v1/episodes",
            "pooled": False,
        },
    ]
    materializer = args.annotator.parent / "materialize_open_drawer_pooled_rescore_timelines.py"
    manifest: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        summary = Path(variant["summary"])
        timeline_dir = Path(variant["timeline_dir"])
        if variant["pooled"]:
            subprocess.run(
                [
                    str(args.python),
                    str(materializer),
                    "--summary",
                    str(summary),
                    "--asset",
                    str(variant["asset"]),
                    "--output-dir",
                    str(timeline_dir),
                ],
                check=True,
            )
        output_variant_dir = args.output_dir / str(variant["name"])
        output_variant_dir.mkdir(parents=True, exist_ok=True)
        for index in range(6):
            if variant["pooled"]:
                timeline = timeline_dir / f"episode_{index:06d}.json"
            else:
                timeline = timeline_dir / f"episode_{index:06d}" / "timeline.json"
            video = _video_for_index(index)
            output = output_variant_dir / f"episode_{index:06d}_score_curve.mp4"
            command = [
                "taskset",
                "-c",
                "140-159",
                str(args.python),
                str(args.annotator),
                "--video",
                str(video),
                "--timeline",
                str(timeline),
                "--thresholds-json",
                str(summary),
                "--output",
                str(output),
                "--font",
                args.font,
            ]
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            manifest.append(
                {
                    "variant": variant["name"],
                    "index": index,
                    "source_video": str(video),
                    "timeline": str(timeline),
                    "thresholds": str(summary),
                    "output": str(output),
                    "frames": result["frames"],
                    "detectors": result["detectors"],
                }
            )

    manifest_path = args.output_dir / "score_curve_video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    marker = args.output_dir / "SCORE_CURVE_VIDEOS_COMPLETE"
    marker.write_text(f"{len(manifest)} score-curve videos rendered with synchronized PCA alarm times\n")
    print(json.dumps({"videos": len(manifest), "manifest": str(manifest_path), "marker": str(marker)}, sort_keys=True))


if __name__ == "__main__":
    main()
