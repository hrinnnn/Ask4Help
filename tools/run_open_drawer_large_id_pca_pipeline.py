#!/usr/bin/env python3
"""Fit OpenDrawer PCA from the large ID training set and rescore fixed rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


SOURCE_ID = Path(
    "/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_id_control_retry1/videos"
)
SOURCE_OOD = Path(
    "/sdd/ask4help-open-drawer/results/open_drawer_representative_pca_v1_retry4/grasp_ood_smoke/videos"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_marker(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def _snapshot_gpu(index: int) -> dict:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    selected = None
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        if int(values[0]) == index:
            selected = {
                "index": index,
                "uuid": values[1],
                "total_mib": int(values[2]),
                "used_mib": int(values[3]),
                "free_mib": int(values[4]),
                "utilization_percent": int(values[5]),
            }
            break
    if selected is None:
        raise RuntimeError(f"GPU index {index} not found")
    app_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    selected["compute_apps"] = []
    for row in app_rows:
        if not row.strip():
            continue
        values = [value.strip() for value in row.split(",")]
        if values[0] == selected["uuid"]:
            selected["compute_apps"].append(
                {"gpu_uuid": values[0], "pid": int(values[1]), "process": values[2], "used_mib": int(values[3])}
            )
    selected["checked_at"] = _now()
    return selected


def _update_state(path: Path, **updates: object) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(updates)
    current["updated_at"] = _now()
    _write_json(path, current)


def _run(command: list[str], env: dict[str, str], state_path: Path, stage: str) -> None:
    _update_state(state_path, stage=stage, command=command)
    print(json.dumps({"stage": stage, "command": command}, sort_keys=True), flush=True)
    subprocess.run(command, check=True, env=env)


def _source_video(index: int) -> Path:
    if index < 3:
        return SOURCE_ID / f"episode_{index:06d}_seed_{910000 + index}.mp4"
    return SOURCE_OOD / f"episode_{index - 3:06d}_seed_{920000 + index - 3}.mp4"


def _write_alarm_table(summary_path: Path, output_dir: Path, asset_name: str) -> None:
    payload = json.loads(summary_path.read_text())
    thresholds = payload["thresholds"][asset_name]
    rows: list[dict[str, object]] = []
    markdown = [
        "# OpenDrawer large-ID PCA gate alarm times",
        "",
        "Scores are evaluated at decision times (every 5 low-level steps). Alarm steps are the first threshold crossing.",
        "",
        "| split | seed | drawer opened | layer | q=.80 first alarm | q=.95 first alarm |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for episode in payload["rows"]:
        first = episode["first_alarm_env_step"][asset_name]
        for detector in episode["timelines"][asset_name][0]["scores"]:
            row = {
                "split": episode["split"],
                "seed": episode["seed"],
                "first_drawer_opened_env_step": episode.get("first_drawer_opened_env_step"),
                "detector": detector,
                "q80_first_alarm_env_step": first["0.8"].get(detector),
                "q95_first_alarm_env_step": first["0.95"].get(detector),
                "q80_threshold": thresholds["0.8"].get(detector),
                "q95_threshold": thresholds["0.95"].get(detector),
            }
            rows.append(row)
            markdown.append(
                f"| {row['split']} | {row['seed']} | {row['first_drawer_opened_env_step'] if row['first_drawer_opened_env_step'] is not None else '-'} | "
                f"{detector} | {row['q80_first_alarm_env_step'] if row['q80_first_alarm_env_step'] is not None else '-'} | "
                f"{row['q95_first_alarm_env_step'] if row['q95_first_alarm_env_step'] is not None else '-'} |"
            )
    _write_json(output_dir / "gate_alarm_times.json", {"asset": asset_name, "rows": rows, "thresholds": thresholds})
    (output_dir / "gate_alarm_times.md").write_text("\n".join(markdown) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pi05-base", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()

    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    _update_state(
        state_path,
        pipeline_id="open_drawer_large_id_train512_pca_v1",
        authorized=True,
        stage="starting",
        next_stage="fit_pca_from_id_merged512",
        dataset_root=str(args.dataset_root),
        checkpoint=str(args.checkpoint),
        distribution_name="id_train_merged512",
        no_new_ood_collection=True,
        gpu_index=args.gpu_index,
    )

    snapshot = None
    while snapshot is None or snapshot["free_mib"] < 28 * 1024 or snapshot["compute_apps"] or snapshot["utilization_percent"] > 5:
        snapshot = _snapshot_gpu(args.gpu_index)
        _update_state(state_path, stage="waiting_for_gpu", next_stage="fit_pca_from_id_merged512", resource_snapshot=snapshot)
        print(json.dumps({"stage": "waiting_for_gpu", "resource_snapshot": snapshot}, sort_keys=True), flush=True)
        time.sleep(args.poll_seconds)

    env = os.environ.copy()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "ASK4HELP_RLINF_ROOT": "/data/zhaozhixuan/Ask4Help-open-drawer/RLinf",
            "PYTHONPATH": "/data/zhaozhixuan/Ask4Help-open-drawer:/data/zhaozhixuan/Ask4Help-open-drawer/RLinf:/data/zhaozhixuan",
        }
    )
    _update_state(state_path, stage="resource_gate_passed", resource_snapshot=_snapshot_gpu(args.gpu_index))

    feature_dir = root / "feature_cache_build"
    feature_marker = root / "FEATURE_CACHE_BUILD_COMPLETE"
    if not feature_marker.exists():
        _run(
            [
                str(args.python),
                str(args.tools_dir / "build_open_drawer_representative_pca_assets.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--pi05-base",
                str(args.pi05_base),
                "--norm-stats",
                str(args.norm_stats),
                "--dataset-root",
                str(args.dataset_root),
                "--output-dir",
                str(feature_dir),
                "--max-observations",
                "0",
            ],
            env,
            state_path,
            "fit_feature_cache_from_id_merged512",
        )
        _write_marker(feature_marker, "91533 ID training observations feature-extracted")

    asset_dir = root / "pca_asset"
    asset_marker = root / "PCA_ASSET_COMPLETE"
    if not asset_marker.exists():
        _run(
            [
                str(args.python),
                str(args.tools_dir / "rebuild_open_drawer_pca_from_feature_cache.py"),
                "--feature-cache",
                str(feature_dir / "feature_cache.pt"),
                "--episode-lengths",
                str(args.dataset_root / "meta/episodes.jsonl"),
                "--output-dir",
                str(asset_dir),
                "--distribution-name",
                "id_train_merged512",
                "--fit-episode-fraction",
                "0.8",
                "--quantiles",
                "0.80",
                "0.95",
            ],
            env,
            state_path,
            "fit_episode_separated_pca_asset",
        )
        _write_marker(asset_marker, "large ID training distribution PCA asset complete")

    asset_name = "id_train_merged512"
    rescore_dir = root / "rescore_fixed_id_grasp_smoke"
    rescore_marker = root / "RESCORE_FIXED_SMOKE_COMPLETE"
    if not rescore_marker.exists():
        _run(
            [
                str(args.python),
                str(args.tools_dir / "rescore_open_drawer_pca_assets.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--pi05-base",
                str(args.pi05_base),
                "--norm-stats",
                str(args.norm_stats),
                "--source-root",
                str(SOURCE_ID.parent),
                "--source-root",
                str(SOURCE_OOD.parent),
                "--output-dir",
                str(rescore_dir),
                "--asset",
                f"{asset_name}={asset_dir / 'representative_pca_assets.pt'}",
                "--device",
                "cuda",
            ],
            env,
            state_path,
            "rescore_fixed_id_and_grasp_ood_smoke",
        )
        _write_marker(rescore_marker, "3 fixed ID + 3 fixed Grasp-OOD rollouts rescored")

    timeline_dir = root / "annotated_timelines" / asset_name
    timeline_marker = root / "MATERIALIZED_TIMELINES_COMPLETE"
    if not timeline_marker.exists():
        _run(
            [
                str(args.python),
                str(args.tools_dir / "materialize_open_drawer_pooled_rescore_timelines.py"),
                "--summary",
                str(rescore_dir / "summary.json"),
                "--asset",
                asset_name,
                "--output-dir",
                str(timeline_dir),
            ],
            env,
            state_path,
            "materialize_gate_timelines",
        )
        _write_marker(timeline_marker, "6 gate timelines materialized")

    video_dir = root / "score_curve_videos"
    video_marker = root / "SCORE_CURVE_VIDEOS_COMPLETE"
    if not video_marker.exists():
        video_dir.mkdir(parents=True, exist_ok=True)
        annotator = args.tools_dir / "annotate_open_drawer_score_curve_video.py"
        for index in range(6):
            output = video_dir / f"episode_{index:06d}_score_curve.mp4"
            _run(
                [
                    "taskset",
                    "-c",
                    "140-159",
                    str(args.python),
                    str(annotator),
                    "--video",
                    str(_source_video(index)),
                    "--timeline",
                    str(timeline_dir / f"episode_{index:06d}.json"),
                    "--output",
                    str(output),
                ],
                os.environ.copy(),
                state_path,
                f"render_score_curve_video_{index:02d}",
            )
        _write_marker(video_marker, "6 large-ID PCA score-curve videos rendered")

    _write_alarm_table(rescore_dir / "summary.json", root, asset_name)
    complete = root / "LARGE_ID_PCA_TIMING_COMPLETE"
    _write_marker(complete, "large ID training distribution PCA timing pipeline complete")
    _update_state(
        state_path,
        stage="completed",
        status="diagnostic_complete",
        next_stage="user_review_large_id_pca_alarm_times",
        completion_marker=str(complete),
        score_curve_videos=str(video_dir),
        gate_alarm_table=str(root / "gate_alarm_times.md"),
    )
    print(json.dumps({"stage": "completed", "output_root": str(root), "marker": str(complete)}, sort_keys=True))


if __name__ == "__main__":
    main()
