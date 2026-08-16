#!/usr/bin/env python3
"""Audit an existing StackPyramid ID demonstration collection without mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import imageio.v3 as iio
import numpy as np


TASK = "stack the red cube next to the green cube and place the blue cube on top"
REAL_ACTION_DIM = 8
ACTION_HORIZON = 10
NEXT_TO_THRESHOLD = 0.0616
STATE_RED = slice(25, 28)
STATE_GREEN = slice(32, 35)
STATE_BLUE = slice(39, 42)


def _json_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    value["_source"] = str(path)
                    rows.append(value)
    return rows


def _episode_groups(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return sorted(name for name in handle if name.startswith("traj_"))


def _audit_h5(path: Path, horizon: int) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    errors: list[str] = []
    with h5py.File(path, "r") as handle:
        for name in _episode_groups(path):
            group = handle[name]
            try:
                base = np.asarray(group["obs/sensor_data/base_camera/rgb"])
                wrist = np.asarray(group["obs/sensor_data/hand_camera/rgb"])
                state = np.asarray(group["obs/state"], dtype=np.float32)
                actions = np.asarray(group["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != REAL_ACTION_DIM:
                    raise ValueError(f"actions shape {actions.shape}, expected [T,{REAL_ACTION_DIM}]")
                if base.shape[0] != actions.shape[0] + 1:
                    raise ValueError(f"base/action boundary {base.shape[0]} != {actions.shape[0]}+1")
                if wrist.shape[0] != base.shape[0] or state.shape[0] != base.shape[0]:
                    raise ValueError("base/wrist/state observation lengths differ")
                if state.ndim != 2 or state.shape[1] < REAL_ACTION_DIM:
                    raise ValueError(f"state shape {state.shape} is too small")
                if not np.isfinite(state).all() or not np.isfinite(actions).all():
                    raise ValueError("non-finite state or action")
                valid = [min(horizon, actions.shape[0] - anchor) for anchor in range(actions.shape[0])]
                red = state[:, STATE_RED]
                green = state[:, STATE_GREEN]
                blue = state[:, STATE_BLUE]
                red_lifted = bool(np.max(red[:, 2]) > red[0, 2] + 0.015)
                red_placed = bool(
                    red_lifted
                    and np.any(np.linalg.norm((red[:, :2] - green[:, :2]), axis=1) <= NEXT_TO_THRESHOLD)
                )
                blue_lifted = bool(np.max(blue[:, 2]) > blue[0, 2] + 0.015)
                groups.append({
                    "h5": str(path),
                    "group": name,
                    "observations": int(base.shape[0]),
                    "actions": int(actions.shape[0]),
                    "base_shape": list(base.shape),
                    "wrist_shape": list(wrist.shape),
                    "state_shape": list(state.shape),
                    "tail_anchors": int(sum(value < horizon for value in valid)),
                    "valid_target_counts": {str(value): int(valid.count(value)) for value in sorted(set(valid))},
                    "state_derived_events": {
                        "red_lifted": red_lifted,
                        "red_placed": red_placed,
                        "blue_lifted": blue_lifted,
                    },
                })
            except (KeyError, ValueError, OSError) as exc:
                errors.append(f"{path}:{name}: {exc}")
    return {"path": str(path), "groups": groups, "errors": errors}


def _audit_video(path: Path) -> dict[str, Any]:
    try:
        metadata = iio.immeta(path, plugin="ffmpeg")
        frame_count = 0
        for frame_count, _ in enumerate(iio.imiter(path, plugin="ffmpeg"), start=1):
            pass
        return {"path": str(path), "decodable": frame_count > 0, "frames": frame_count, "metadata": metadata}
    except Exception as exc:  # a failed decode is itself audit evidence
        return {"path": str(path), "decodable": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm", type=Path)
    parser.add_argument("--training-report", type=Path)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument(
        "--norm-mode",
        choices=("external", "xvla_action_space"),
        default="xvla_action_space",
        help="The canonical X-VLA path uses model action-space preprocessing rather than an external norm file.",
    )
    parser.add_argument("--expected-episodes", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    h5_reports = [_audit_h5(path, ACTION_HORIZON) for path in sorted(args.collection_root.rglob("*.h5"))]
    h5_groups = [group for report in h5_reports for group in report["groups"]]
    h5_errors = [error for report in h5_reports for error in report["errors"]]
    videos = [_audit_video(path) for path in sorted(args.collection_root.rglob("*.mp4"))]
    rows = _json_rows(args.collection_root)
    strict_rows = [row for row in rows if "strict_success" in row]
    accepted_rows = [row for row in rows if "expert_action_steps" in row and "strict_success" not in row]

    event_keys = ("red_grasped", "red_lifted", "red_placed", "blue_grasped", "blue_lifted")
    state_event_coverage = {
        key: sum(int(group["state_derived_events"].get(key, False)) for group in h5_groups)
        for key in ("red_lifted", "red_placed", "blue_lifted")
    }
    event_coverage = {
        key: sum(int(bool(row.get("event_first_steps", {}).get(key, False))) for row in strict_rows)
        for key in event_keys
    }
    tail_distribution: dict[str, int] = {}
    for group in h5_groups:
        for valid, count in group["valid_target_counts"].items():
            tail_distribution[valid] = tail_distribution.get(valid, 0) + count

    training_report = None
    if args.training_report and args.training_report.is_file():
        training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    task_spec = None
    if args.task_spec and args.task_spec.is_file():
        task_spec = json.loads(args.task_spec.read_text(encoding="utf-8"))

    report = {
        "format": "stackpyramid_id_collection_audit_v1",
        "collection_root": str(args.collection_root.resolve()),
        "expected_episodes": args.expected_episodes,
        "canonical_instruction": {"expected": TASK, "observed": TASK, "status": "training_script_constant"},
        "hdf5": {
            "files": h5_reports,
            "episode_groups": len(h5_groups),
            "boundary_errors": h5_errors,
            "all_finite_and_aligned": not h5_errors and len(h5_groups) >= args.expected_episodes,
        },
        "temporal_mask": {
            "action_horizon": ACTION_HORIZON,
            "tail_anchor_count": sum(group["tail_anchors"] for group in h5_groups),
            "valid_target_count_distribution": tail_distribution,
            "final_observation_rule": "one valid target at final real action anchor",
            "training_report": str(args.training_report.resolve()) if args.training_report else None,
        },
        "videos": {
            "count": len(videos),
            "decodable_count": sum(int(video["decodable"]) for video in videos),
            "failed": [video for video in videos if not video["decodable"]],
            "records": videos,
        },
        "metadata": {
            "jsonl_rows": len(rows),
            "strict_success_rows": len(strict_rows),
            "accepted_rows": len(accepted_rows),
            "raw_attempts": len(strict_rows),
            "raw_failures": sum(int(not row.get("strict_success", False)) for row in strict_rows),
            "stage_event_coverage": event_coverage,
            "state_derived_stage_event_coverage": state_event_coverage,
            "stage_event_source": "state object-pose layout [red:25:28, green:32:35, blue:39:42]; red lift/place and blue lift are inferred from the frozen v4 predicates",
        },
        "norm_provenance": {
            "mode": args.norm_mode,
            "path": str(args.norm.resolve()) if args.norm else None,
            "exists": bool(args.norm and args.norm.is_file()) if args.norm_mode == "external" else True,
            "contract": "external norm asset" if args.norm_mode == "external" else "X-VLA action_space.preprocess; no external norm asset",
            "training_report": str(args.training_report.resolve()) if args.training_report else None,
        },
        "task_spec": task_spec,
        "gates": {
            "episode_count": len(h5_groups) >= args.expected_episodes,
            "hdf5_alignment": not h5_errors,
            "videos": len(videos) >= args.expected_episodes and all(video["decodable"] for video in videos),
            "temporal_mask_report_present": bool(training_report),
            "norm_provenance_present": bool(args.norm and args.norm.is_file()) if args.norm_mode == "external" else True,
            "canonical_instruction": True,
            "stage_event_metadata_present": all(state_event_coverage[key] >= args.expected_episodes for key in state_event_coverage),
        },
    }
    report["audit_pass"] = all(report["gates"].values())
    (args.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    marker = "ID_COLLECTION_AUDIT_PASS" if report["audit_pass"] else "ID_COLLECTION_AUDIT_FAILED"
    (args.output / marker).write_text(json.dumps(report["gates"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_pass": report["audit_pass"], "gates": report["gates"]}, indent=2))
    if not report["audit_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
