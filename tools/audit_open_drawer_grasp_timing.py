#!/usr/bin/env python3
"""Independently audit fixed-timing OpenDrawer collection artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def audit_anchor(path: Path, *, target: int) -> dict[str, Any]:
    errors: list[str] = []
    summary_path = path / "summary.json"
    accepted_path = path / "accepted_experts.jsonl"
    if not (path / "COLLECTION_COMPLETE").is_file():
        errors.append("missing COLLECTION_COMPLETE")
    if not summary_path.is_file():
        errors.append("missing summary.json")
        return {"path": str(path), "errors": errors, "pass": False}
    summary = _read_json(summary_path)
    rows = [json.loads(line) for line in accepted_path.read_text(encoding="utf-8").splitlines() if line] if accepted_path.is_file() else []
    accepted = int(summary.get("accepted", -1))
    if accepted != len(rows):
        errors.append(f"accepted mismatch summary={accepted} manifest={len(rows)}")
    if accepted < target:
        errors.append(f"accepted below target {accepted}/{target}")
    dataset = Path(str(summary.get("dataset", path / "lerobot_dataset")))
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        errors.append("missing LeRobot meta/info.json")
    else:
        info = _read_json(info_path)
        if int(info.get("total_episodes", -1)) != accepted:
            errors.append("dataset total_episodes mismatch")
        if int(info.get("total_videos", 0)) != 0:
            errors.append("unexpected embedded video count in LeRobot dataset")

    checked = 0
    for row in rows:
        episode_dir = Path(str(row.get("accepted_dir", "")))
        for name in ("actions.npy", "states.npy", "reset_metadata.json", "task_state_timeline.json"):
            if not (episode_dir / name).is_file():
                errors.append(f"missing {episode_dir / name}")
        try:
            actions = np.load(episode_dir / "actions.npy")
            states = np.load(episode_dir / "states.npy")
            timeline = _read_json(episode_dir / "task_state_timeline.json")
            timeline_rows = timeline.get("rows", [])
            if actions.ndim != 2 or actions.shape[1] != 8:
                errors.append(f"invalid action shape {actions.shape} in {episode_dir}")
            if states.ndim != 2 or states.shape != (len(actions) + 1, 9):
                errors.append(f"invalid state shape {states.shape} in {episode_dir}")
            if len(timeline_rows) != len(actions) + 1:
                errors.append(f"task timeline length mismatch in {episode_dir}")
            reset = _read_json(episode_dir / "reset_metadata.json")
            if reset.get("split") != "grasp_ood":
                errors.append(f"reset split mismatch in {episode_dir}")
            if reset.get("instruction") != "open the drawer, retrieve the blue object, and place it in the green tray":
                errors.append(f"prompt mismatch in {episode_dir}")
            checked += 1
        except Exception as exc:
            errors.append(f"artifact read failed in {episode_dir}: {exc!r}")
    return {
        "path": str(path),
        "scheduled_takeover_step": summary.get("scheduled_takeover_step"),
        "accepted": accepted,
        "raw_attempts": summary.get("raw_attempts"),
        "expert_actions": summary.get("expert_actions"),
        "episodes_checked": checked,
        "errors": errors,
        "pass": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--anchors", type=int, nargs="+", required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args()
    results = [audit_anchor(args.root / f"anchor_{step}", target=args.target) for step in args.anchors]
    payload = {"format": "open_drawer_grasp_timing_audit_v1", "target": args.target, "anchors": results, "pass": all(row["pass"] for row in results)}
    out = args.root / "audit.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if payload["pass"]:
        (args.root / "AUDIT_PASS").write_text("independent artifact audit passed\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
    else:
        (args.root / "AUDIT_FAILED").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
