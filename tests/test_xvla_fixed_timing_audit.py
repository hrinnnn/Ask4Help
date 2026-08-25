from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "audit_xvla_fixed_timing_calibration.py"
    spec = importlib.util.spec_from_file_location("fixed_timing_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_records_unrecoverable_anchor(tmp_path: Path) -> None:
    module = _load_module()
    root = tmp_path / "calibration"
    anchor = root / "step_0"
    states = anchor / "raw_archive/task_states"
    actions = anchor / "raw_archive/actions"
    videos = anchor / "raw_archive/videos"
    states.mkdir(parents=True)
    actions.mkdir(parents=True)
    videos.mkdir(parents=True)
    rows = []
    for index, seed in enumerate((10, 11)):
        state = states / f"episode_{index:06d}_seed_{seed:06d}.npy"
        np.save(state, np.zeros((3, 2), dtype=np.float32))
        (actions / f"episode_{index:06d}_seed_{seed:06d}.npy").write_bytes(b"x")
        video = videos / f"episode_{index:06d}_seed_{seed:06d}.mp4"
        video.write_bytes(b"x")
        rows.append({
            "seed": seed,
            "attempt_index": index,
            "success": index == 0,
            "steps": 2,
            "task_states": str(state),
            "video": str(video),
        })
    (anchor / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (anchor / "summary.json").write_text('{"raw_total": 2, "accepted_total": 1}\n', encoding="utf-8")
    result = module.audit(root, [0], seeds=[10, 11], minimum_recoverability=0.9)
    assert result["status"] == "FAIL"
    assert any("UNRECOVERABLE_REGION" in error for error in result["errors"])
