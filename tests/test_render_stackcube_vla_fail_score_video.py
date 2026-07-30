"""Small contract tests for VLA-FAIL diagnostic-video metadata handling."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "render_stackcube_vla_fail_score_video.py"
SPEC = importlib.util.spec_from_file_location("render_stackcube_vla_fail_score_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_score_trace_scales_first_and_last_decision_to_panel_bounds() -> None:
    points = MODULE._scale_points([0.0, 5.0], threshold=2.5, width=101, height=51)

    assert points[0][0] == 0
    assert points[-1][0] == 100
    assert 0 <= points[0][1] <= 50
    assert 0 <= points[-1][1] <= 50
    assert points[-1][1] < points[0][1]


def test_episode_lookup_uses_seed_and_requires_calibrated_thresholds(tmp_path: Path) -> None:
    payload = {
        "format": "stackcube_vla_fail_rollout_v1",
        "thresholds": {"llmd_threshold": 1.0, "acc_threshold": 2.0},
        "episodes": [{"seed": 42, "episode_index": 0, "timeline": []}],
    }
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    episode, thresholds = MODULE._episode_by_seed(path, 42)

    assert episode["seed"] == 42
    assert thresholds["llmd_threshold"] == 1.0
