from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "render_stackcube_multilayer_llmd_score_video.py"
SPEC = importlib.util.spec_from_file_location("multilayer_score_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return {
        "format": "stackcube_multilayer_llmd_rollout_v1",
        "thresholds": {"layers": {"vlm_bridge_final_mean": {"threshold": 2.5}}},
        "episodes": [
            {
                "seed": 20000,
                "timeline": [
                    {
                        "env_step": 0,
                        "scores": {"vlm_bridge_final_mean": 1.0},
                        "alarms": {"vlm_bridge_final_mean": False},
                    },
                    {
                        "env_step": 5,
                        "scores": {"vlm_bridge_final_mean": 3.0},
                        "alarms": {"vlm_bridge_final_mean": True},
                    },
                ],
            }
        ],
    }


def test_load_episode_and_build_trace_specs(tmp_path: Path) -> None:
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    episode, thresholds = MODULE.load_episode(path, 20000)
    assert episode["episode_index"] == 0
    assert MODULE.trace_specs(episode, thresholds) == [
        ("vlm_bridge_final_mean", "VLM-to-Action bridge (final)", [1.0, 3.0], 2.5)
    ]


def test_trace_specs_rejects_missing_threshold() -> None:
    episode = _payload()["episodes"][0]
    with pytest.raises(ValueError, match="missing thresholds"):
        MODULE.trace_specs(episode, {})
