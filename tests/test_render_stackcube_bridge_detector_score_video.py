from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "render_stackcube_bridge_detector_score_video.py"
SPEC = importlib.util.spec_from_file_location("bridge_detector_score_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_loads_two_bridge_traces_and_thresholds(tmp_path: Path) -> None:
    payload = {
        "format": "stackcube_internal_detector_rollout_v1",
        "thresholds": {"detectors": {
            MODULE.BRIDGE_LLMD: {"threshold": 2.5}, MODULE.BRIDGE_KNN: {"threshold": 0.1},
        }},
        "episodes": [{"seed": 20000, "timeline": [{"env_step": 0, "scores": {MODULE.BRIDGE_LLMD: 1.0, MODULE.BRIDGE_KNN: 0.01}, "alarms": {MODULE.BRIDGE_LLMD: False, MODULE.BRIDGE_KNN: False}}]}],
    }
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    episode, thresholds = MODULE.load_episode(path, 20000)
    assert episode["episode_index"] == 0
    assert MODULE.bridge_traces(episode, thresholds) == [
        (MODULE.BRIDGE_LLMD, [1.0], 2.5), (MODULE.BRIDGE_KNN, [0.01], 0.1),
    ]
