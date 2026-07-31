from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("cv2")
MODULE_PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "render_score_video.py"
SPEC = importlib.util.spec_from_file_location("render_libero_plus_failure_score_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_renderer_writes_nonempty_annotated_video(tmp_path: Path) -> None:
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    source = episode_dir / "rollout.mp4"
    imageio.mimwrite(source, [np.zeros((48, 64, 3), dtype=np.uint8), np.ones((48, 64, 3), dtype=np.uint8) * 120], fps=10)
    methods = ("bridge_llmd", "bridge_deep_knn")
    scored = [{
        "episode_path": str(episode_dir), "video_path": str(source), "decision_steps": [0, 5],
        "scores": {"bridge_llmd": [0.1, 0.8], "bridge_deep_knn": [0.2, 0.1]},
        "first_alert": {"bridge_llmd": 1, "bridge_deep_knn": None},
    }]
    scored_path = tmp_path / "scored.json"
    scored_path.write_text(json.dumps(scored), encoding="utf-8")
    threshold_path = tmp_path / "thresholds.json"
    threshold_path.write_text(json.dumps({"thresholds": {"bridge_llmd": {"threshold": 0.5}, "bridge_deep_knn": {"threshold": 0.5}}}), encoding="utf-8")
    output = tmp_path / "annotated.mp4"
    MODULE.render(scored_path, threshold_path, episode_dir, output, methods)
    assert output.is_file() and output.stat().st_size > 0
