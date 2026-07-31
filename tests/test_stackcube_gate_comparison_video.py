import importlib.util
from pathlib import Path
import sys

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "render_stackcube_gate_comparison_video.py"
SPEC = importlib.util.spec_from_file_location("stackcube_gate_video", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_overlay_uses_recorded_controller_boundary_and_preserves_image_content():
    row = {
        "method": "bridge_knn",
        "steps": 20,
        "success": True,
        "expert_start_step": 10,
        "timeline": [
            {"env_step": 0, "controller": "policy", "score": 0.1, "threshold": 0.2, "alarm": False},
            {"env_step": 10, "controller": "expert", "score": 0.3, "threshold": 0.2, "alarm": True},
        ],
    }
    raw = np.full((24, 48, 3), 17, dtype=np.uint8)
    before_gate = MODULE.build_annotated_frame(raw, row, frame_index=4, title="robot")
    after_gate = MODULE.build_annotated_frame(raw, row, frame_index=15, title="robot")

    assert before_gate.shape == (116, 48, 3)
    assert np.array_equal(before_gate[:24], raw)
    assert np.array_equal(after_gate[:24], raw)
    # The bottom bar keeps the recorded policy/expert split independent of
    # which frame is currently being displayed.
    assert tuple(before_gate[98, 16]) == MODULE.POLICY_COLOR
    assert tuple(after_gate[98, 36]) == MODULE.EXPERT_COLOR


def test_timeline_point_never_infers_a_new_boundary():
    row = {
        "timeline": [
            {"env_step": 0, "controller": "policy"},
            {"env_step": 50, "controller": "expert"},
        ]
    }
    assert MODULE._timeline_point(row, 49)["controller"] == "policy"
    assert MODULE._timeline_point(row, 50)["controller"] == "expert"


def test_render_uses_imageio_writer_append_data(tmp_path, monkeypatch):
    class Writer:
        def __init__(self):
            self.frames = []
            self.closed = False

        def append_data(self, frame):
            self.frames.append(frame)

        def close(self):
            self.closed = True

    writer = Writer()
    source = tmp_path / "raw.mp4"
    source.touch()
    monkeypatch.setattr(MODULE.imageio, "get_writer", lambda *_args, **_kwargs: writer)
    monkeypatch.setattr(
        MODULE.imageio,
        "get_reader",
        lambda *_args, **_kwargs: [np.zeros((12, 20, 3), dtype=np.uint8) for _ in range(2)],
    )
    row = {
        "video_path": str(source),
        "method": "late_success",
        "steps": 10,
        "success": True,
        "timeline": [{"env_step": 0, "controller": "policy"}],
    }
    summary = MODULE.render_episode(row=row, output=tmp_path / "out.mp4", title="human", fps=10)
    assert summary["frames"] == 2
    assert len(writer.frames) == 2
    assert writer.closed
