from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "export_stackcube_grm_oneshot.py"
SPEC = importlib.util.spec_from_file_location("stackcube_grm_export", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_privileged_keyframes_include_success_and_monotonic_boundaries():
    keyframes = MODULE._keyframes({0: 0.0, 8: 0.25, 17: 0.5, 26: 0.75, 35: 1.0}, 36)
    assert keyframes[-1]["anotation"] == "success"
    assert all(item["end_frame_id"] > item["start_frame_id"] for item in keyframes)
    assert [item["start_frame_id"] for item in keyframes] == sorted(item["start_frame_id"] for item in keyframes)


def test_privileged_keyframes_reject_non_successful_source():
    with pytest.raises(ValueError, match="privileged-successful"):
        MODULE._keyframes({0: 0.0, 8: 0.25, 17: 0.5}, 20)
