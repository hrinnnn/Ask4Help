"""Unit tests for the lightweight VFD re-audit analysis helpers."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "stackcube_vfd_reaudit_analysis.py"
SPEC = importlib.util.spec_from_file_location("stackcube_vfd_reaudit_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def test_fiper_threshold_uses_per_trajectory_maxima_not_all_chunks() -> None:
    input_traces = [(1, [0.0, 8.0]), (2, [4.0, 4.0]), (3, [5.0, 5.0])]

    threshold = analysis.fiper_constant_threshold(input_traces, 0.5)

    assert threshold == pytest.approx(5.0)


def test_trace_extraction_filters_unsuccessful_id_episodes() -> None:
    episodes = [
        {
            "seed": 1,
            "success": True,
            "timeline": [{"oneway_vfd": 1.25, "twoway_vfd": 1.5}],
        },
        {
            "seed": 2,
            "success": False,
            "timeline": [{"oneway_vfd": 9.0, "twoway_vfd": 10.0}],
        },
    ]

    assert analysis.traces(episodes, "oneway_vfd", successful_only=True) == [
        (1, [1.25])
    ]
    assert analysis.traces(episodes, "twoway_vfd") == [(1, [1.5]), (2, [10.0])]
