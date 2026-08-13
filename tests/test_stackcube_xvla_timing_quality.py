import numpy as np

from tools.summarize_stackcube_xvla_timing_quality import (
    aligned_completion_distance,
    dce,
    dtw_average,
)


def test_dtw_aligns_different_execution_speeds() -> None:
    slow = np.asarray([[0.0], [0.0], [1.0], [1.0], [2.0]])
    fast = np.asarray([[0.0], [1.0], [2.0]])
    assert dtw_average(slow, fast) == 0.0


def test_completion_alignment_can_start_inside_nominal_trajectory() -> None:
    nominal = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    expert = np.asarray([[2.0], [3.0]])
    assert aligned_completion_distance(expert, nominal) == 0.0


def test_dce_requires_both_alignment_and_saving() -> None:
    assert dce(1.0, 1.0) > dce(1.0, 0.2)
    assert dce(1.0, 0.2) > dce(0.1, 0.2)
