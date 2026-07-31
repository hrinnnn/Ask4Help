from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).parents[1] / "tools" / "libero_plus_failure" / "rollout_records.py"
SPEC = importlib.util.spec_from_file_location("libero_plus_failure_rollout_records", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_acc_and_single_sample_overlap_use_replan_alignment() -> None:
    previous = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    current = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    raw, ema = MODULE.velocity_normalized_acc(previous, current, execute_horizon=1, min_velocity=0.1)
    assert raw == pytest.approx(0.0)
    assert ema == pytest.approx(0.0)
    assert MODULE.single_sample_overlap(previous, current, execute_horizon=1) == pytest.approx(0.0)


def test_rollout_io_rejects_misaligned_feature_timelines(tmp_path: Path) -> None:
    episode = {"timeline": [{"decision": 0}, {"decision": 1}]}
    with pytest.raises(ValueError, match="expected 2"):
        MODULE.write_rollout(tmp_path / "bad", episode=episode, features={"bridge": [np.zeros(4)]})

    MODULE.write_rollout(
        tmp_path / "good",
        episode=episode,
        features={"bridge": [np.zeros(4), np.ones(4)], "action_expert_final": [np.zeros((10, 4)), np.ones((10, 4))]},
    )
    loaded, features = MODULE.read_rollout(tmp_path / "good")
    assert len(loaded["timeline"]) == features["bridge"].shape[0] == 2
