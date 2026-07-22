from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "tools" / "evaluate_rlinf_maniskill_pi05_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("official_maniskill_pi05_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checkpoint_native_action_chunk_is_truncated_and_clipped():
    actions = np.array([[0.2, -0.2, 0, 0, 0, 0, 2.0]] * 8, dtype=np.float32)
    low = np.array([-0.1] * 6 + [-1.0], dtype=np.float32)
    high = np.array([0.1] * 6 + [1.0], dtype=np.float32)
    chunk = MODULE.clip_action_chunk(actions, low, high, chunk_size=5)
    assert chunk.shape == (5, 7)
    assert np.allclose(chunk[:, 0], 0.1)
    assert np.allclose(chunk[:, 1], -0.1)
    assert np.allclose(chunk[:, -1], 1.0)


def test_checkpoint_native_action_chunk_rejects_joint_actions():
    with pytest.raises(ValueError, match=r"\[H,7\]"):
        MODULE.clip_action_chunk(np.zeros((5, 8)), np.zeros(7), np.ones(7), 5)
