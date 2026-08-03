from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SOURCE = Path(__file__).parents[1] / "tools" / "pick_single_ycb_airplane_eval_common.py"
_SPEC = importlib.util.spec_from_file_location("airplane_eval_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class ClipActionChunkTest(unittest.TestCase):
    def test_bounds_and_horizon(self) -> None:
        values = np.array([[[2.0, -2.0], [0.5, 0.25], [0.1, 0.2]]], dtype=np.float32)
        clipped = _MODULE.clip_action_chunk(values, np.array([-1.0, -1.0]), np.array([1.0, 1.0]), 2)
        np.testing.assert_allclose(clipped, np.array([[1.0, -1.0], [0.5, 0.25]], dtype=np.float32))

    def test_rejects_short_or_wrong_width(self) -> None:
        bounds = np.array([-1.0, -1.0], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Expected action chunk"):
            _MODULE.clip_action_chunk(np.zeros((1, 1, 2), dtype=np.float32), bounds, -bounds, 2)
        with self.assertRaisesRegex(ValueError, "Action bounds"):
            _MODULE.clip_action_chunk(np.zeros((1, 2, 3), dtype=np.float32), bounds, -bounds, 2)
