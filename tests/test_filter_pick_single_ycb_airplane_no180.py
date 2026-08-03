from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_SOURCE = Path(__file__).parents[1] / "tools" / "filter_pick_single_ycb_airplane_no180.py"
_SPEC = importlib.util.spec_from_file_location("airplane_filter_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class No180SelectionTest(unittest.TestCase):
    def test_only_negative_direct_grasp_band_is_retained(self) -> None:
        rows = [{"object_yaw": np.deg2rad(-90.0)}, {"object_yaw": np.deg2rad(90.0)}, {"object_yaw": 0.0}]
        selected = _MODULE.select_no180_rows(rows)
        self.assertEqual(selected, [rows[0]])
