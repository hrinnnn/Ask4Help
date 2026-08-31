import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_open_drawer_piecewise_ed import align, onset


class AlignmentTests(unittest.TestCase):
    def test_onset_and_confirmation_are_distinct(self):
        values = np.zeros(21)
        values[10] = values[15] = 2
        self.assertEqual(onset(values, 1), (10, 15))

    def test_lone_terminal_crossing_is_censored(self):
        self.assertEqual(onset([0]*20+[2], 1), (None, None))

    def test_strict_threshold(self):
        self.assertEqual(onset([1]*21, 1), (None, None))

    def test_greedy_prefix_invariance(self):
        costs = np.random.default_rng(1).random((60, 80))
        np.testing.assert_array_equal(align(costs[:20], 'clockfree_monotone'),
                                      align(costs, 'clockfree_monotone')[:20])

    def test_dp_order_jump_and_free_endpoint(self):
        costs = np.ones((30, 40))*20
        costs[:, 3] = 0
        mapping = align(costs, 'open_end_monotone_dp')
        self.assertTrue(np.all(np.diff(mapping)>=0))
        self.assertTrue(np.all(np.diff(mapping)<=5))
        self.assertEqual(mapping[-1],3)

    def test_dp_escapes_myopic_local_minimum(self):
        costs = np.full((3,8),50.)
        costs[0,0]=0;costs[0,4]=1
        costs[1,0]=0;costs[1,6]=0
        costs[2,7]=0
        greedy=align(costs,'clockfree_monotone')
        dp=align(costs,'open_end_monotone_dp')
        self.assertLess(costs[np.arange(3),dp].sum(),costs[np.arange(3),greedy].sum())


if __name__=='__main__':
    unittest.main()
