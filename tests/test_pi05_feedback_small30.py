import unittest
import numpy as np
from tools.pi05_timing_feedback import TimingFeedbackGate

class Small30Tests(unittest.TestCase):
    def make(self,direction):
        g=TimingFeedbackGate(1.,np.zeros(2),1.,1.,support_mode='continuous',max_wait_blocks=None)
        g.begin_episode(0);g.commit({'direction':direction},[0,0],completed_episode=0);g.begin_episode(1)
        return g
    def test_single_event_changes_threshold(self):
        q=self.make(1).query([0,0],.9,0)
        self.assertAlmostEqual(q['threshold'],np.exp(-.5/3))
        self.assertTrue(q['stop'])
    def test_no_forced_delayed_takeover(self):
        g=self.make(-1)
        for step in [0,5,10,30,90]:
            q=g.query([0,0],1.1,step)
            self.assertFalse(q['stop']);self.assertIsNone(q['deadline'])
    def test_recovered_score_crossing_still_takes_over(self):
        self.assertTrue(self.make(-1).query([0,0],2.,20)['stop'])
    def test_far_evidence_decays(self):
        g=self.make(1)
        self.assertLess(g.query([0,0],0.,0)['threshold'],g.query([2,0],0.,0)['threshold'])
        self.assertAlmostEqual(g.query([10,0],0.,0)['threshold'],1.)
    def test_conflict_is_continuous_not_vote_cliff(self):
        g=self.make(1);g.commit({'direction':-1},[0,0],completed_episode=1);g.begin_episode(2)
        self.assertEqual(g.query([0,0],0.,0)['threshold'],1.)
    def test_future_event_is_still_forbidden(self):
        g=self.make(1);g.begin_episode(0)
        with self.assertRaises(ValueError):g.query([0,0],0.,0)
