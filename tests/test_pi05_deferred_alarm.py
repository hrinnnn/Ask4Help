import unittest
import numpy as np
from tools.pi05_timing_feedback import TimingFeedbackGate

class DeferredAlarmTests(unittest.TestCase):
    def gate(self):
        g=TimingFeedbackGate(1.,np.zeros(1),1.,1.,support_mode='continuous',max_wait_blocks=None,remember_deferred_alarm=True)
        g.begin_episode(0);g.commit({'direction':-1},[0],completed_episode=0)
        g.begin_episode(1);g.commit({'direction':1},[3],completed_episode=1)
        g.begin_episode(2)
        return g
    def test_vanished_score_does_not_erase_pending_alarm(self):
        g=self.gate()
        self.assertFalse(g.query([0],1.05,10)['stop'])
        q=g.query([3],.1,25)
        self.assertTrue(q['stop']);self.assertTrue(q['deferred_evidence_lost'])
        self.assertEqual(q['pending_alarm'],10)
    def test_wait_can_exceed_five_steps_with_later_evidence(self):
        g=self.gate()
        for t in [10,15,30,50]:self.assertFalse(g.query([0],1.05,t)['stop'])
    def test_no_alarm_before_any_risk_crossing(self):
        self.assertFalse(self.gate().query([3],.1,25)['stop'])
    def test_new_episode_has_no_stale_alarm(self):
        g=self.gate();g.query([0],1.05,10);g.begin_episode(3)
        self.assertIsNone(g.pending_alarm)
        self.assertFalse(g.query([3],.1,0)['stop'])
