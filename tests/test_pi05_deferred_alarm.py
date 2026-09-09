import unittest
import numpy as np
from tools.pi05_timing_feedback import TimingFeedbackGate, observed_agreement_steps

class DeferredAlarmTests(unittest.TestCase):
    def test_feedback_budget_survives_later_episodes(self):
        g=TimingFeedbackGate(1.,np.zeros(1),1.,1.,max_feedback_events=2)
        for i in range(5):
            g.begin_episode(i);g.commit({'direction':1},[0],completed_episode=i)
        self.assertEqual([x['episode'] for x in g.memory],[0,1])
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
    def test_measured_duration_expires_without_restarting(self):
        g=self.gate();g.use_later_duration=True
        g.memory[0]['later_valid_steps']=15
        self.assertFalse(g.query([0],1.05,10)['stop'])
        self.assertFalse(g.query([0],.1,20)['stop'])
        self.assertTrue(g.query([0],.1,25)['stop'])
    def test_gripper_mismatch_is_not_diluted_by_arm_agreement(self):
        blocks=[dict(offset=k,overall_MSE=.06,gripper_sign_disagreement=0.) for k in [0,5,10]]
        blocks[2]['gripper_sign_disagreement']=.2
        self.assertEqual(observed_agreement_steps(blocks,.18),10)
    def test_no_extrapolation_past_observed_prefix(self):
        self.assertEqual(observed_agreement_steps([dict(offset=0,overall_MSE=.01,gripper_sign_disagreement=0.)],.18),5)
