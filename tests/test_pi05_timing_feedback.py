import unittest
import random
import numpy as np
from tools.pi05_timing_feedback import TimingFeedbackGate, action_block_error, corrective_motion, timing_cue, isolated_python_numpy_rng
from tools.pi05_timing_feedback import gripper_commitment_opening, commitment_timing_cue


class TimingFeedbackTests(unittest.TestCase):
    def test_saturated_gripper_commitment_is_distinguished_from_keep(self):
        opening=gripper_commitment_opening([-1]*5,[1]*5,0.,.079)
        self.assertAlmostEqual(opening,.079)
        self.assertEqual(timing_cue([.5],.1)['direction'],0)
        cue=commitment_timing_cue([.5],.1,opening_score=opening,opening_reference=.01,has_previous_query=True)
        self.assertEqual(cue['direction'],1)

    def test_normal_release_or_unobserved_prior_does_not_advance(self):
        self.assertEqual(commitment_timing_cue([.01]*5,.1,opening_score=.08,opening_reference=.01,has_previous_query=True)['direction'],-1)
        self.assertEqual(commitment_timing_cue([.5],.1,opening_score=.08,opening_reference=.01,has_previous_query=False)['direction'],0)
        self.assertEqual(gripper_commitment_opening([-1]*5,[-1]*5,0.,.08),0.)
        self.assertEqual(commitment_timing_cue([.5],.1,opening_score=.08,opening_reference=.09,has_previous_query=True)['direction'],0)

    def test_external_rng_pairing_and_restoration(self):
        random.seed(17); np.random.seed(21)
        expected=(random.random(),np.random.rand())
        random.seed(17); np.random.seed(21)
        with isolated_python_numpy_rng(23009):
            first=(random.randrange(10),np.random.rand())
        with isolated_python_numpy_rng(23009):
            second=(random.randrange(10),np.random.rand())
        self.assertEqual(first,second)
        self.assertEqual((random.random(),np.random.rand()),expected)

    def gate(self, enabled=True):
        return TimingFeedbackGate(1., np.zeros(2), 1., 1., enabled=enabled)

    def supported(self, direction, enabled=True):
        gate = self.gate(enabled)
        for episode in range(2):
            gate.begin_episode(episode)
            gate.commit({"direction": direction}, np.zeros(2), completed_episode=episode)
        gate.begin_episode(2)
        return gate

    def test_empty_memory_recovers_same_first_episode(self):
        a, b = self.gate(), self.gate(False)
        a.begin_episode(0); b.begin_episode(0)
        for s in [.5, 1., 1.1]:
            self.assertEqual(a.query([0, 0], s, 0), b.query([0, 0], s, 0))

    def test_weighted_fit_matches_equation(self):
        q = self.supported(1).query([0, 0], .9, 10)
        self.assertAlmostEqual(q['direction'], .5)
        self.assertAlmostEqual(q['threshold'], np.exp(-.25))
        self.assertTrue(q['stop'])

    def test_one_wait_block_is_not_renewed(self):
        gate = self.supported(-1)
        self.assertFalse(gate.query([0, 0], 1.1, 10)['stop'])
        self.assertEqual(gate.query([0, 0], 1.1, 12)['deadline'], 15)
        self.assertTrue(gate.query([9, 9], .2, 15)['stop'])
        gate.begin_episode(3)
        self.assertIsNone(gate.deadline)

    def test_neutral_and_far_recover_fixed_threshold(self):
        self.assertEqual(self.supported(0).query([0, 0], .9, 10)['threshold'], 1.)
        self.assertEqual(self.supported(1).query([4, 4], .9, 10)['threshold'], 1.)

    def test_disabled_gate_never_collects_memory(self):
        gate = self.supported(-1, enabled=False)
        self.assertEqual(gate.memory, [])
        self.assertTrue(gate.query([0, 0], 1.1, 10)['stop'])

    def test_no_current_episode_feedback(self):
        gate = self.gate(); gate.begin_episode(0)
        gate.commit({'direction': 1}, [0, 0], completed_episode=0)
        with self.assertRaises(ValueError): gate.query([0, 0], .9, 0)

    def test_cues_and_censoring(self):
        self.assertEqual(timing_cue([2.], 1.)['direction'], 0)
        self.assertEqual(timing_cue([.2] * 5, 1.)['direction'], -1)
        self.assertIsNone(timing_cue([.2] * 4 + [None], 1.))
        cue = timing_cue([.2] * 5, 1., reversal=2., reversal_reference=1., has_previous_query=True)
        self.assertEqual((cue['direction'], cue['attribution']), (1, 'previous_query'))

    def test_error_uses_only_actual_execution_block(self):
        pred = np.zeros((2, 10, 8)); pred[:, 5:] = 100
        self.assertEqual(action_block_error(pred, np.zeros((5, 8))), 0.)
        self.assertIsNone(action_block_error(pred, np.zeros((4, 8))))
        self.assertEqual(action_block_error(pred, np.ones((5, 8))), 1.)

    def test_motion_direction(self):
        self.assertEqual(corrective_motion([0, 0, 0], [.02, 0, 0], [.04, 0, 0], .04, .02, .01), 0.)
        self.assertAlmostEqual(corrective_motion([0, 0, 0], [.02, 0, 0], [0, 0, 0], .04, .02, .03), np.sqrt(2))


if __name__ == '__main__':
    unittest.main()
