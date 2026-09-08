import unittest
import numpy as np
from tools.pi05_timing_feedback import TimingFeedbackGate


class SoftSupportTest(unittest.TestCase):
    def gate(self,points,directions=None,mode='soft_mass'):
        gate=TimingFeedbackGate(1.,np.zeros(2),1.,1.,support_mode=mode)
        for i,p in enumerate(points):
            gate.begin_episode(i)
            gate.commit({'direction':(directions or [1]*len(points))[i]},np.asarray(p),completed_episode=i)
        gate.begin_episode(len(points));return gate

    def test_moderately_similar_events_can_pool_weight_without_widening_bandwidth(self):
        points=[[1.1,0]]*3
        soft=self.gate(points).query([0,0],.95,0)
        hard=self.gate(points,mode='hard_radius').query([0,0],.95,0)
        self.assertTrue(soft['support_ready']);self.assertTrue(soft['stop'])
        self.assertEqual(hard['direction'],0)
        self.assertAlmostEqual(soft['support_mass'],3*np.exp(-.5*1.1**2))

    def test_one_close_event_with_far_events_is_insufficient(self):
        row=self.gate([[0,0]]+[[10,0]]*20).query([0,0],.95,0)
        self.assertFalse(row['support_ready']);self.assertEqual(row['direction'],0)

    def test_distant_events_do_not_activate_by_count_alone(self):
        row=self.gate([[10,0]]*30).query([0,0],.95,0)
        self.assertFalse(row['support_ready']);self.assertEqual(row['direction'],0)

    def test_soft_postponement_keeps_the_original_single_block_deadline(self):
        gate=self.gate([[1.1,0]]*3,[-1,-1,-1])
        first=gate.query([0,0],1.05,0)
        self.assertFalse(first['stop']);self.assertEqual(first['deadline'],5)
        self.assertTrue(gate.query([0,0],.5,5)['stop'])


if __name__=='__main__':unittest.main()
