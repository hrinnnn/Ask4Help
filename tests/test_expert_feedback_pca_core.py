import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import numpy as np
from expert_feedback_pca_core import ExpertFeedbackPCA,first_boundary_feedback

class FeedbackTests(unittest.TestCase):
 def test_chunk_label_times(self):
  loss=np.r_[np.zeros(13),2.,np.zeros(15)];v=np.minimum(10,len(loss)-np.arange(len(loss)));f=first_boundary_feedback(loss,v,q=1.)
  self.assertEqual(f['event'],13);self.assertFalse(f['request'][8]);self.assertTrue(f['request'][9])
 def test_censor_does_not_invent_event(self):
  n=20;v=np.minimum(10,n-np.arange(n));f=first_boundary_feedback(np.zeros(n),v,allow_censored_wait=True)
  self.assertIsNone(f['event']);np.testing.assert_array_equal(f['indices'],np.arange(6));self.assertFalse(f['request'].any())
 def test_gap_truncation(self):
  n=30;v=np.minimum(10,n-np.arange(n));loss=np.zeros(n);loss[4]=np.nan;loss[7]=9;f=first_boundary_feedback(loss,v)
  self.assertIsNone(f['event']);self.assertEqual(f['observed_full_windows'],4)
 def test_actual_pca_consistent_constraints(self):
  # Same score function |second coordinate| for all examples; neighborhood
  # contains distinct nearby states, not arbitrary scores on identical states.
  gate=ExpertFeedbackPCA(3.,np.zeros(2),1.,.5,1.,H=1,delta=1)
  self.assertFalse(gate.query(np.array([0.,2.6]),2.6)['stop'])
  gate.add_episode('a',np.array([[0.,2.4]]),np.array([2.4]),np.array([2.]),np.array([1]))
  self.assertTrue(gate.query(np.array([0.,2.6]),2.6)['stop'])
  self.assertFalse(gate.query(np.array([9.,2.6]),2.6)['stop'])
 def test_wait_and_conflict(self):
  g=ExpertFeedbackPCA(3.,np.zeros(2),1.,1.,1.,H=1,delta=1)
  g.add_episode('a',np.array([[0.,3.4],[2.,4.]]),np.array([3.4,4.]),np.array([0.,2.]),np.array([1,1]))
  d=g.query(np.array([0.,3.2]),3.2);self.assertFalse(d['stop']);self.assertTrue(d['baseline_stop'])
  g.add_episode('b',np.array([[0.,2.5]]),np.array([2.5]),np.array([2.]),np.array([1]))
  self.assertEqual(g.query(np.array([0.,3.2]),3.2)['reason'],'conflict')
 def test_invalid_future_mask(self):
  with self.assertRaises(ValueError):first_boundary_feedback(np.zeros(2),np.array([10,10]))
 def test_prefix_credit_only_observed_previous_query(self):
  gate=ExpertFeedbackPCA(3.,np.zeros(2),1.,.3,1.,H=1,delta=5)
  x=np.array([[0.,2.4],[2.,3.4]]);score=np.array([2.4,3.4]);times=np.array([10,15])
  self.assertFalse(gate.query(x[0],score[0])['stop'])
  result=gate.add_takeover_prefix_credit('late',x,score,times,np.ones(8)*2,np.ones(8),regression=.2,regression_bound=.05,progress_supported=True)
  self.assertEqual(result['credit_step'],10);self.assertTrue(gate.query(x[0],score[0])['stop'])
 def test_high_loss_alone_does_not_invent_earlier_failure(self):
  gate=ExpertFeedbackPCA(3.,np.zeros(2),1.,.3,1.,H=1,delta=5)
  result=gate.add_takeover_prefix_credit('gap',np.array([[0.,2.4],[2.,3.4]]),np.array([2.4,3.4]),np.array([10,15]),np.ones(8)*2,np.ones(8))
  self.assertEqual(result['credit_step'],15)
  # If all request labels are recorded only where the original gate already
  # fires, projecting tau0 cannot lower it: the current-only cold-start trap.
  self.assertEqual(gate.query(np.array([2.,3.4]),3.4)['threshold'],3.)
 def test_long_low_loss_prefix_waits_at_true_takeover_state(self):
  gate=ExpertFeedbackPCA(3.,np.zeros(2),1.,.3,1.,H=1,delta=5)
  result=gate.add_takeover_prefix_credit('early',np.array([[2.,3.4]]),np.array([3.4]),np.array([15]),np.r_[np.zeros(6),2.,0.],np.ones(8))
  self.assertFalse(result['request']);self.assertFalse(gate.query(np.array([2.,3.4]),3.4)['stop'])

if __name__=='__main__':unittest.main()
