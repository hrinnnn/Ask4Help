import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from expert_feedback_flexible import FlexibleFeedbackPCA

class FlexibleTests(unittest.TestCase):
 def gate(self):return FlexibleFeedbackPCA(3.,np.zeros(2),1.,1.,1.,H=1,delta=1).configure()
 def test_current_request_can_shift_with_support(self):
  g=self.gate();x=np.array([0.,3.1]);self.assertEqual(g.query(x,2.9)['threshold'],3.)
  for e in ['a','b']:g.memory.append(dict(episode=e,z=x,score=3.1,request=True))
  self.assertTrue(g.query(x,2.9)['stop']);self.assertGreater(g.query(x,2.9)['threshold'],3*np.exp(-.5))
 def test_wait_raises_threshold(self):
  g=self.gate();x=np.array([0.,3.1])
  for e in ['a','b']:g.memory.append(dict(episode=e,z=x,score=3.1,request=False))
  self.assertFalse(g.query(x,3.1)['stop'])
 def test_single_episode_cannot_fill_support(self):
  g=self.gate();g.memory=[dict(episode='a',z=np.zeros(2),score=3.1,request=True)]*20
  self.assertEqual(g.query(np.zeros(2),2.9)['threshold'],3.)
 def test_conflict_and_far_fallback(self):
  g=self.gate();g.memory=[dict(episode=str(i),z=np.zeros(2),score=3.,request=i==0) for i in range(2)]
  self.assertEqual(g.query(np.zeros(2),2.9)['reason'],'conflict');self.assertEqual(g.query(np.ones(2)*8,2.9)['reason'],'no_support')

if __name__=='__main__':unittest.main()
