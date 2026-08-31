import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_fixedgrid_target_ratio import define_blocks,compare


class TargetRatioTests(unittest.TestCase):
    def fixture(self,task='stackcube',prefix=0):
        n=26 if task=='stackcube' else 40
        actions=np.zeros((prefix+n,8));actions[:,-1]=1;actions[prefix+10:,-1]=-1
        ts=np.zeros((prefix+n+1,18 if task=='stackcube' else 26));ts[prefix+11:,-2]=1;ts[prefix+20:,2]=.1
        meta=dict(seed=1)
        if task=='airplane':meta['oracle']=dict(selected_candidate='neck_center_test',attempts=[dict(candidate='neck_center_test',accepted=True,accepted_lift=True,still_grasped_after_lift=True,close_executed_steps=4,stable_grasp_steps=4)])
        return dict(train=dict(expert_start_step=prefix,expert_action_steps=n),meta=meta),dict(actions=actions,task_states=ts)

    def test_stackcube_excludes_first_lift_chunk(self):
        e,a=self.fixture();blocks,_=define_blocks('stackcube',e,a)
        self.assertEqual(blocks,dict(approach_alignment=(0,10),close=(10,15)))

    def test_airplane_uses_saved_stable_close_count(self):
        e,a=self.fixture('airplane');blocks,_=define_blocks('airplane',e,a)
        self.assertEqual(blocks['close'],(10,14))

    def test_policy_prefix_not_part_of_target_bounds(self):
        e,a=self.fixture(prefix=20);a['actions'][:20,-1]=-1
        blocks,_=define_blocks('stackcube',e,a);self.assertEqual(blocks['close'],(10,15))

    def test_transient_contact_rejected_for_reference(self):
        e,a=self.fixture();a['task_states'][14,-2]=0
        with self.assertRaises(AssertionError):define_blocks('stackcube',e,a)

    def test_no_closing_command_is_not_silently_accepted(self):
        e,a=self.fixture();a['actions'][:,-1]=1
        with self.assertRaises(ValueError):define_blocks('stackcube',e,a)


if __name__=='__main__':unittest.main()
