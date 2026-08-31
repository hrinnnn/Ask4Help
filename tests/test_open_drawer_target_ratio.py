import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_open_drawer_target_ratio import stage_blocks,relative_pose,compare_block


class TargetTests(unittest.TestCase):
    def fixture(self):
        return dict(actual_takeover_step=0,expert_action_steps=159,accepted=True,expert_result=dict(
            drawer_opened_after_takeover=True,drawer_opened_before_takeover=False,
            direct_handle_pregrasp_steps=26,direct_handle_reach_steps=10,direct_pull_steps=27,
            direct_pull_from_existing_handle_grasp=False,object_grasped_before_takeover=False,
            direct_object_pregrasp_steps=27,direct_object_reach_steps=10,direct_object_close_steps=3,
            direct_lift_steps=10,direct_transport_steps=17,direct_place_steps=9))

    def test_target_stops_before_lift(self):
        blocks,end=stage_blocks(self.fixture())
        self.assertEqual(blocks,dict(rotation_pregrasp=(79,106),reach=(106,116),close=(116,119)))
        self.assertEqual(end,119)
        self.assertFalse(any(lo<=119<hi for lo,hi in blocks.values()))

    def test_absolute_takeover_offset(self):
        m=self.fixture();m['actual_takeover_step']=50
        blocks,end=stage_blocks(m);self.assertEqual(blocks['close'],(166,169));self.assertEqual(end,169)

    def test_no_target_before_drawer_open(self):
        m=self.fixture();m['expert_result']['drawer_opened_after_takeover']=False
        self.assertEqual(stage_blocks(m),({},None))

    def test_already_grasped_needs_no_grasp_supervision(self):
        m=self.fixture();m['expert_result']['object_grasped_before_takeover']=True
        self.assertEqual(stage_blocks(m)[0],{})

    def test_relative_pose_invariant_to_shared_world_transform(self):
        pose=dict(position=np.array([[.1,.2,.3]]),quaternion=np.array([[1.,0,0,0]]),width=np.array([[.05]]))
        rows=[dict(object_position=[.2,.1,0],object_quaternion=[1.,0,0,0])]
        r=relative_pose(pose,rows);G=Rotation.from_rotvec([.2,-.1,.8]);p=np.array([1.,2.,3.])
        gq=G.as_quat()[[3,0,1,2]]
        moved=dict(position=G.apply(pose['position'])+p,quaternion=gq[None],width=pose['width'])
        shifted=[dict(object_position=(G.apply(rows[0]['object_position'])+p).tolist(),object_quaternion=gq.tolist())]
        actual=relative_pose(moved,shifted)
        np.testing.assert_allclose(actual['position'],r['position'],atol=1e-12)
        np.testing.assert_allclose(abs(actual['quaternion']),abs(r['quaternion']),atol=1e-12)


if __name__=='__main__':unittest.main()
