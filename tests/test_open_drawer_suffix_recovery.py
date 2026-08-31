import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_open_drawer_suffix_recovery import motion,pose_channels,suffix_map,trim_success


def pose(n=41):
    return dict(position=np.zeros((n,3)),quaternion=np.tile([1.,0,0,0],(n,1)),width=np.ones((n,1))*.08)


class SuffixTests(unittest.TestCase):
    def test_full_turn_is_not_zero_motion(self):
        p=pose();vectors=np.zeros((41,3));vectors[:,2]=np.linspace(0,2*np.pi,41)
        p['quaternion']=Rotation.from_rotvec(vectors).as_quat()[:,[3,0,1,2]]
        self.assertAlmostEqual(np.rad2deg(motion(p)[1]),360,places=8)
        self.assertAlmostEqual(pose_channels(p,p)[0,-1,1],0,places=7)

    def test_quaternion_sign_not_a_turn(self):
        p=pose();p['quaternion'][::2]*=-1
        self.assertAlmostEqual(motion(p)[1],0,places=9)

    def test_gripper_cycle_retained(self):
        p=pose(5);p['width']=np.array([.08,0,.08,0,.08])[:,None]
        self.assertAlmostEqual(motion(p)[2],.32)

    def test_motion_length_independent_of_sampling_speed(self):
        a=pose(21);b=pose(101)
        a['position'][:,0]=np.linspace(0,1,21);b['position'][:,0]=np.linspace(0,1,101)
        np.testing.assert_allclose(motion(a),motion(b))

    def test_no_query_prefix_discard_and_monotone(self):
        cost=np.random.default_rng(4).random((40,50))
        mapping=suffix_map(cost,(5,15),True)
        self.assertEqual(len(mapping),40)
        self.assertTrue(5<=mapping[0]<=15)
        self.assertEqual(mapping[-1],49)
        self.assertTrue(np.all((np.diff(mapping)>=0)&(np.diff(mapping)<=5)))

    def test_cut_after_first_success(self):
        item=dict(pose=pose(10),rows=[dict(step=i,success=i==5) for i in range(10)])
        out=trim_success(item)
        self.assertEqual(len(out['rows']),6)


if __name__=='__main__':unittest.main()
