import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from sweep_open_drawer_overlap_tolerance import classify


class ToleranceTests(unittest.TestCase):
    def fixture(self):
        return dict(takeover=10,distance=[None,.5,1.2,2.5,None],blocks={'target':(11,14)},thresholds={'target':1.})

    def test_baseline_and_monotonic_widening(self):
        r=self.fixture();c=[None,True,True,True,None]
        self.assertEqual([classify(r,c,f).count('target_compatible') for f in [1,1.5,2,3]],[1,2,2,3])

    def test_non_target_never_changes(self):
        r=self.fixture();labels=classify(r,[None,True,True,True,None],100.)
        self.assertEqual(labels[0],'non_target');self.assertEqual(labels[-1],'non_target')

    def test_contact_mismatch_cannot_be_fixed_by_radius(self):
        labels=classify(self.fixture(),[None,True,False,True,None],100.)
        self.assertEqual(labels[2],'target_mismatch')

    def test_equality_is_included(self):
        self.assertEqual(classify(self.fixture(),[None,True,True,True,None],1.2)[2],'target_compatible')

    def test_original_is_not_mutated(self):
        r=self.fixture();classify(r,[None,True,True,True,None],3.)
        self.assertEqual(r['thresholds'],{'target':1.});self.assertEqual(r['distance'][2],1.2)


if __name__=='__main__':unittest.main()
