import json,unittest
from pathlib import Path


class ReservedEvaluationTest(unittest.TestCase):
    def test_reserved_pools_are_disjoint_and_preserve_endpoints(self):
        root=Path(__file__).resolve().parents[1]
        p=json.loads((root/'configs/pipelines/pi05_feedback_reserved_evaluation_v1.json').read_text())
        self.assertEqual(p['episodes_per_split'],100)
        self.assertEqual(p['inference_mode'],'eval')
        seen=set()
        for task,spec in p['tasks'].items():
            for split in ['id','ood']:
                pool=set(range(spec[split+'_start'],spec[split+'_start']+100))
                self.assertFalse(seen & pool)
                self.assertGreater(min(pool),9000000)
                seen.update(pool)
            self.assertEqual(spec['primary_endpoint'],'ever_grasped' if task=='airplane_yaw_ood' else 'strict_success')
        self.assertEqual(len(seen),800)


if __name__=='__main__':unittest.main()
