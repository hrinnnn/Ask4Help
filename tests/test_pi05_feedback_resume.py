import json,tempfile,unittest
from pathlib import Path
import numpy as np
from tools.pi05_feedback_artifacts import completed_rows,restore_memory,episode_path
from tools.pi05_timing_feedback import TimingFeedbackGate


class ResumeTest(unittest.TestCase):
    def save(self,root,i,direction):
        p=root/('episode_%04d'%i);p.mkdir()
        feature=np.array([[1.+i*.1,0],[1.1+i*.1,0]],dtype=np.float32)
        previous=direction==1
        cue={'direction':direction,'attribution':'previous_query' if previous else 'takeover_query','credit_step':0 if previous else 5}
        r={'episode':i,'cue':cue}
        (p/'result.json').write_text(json.dumps(r));(p/'trajectory.mp4').write_bytes(b'unit-fixture')
        np.savez(p/'trace.npz',query_features=feature,query_steps=[0,5])
        return r,feature[0 if previous else 1]

    def test_partial_prefix_and_later_chunk_restore_identical_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'first';first.mkdir();(first/'provenance.json').write_text('{}')
            reference=TimingFeedbackGate(1.,np.zeros(2),1.,1.,support_mode='soft_mass')
            for i,y in enumerate([1,-1,1]):
                r,f=self.save(first,i,y);reference.begin_episode(i);reference.commit(r['cue'],f,completed_episode=i)
            later=Path(tmp)/'later';later.mkdir()
            (later/'provenance.json').write_text(json.dumps({'resume_from':str(first)}))
            r,f=self.save(later,3,-1);reference.begin_episode(3);reference.commit(r['cue'],f,completed_episode=3)
            rows=completed_rows(later);self.assertEqual([r['episode'] for r in rows],[0,1,2,3])
            self.assertEqual(episode_path(later,rows[0]).parent,first)
            restored=TimingFeedbackGate(1.,np.zeros(2),1.,1.,support_mode='soft_mass')
            restore_memory(restored,rows)
            reference.begin_episode(4);restored.begin_episode(4)
            self.assertEqual(reference.query([1.15,0],1.01,10),restored.query([1.15,0],1.01,10))
            (later/'summary.json').write_text(json.dumps({'rows':rows}))
            self.assertEqual(completed_rows(later),rows)

    def test_noncontiguous_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'provenance.json').write_text('{}')
            self.save(root,1,1)
            with self.assertRaises(AssertionError):completed_rows(root)


if __name__=='__main__':unittest.main()
