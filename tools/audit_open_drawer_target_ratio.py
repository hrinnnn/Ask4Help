#!/usr/bin/env python3
"""Independent membership/denominator audit using raw poses and stage records."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main(root):
    a=json.loads((root/'analysis.json').read_text());rs=a['rows'];lookup={r['seed']:r for r in rs};cache={}
    assert len(rs)==262 and sum(r['accepted'] for r in rs)==180
    def pose(seed):
        if seed not in cache:
            row=lookup[seed];d=Path(row['directory'])
            rows=json.loads((d/'task_state_timeline.json').read_text())['rows']
            p=np.array([r['tcp_position'] for r in rows]);o=np.array([r['object_position'] for r in rows])
            q=np.array([r['tcp_quaternion'] for r in rows]);oq=np.array([r['object_quaternion'] for r in rows])
            R=Rotation.from_quat(q[:,[1,2,3,0]]);O=Rotation.from_quat(oq[:,[1,2,3,0]])
            p=O.inv().apply(p-o);R=O.inv()*R
            g=np.load(d/'states.npy')[:,-2:].sum(axis=1)
            contact=np.array([r['object_grasped'] for r in rows])
            cache[seed]=(p,R,g,contact)
        return cache[seed]
    max_error=0.;green_count=0
    for r in rs:
        n=r['expert_anchors'];assert len(r['labels'])==n
        qp,qR,qg,qc=pose(r['seed']);rp,rR,rg,rc=pose(r['reference_seed'])
        valid=0
        for i,lab in enumerate(r['labels']):
            t=r['takeover']+i
            phase=next((p for p,(lo,hi) in r['blocks'].items() if lo<=t<hi),None)
            if phase is None:
                assert lab=='non_target' and r['reference_mapping'][i]==-1
                continue
            j=r['reference_mapping'][i]
            ra,rb=lookup[r['reference_seed']]['blocks'][phase];assert ra<=j<rb
            delta=np.array([np.linalg.norm(qp[t]-rp[j]),(qR[t]*rR[j].inv()).magnitude(),abs(qg[t]-rg[j])])
            score=float(np.linalg.norm(delta/np.array([.02,np.deg2rad(15),.01])))
            max_error=max(max_error,abs(score-r['distance'][i]))
            ok=(qc[t]==rc[j]) and score<=r['thresholds'][phase]
            assert ok==(lab=='target_compatible'),(r['seed'],i,score,r['thresholds'][phase])
            valid+=int(ok)
        assert valid==r['compatible_anchors'];green_count+=valid
        assert n==0 or abs(valid/n-r['Q'])<1e-12
    for anchor,value in a['summary'].items():
        picked=set(a['budget_manifest']['selected_source_episode_indices']['anchor_'+anchor])
        selected=[r for r in rs if r['anchor']==int(anchor) and r['accepted'] and r['accepted_index'] in picked]
        denominator=sum(r['expert_anchors'] for r in selected);numerator=sum(r['compatible_anchors'] for r in selected)
        assert denominator==2413 and numerator==value['training_budget']['compatible_anchors']
    video_count=0
    for video_manifest in root.glob('video_audit*.json'):
        for v in json.loads(video_manifest.read_text()):
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',v['output']]))['streams'][0]
            assert int(info['nb_frames'])==v['frames'];video_count+=1
    out=dict(status='PASS_DIAGNOSTIC',attempts=262,accepted=180,per_group_training_denominator=2413,
             green_anchors_all_attempts=green_count,max_distance_recompute_error=max_error,videos=video_count,
             scientific_status='target membership diagnostic; not a downstream utility validation')
    (root/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
