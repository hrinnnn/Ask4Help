#!/usr/bin/env python3
"""Independent arithmetic / denominator audit; does not call scoring functions."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np


def travel(rows,states):
    p=np.array([r['tcp_position'] for r in rows],dtype=float)
    q=np.array([r['tcp_quaternion'] for r in rows],dtype=float)
    q/=np.linalg.norm(q,axis=1)[:,None]
    position=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()*100)
    rotation=float(np.rad2deg(2*np.arccos(np.clip(abs((q[1:]*q[:-1]).sum(axis=1)),0,1))).sum())
    gripper=float(abs(np.diff(states[:,-2:].sum(axis=1).astype(float))).sum()*1000)
    return np.array([position,rotation,gripper])


def main(root,previous):
    a=json.loads((root/'analysis.json').read_text());rows=a['attempts'];assert len(rows)==262
    assert sum(r['accepted'] for r in rows)==180
    assert len({r['seed'] for r in rows})==262
    for names in [('reference_seeds','calibration_seeds','heldout_seeds'),('id_reference_seeds','id_calibration_seeds','id_heldout_seeds')]:
        sets=[set(a['protocol'][k]) for k in names]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    refs={r['seed']:r for r in map(json.loads,(previous/'inputs/formal/anchor_0/accepted_experts.jsonl').read_text().splitlines())}
    errors=[]
    for row in rows:
        query_dir=Path(row['directory'])
        qr=json.loads((query_dir/'task_state_timeline.json').read_text())['rows']
        qs=np.load(query_dir/'states.npy');start=row['takeover'];end=start+row['query_steps']+1
        rr=refs[row['reference_seed']]
        rd=previous/'inputs/formal/anchor_0/accepted'/f"episode_{rr['episode_index']:06d}"
        rt=json.loads((rd/'task_state_timeline.json').read_text())['rows'];rs=np.load(rd/'states.npy')
        j0,j1=row['reference_start'],row['reference_end']+1
        actual=np.maximum(travel(qr[start:end],qs[start:end])-travel(rt[j0:j1],rs[j0:j1]),0)
        reported=np.array([row[k] for k in ['extra_path_cm','extra_rotation_deg','extra_gripper_travel_mm']])
        errors.append(abs(actual-reported))
        assert np.allclose(actual,reported,atol=.002), (row['seed'],actual,reported)
        expected=max(row['D_shape'],row['D_motion']);assert abs(expected-row['D_combined'])<1e-10
        mapping=np.array(row['reference_mapping']);assert len(mapping)==row['query_steps']+1
        assert np.all((np.diff(mapping)>=0)&(np.diff(mapping)<=5))
    videos=json.loads((root/'video_audit.json').read_text())
    for v in videos:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=nb_frames,width,height','-of','json',v['output']]))['streams'][0]
        assert int(info['nb_frames'])==v['frames'] and info['width']==1280 and info['height']==1120
    report=dict(status='PASS_DIAGNOSTIC_ARTIFACT_AUDIT',attempts=262,completed=180,incomplete=82,
                videos=len(videos),max_motion_debt_recomputation_error=np.max(errors,axis=0).tolist(),
                scientific_validation='NOT_ESTABLISHED: no blinded recovery labels or downstream utility association')
    (root/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();main(a.root,a.previous)
