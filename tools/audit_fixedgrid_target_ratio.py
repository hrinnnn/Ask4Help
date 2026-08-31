#!/usr/bin/env python3
"""Independently verify labels, task budgets, contact and raw video alignment."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main(root):
    result=json.loads((root/'analysis.json').read_text());report={}
    for task,data in result.items():
        refs={r['seed']:r for r in data['rows'] if r['step']==0};cache={}
        def load(step,seed):
            key=(step,seed)
            if key not in cache:cache[key]=dict(np.load(root/task/f'pose_step_{step}_seed_{seed}.npz'))
            return cache[key]
        max_error=0.;total=0
        for r in data['rows']:
            p=load(r['step'],r['seed']);ref=load(0,r['reference_seed']);raw=np.load(r['arrays'])
            n=r['expert_anchors'];start=r['expert_start']
            assert len(r['labels'])==len(raw['qpos'])==n
            assert r['seed']!=r['reference_seed']
            assert np.allclose(p['world_position'],raw['task_states'][start:start+n,6:9],atol=1e-4)
            axes=p['world_rotation'][r['blocks']['close'][1]-1]
            expected_p=(p['world_position']-p['object_position'])@axes
            assert np.allclose(expected_p,p['position'])
            expected_R=axes.T@p['world_rotation']
            assert np.allclose(expected_R,Rotation.from_quat(p['quaternion'][:,[1,2,3,0]]).as_matrix())
            valid=0
            for i,label in enumerate(r['labels']):
                phase=next((k for k,(a,b) in r['blocks'].items() if a<=i<b),None)
                if phase is None:
                    assert label=='non_target' and r['reference_mapping'][i]==-1;continue
                j=r['reference_mapping'][i];lo,hi=refs[r['reference_seed']]['blocks'][phase];assert lo<=j<hi
                dp=np.linalg.norm(p['position'][i]-ref['position'][j]);qa=p['quaternion'][i]/np.linalg.norm(p['quaternion'][i]);qb=ref['quaternion'][j]/np.linalg.norm(ref['quaternion'][j])
                da=2*np.arccos(np.clip(abs(qa@qb),0,1));dg=float(abs(p['width'][i,0]-ref['width'][j,0]))
                distance=float(np.sqrt((dp/.02)**2+(da/np.deg2rad(15))**2+(dg/.01)**2));max_error=max(max_error,abs(distance-r['distance'][i]))
                ok=(p['contact'][i]==ref['contact'][j]) and distance<=r['thresholds'][phase]+1e-9
                assert ok==(label=='target_compatible'),(task,r['step'],r['seed'],i,distance)
                valid+=int(ok)
            assert valid==r['compatible_anchors'];assert abs(valid/n-r['Q'])<1e-12;total+=n
        for step,summary in data['summary'].items():
            rows=[r for r in data['rows'] if r['step']==int(step) and r['selected']]
            denominator=sum(r['expert_anchors'] for r in rows);numerator=sum(r['compatible_anchors'] for r in rows)
            assert denominator==data['protocol']['budget'];assert numerator==summary['compatible_anchors'];assert abs(numerator/denominator-summary['Q'])<1e-12
        report[task]=dict(episodes=len(data['rows']),expert_anchors_checked=total,budget=data['protocol']['budget'],max_distance_recompute_error=max_error)
    videos=json.loads((root/'video_audit.json').read_text())
    for v in videos:
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames','-of','json',v['output']]))['streams'][0]
        assert int(metadata['nb_frames'])==v['frames'];assert v['source']['state_step_offset']==0
    final=dict(status='PASS_DIAGNOSTIC_ARTIFACT_AUDIT',tasks=report,videos=len(videos),not_a_downstream_utility_validation=True)
    (root/'audit.json').write_text(json.dumps(final,indent=2));print(json.dumps(final))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
