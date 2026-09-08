"""Independent raw evidence check; oracle failures remain failures."""
import argparse,json
from pathlib import Path
import numpy as np


def run(root):
    summary=json.loads((root/'summary.json').read_text());rows=[]
    assert len(summary['rows'])==4
    for i,row in enumerate(summary['rows']):
        d=np.load(root/f'episode_{i:04d}/trace.npz');n=len(d['actions'])
        assert n<=400 and n==row['takeover']+row['expert_actions']
        assert len(d['main'])==len(d['wrist'])==len(d['qpos'])==n+1
        assert d['main'].shape[1:]==d['wrist'].shape[1:]==(384,384,3)
        assert d['actions'].shape[1:]==(8,) and np.isfinite(d['actions']).all()
        assert np.allclose(d['object_p'][0],row['reset']['object_pose']['p'],atol=1e-6)
        assert np.allclose(d['target_p'][0],row['reset']['target_pose']['p'],atol=1e-6)
        assert bool(d['success'][-1])==row['strict_success']
        assert np.array_equal(d['actions'][row['takeover']:],d['all_expert_actions'])
        report=row['expert_report']
        assert report['planner_mode']=='shortest_joint_path'
        if row['drawer_opened_at_takeover']:
            assert report['direct_handle_pregrasp_steps']==report['direct_handle_reach_steps']==report['direct_pull_steps']==0
        rows.append({'episode':i,'split':row['split'],'takeover':row['takeover'],
                     'expert_actions':row['expert_actions'],'success':row['strict_success'],
                     'failure_phase':report.get('failure_phase'),'raw_contract':'PASS'})
    result={'status':'RAW_SMOKE_CONTRACT_PASS','rows':rows,
            'oracle_successes':sum(r['success'] for r in rows),'episodes':len(rows),
            'qualification':'This checks execution evidence, not a universally successful expert or a gate efficacy result.'}
    (root/'INDEPENDENT_SMOKE_AUDIT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
