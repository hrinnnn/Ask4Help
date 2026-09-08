"""Independent raw-array and chronology audit of completed feedback collection."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root):
    summary=json.loads((root/'summary.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    cal_root=Path(provenance['calibration'])
    cal=json.loads((cal_root/'calibration.json').read_text())
    center=np.load(cal_root/'gate_arrays.npz')['center']
    cfg=provenance['feedback'];radius=cal['radius']*cfg['radius_multiplier']
    memory=[];report=[];total_cost=0;accepted=0
    for i,row in enumerate(summary['rows']):
        data=np.load(root/f'episode_{i:04d}'/'trace.npz')
        assert row['episode']==i and row['split']==('id' if i%2==0 else 'ood')
        n=len(data['actions']);assert len(data['main'])==len(data['wrist'])==len(data['qpos'])==n+1
        horizon=400 if summary['task'].startswith('open_drawer_') else (100 if summary['task']=='stackcube_legacy_ood' else 250)
        assert n<=horizon,(i,'trajectory horizon exceeded',n,horizon)
        if row['takeover'] is not None:assert row['takeover']+row['all_executed_expert_actions']<=horizon
        assert np.isfinite(data['actions']).all() and data['actions'].shape[1:]==(8,)
        assert data['main'].shape[1:]==data['wrist'].shape[1:]==(384,384,3)
        assert n==row['total_retained_path_actions']
        assert len(data['all_expert_actions'])==row['all_executed_expert_actions']
        if row['takeover'] is not None:
            suffix=data['actions'][row['takeover']:]
            assert len(suffix)==row['expert_suffix_actions']
            if len(suffix):assert np.array_equal(suffix,data['all_expert_actions'][-len(suffix):])
        assert len(row['queries'])==len(data['query_features'])==len(data['query_steps'])
        deadline=None
        for j,q in enumerate(row['queries']):
            z=(data['query_features'][j].astype(float)-center)/cal['scale']
            neighbors=[]
            for ep,z_i,y in memory:
                assert ep<i
                dist=np.linalg.norm(z-z_i)
                if cfg.get('support_mode','hard_radius')=='soft_mass' or dist<=radius:
                    neighbors.append((np.exp(-.5*(dist/radius)**2),y))
            direction=0.
            mass=sum(w for w,y in neighbors)
            ready=len(neighbors)>=cfg['minimum_interventions']
            if cfg.get('support_mode','hard_radius')=='soft_mass':
                ready=ready and mass>=cfg['minimum_interventions']*np.exp(-.5)
            if provenance['arm']=='feedback' and ready:
                weighted=sum(w*y for w,y in neighbors)
                if abs(weighted/mass)>=cfg['minimum_absolute_vote']:direction=weighted/(cfg['lambda']+mass)
            threshold=cal['baseline_threshold']*np.exp(-cfg['beta']*direction)
            assert np.isclose(threshold,q['threshold'],rtol=1e-7)
            if provenance['arm']=='feedback' and deadline is None and q['score']>cal['baseline_threshold'] and q['score']<=threshold:
                deadline=q['step']+cfg['execution_block']
            stop=q['score']>threshold or (deadline is not None and q['step']>=deadline)
            assert bool(stop)==q['stop'] and q['deadline']==deadline
            assert len(memory)==q['memory_episodes']
        cue=row['cue']
        if cfg.get('rule')=='commitment_v2' and row['takeover'] is not None:
            t=row['takeover'];opening=0.
            if t>=5 and row['expert_suffix_actions']>=5:
                width=data['qpos'][:,-2:].sum(1)
                opening=max(0.,-float(data['actions'][t-5:t,-1].mean()))*max(0.,float(data['actions'][t:t+5,-1].mean()))*max(0.,float(width[t+5]-width[t]))
            assert np.isclose(opening,row['opening_score_m'],atol=1e-7)
            errors=row['feedback_errors'];has_previous=len(row['queries'])>=2
            reversal=row['motion_reversal']
            if has_previous and reversal is not None and reversal>cal['reversal_reference']:
                expected_direction=1
            elif has_previous and errors and errors[0]>cal['error_reference'] and opening>cfg['opening_reference_m']:
                expected_direction=1
            elif any(error>cal['error_reference'] for error in errors[:5]):expected_direction=0
            elif len(errors)>=5:expected_direction=-1
            else:expected_direction=None
            assert (cue['direction'] if cue else None)==expected_direction
        if provenance['arm']=='feedback' and cue is not None:
            idx=-2 if cue['attribution']=='previous_query' else -1
            assert cue['credit_step']==int(data['query_steps'][idx])
            memory.append((i,(data['query_features'][idx].astype(float)-center)/cal['scale'],cue['direction']))
        total_cost+=row['all_executed_expert_actions'];accepted+=int(row['accepted'])
        report.append({'episode':i,'actions':n,'expert_cost':row['all_executed_expert_actions'],'status':'PASS'})
    assert total_cost==summary['all_expert_actions'] and accepted==summary['accepted']
    return {'status':'PASS','episodes':len(report),'all_expert_actions':total_cost,'accepted':accepted,'rows':report}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    result=audit(args.root);(args.root/'INDEPENDENT_COLLECTION_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
