"""Geometry/support diagnostics on fixed recorded histories; no new efficacy claim."""
import argparse,json
from pathlib import Path
import numpy as np
from pi05_feedback_artifacts import episode_path


def run(a):
    cal=json.loads((a.calibration/'calibration.json').read_text())
    center=np.load(a.calibration/'gate_arrays.npz')['center'];radius=cal['radius']*2
    result={}
    for task in ['grasp','goal']:
        root=a.root/task/'feedback';summary=json.loads((root/'summary.json').read_text())
        memory=[];labels=[];second=[];at_alarm=[];hard=0;soft=0;eligible=0;episodes=[]
        for row in summary['rows']:
            data=np.load(episode_path(root,row)/'trace.npz')
            z=(data['query_features'].astype(float)-center)/cal['scale']
            if len(memory)>=2:
                distances=np.linalg.norm(z[:,None,:]-np.asarray(memory)[None,:,:],axis=2)
                d2=np.sort(distances,axis=1)[:,1];second.extend(d2.tolist())
                hard+=int((d2<=radius).sum());eligible+=len(d2)
                weights=np.exp(-.5*(distances/radius)**2);mass=weights.sum(1)
                totals=weights@np.asarray(labels)
                votes=np.divide(totals,mass,out=np.zeros_like(totals),where=mass>0)
                ready=(mass>=2*np.exp(-.5))&(abs(votes)>=.25)
                soft+=int(ready.sum())
                episodes.append({'episode':row['episode'],'memory_events':len(memory),'queries':len(z),
                                 'hard_two_support':int((d2<=radius).sum()),'soft_mass_and_vote_ready':int(ready.sum())})
                if row['takeover'] is not None:at_alarm.append(float(d2[-1]))
            cue=row['cue']
            if cue is not None:
                memory.append(z[-2 if cue['attribution']=='previous_query' else -1]);labels.append(cue['direction'])
        result[task]={'bandwidth_and_old_cutoff':radius,'old_boundary_weight':float(np.exp(-.5)),
             'eligible_queries_with_two_past_events':eligible,'hard_two_support_queries':hard,
             'actual_nonzero_updates':sum(abs(q['direction'])>0 for r in summary['rows'] for q in r['queries']),
             'second_neighbor_quantiles':dict(zip(['q10','q50','q90'],np.quantile(second,[.1,.5,.9]).tolist())),
             'second_neighbor_distances_at_alarm':at_alarm,'soft_mass_and_vote_ready_on_fixed_history':soft,
             'soft_minimum_mass':float(2*np.exp(-.5)),'per_episode':episodes}
    output={'status':'RECORDED_HISTORY_DIAGNOSTIC_ONLY','results':result,
            'caveat':'Soft support is evaluated on the original recorded cues/states, which would change in a new closed loop. No timing, TASR or SR improvement is inferred.'}
    a.output.write_text(json.dumps(output,indent=2));print(json.dumps(output))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','calibration','output']:p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
