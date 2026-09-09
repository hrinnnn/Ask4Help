"""Frozen-memory detection and actual paired timing; never learns from test labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from pi05_timing_feedback import TimingFeedbackGate
from pi05_feedback_artifacts import restore_memory, episode_path

def metrics(labels, alarms):
    y=np.asarray(labels,dtype=bool);a=np.asarray(alarms,dtype=bool)
    tp=int((y&a).sum());fn=int((y&~a).sum());fp=int((~y&a).sum());tn=int((~y&~a).sum())
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,BA=(tp/(tp+fn)+tn/(tn+fp))/2 if tp+fn and tn+fp else None,
                FPR=fp/(fp+tn) if fp+tn else None,FNR=fn/(tp+fn) if tp+fn else None)

def main(root, plan, task, output_name='timing_BA_report.json'):
    state=json.loads((root/'controller_state.json').read_text())
    arms={k:Path(v) for k,v in state['completed_roots'].items()}
    passive=json.loads((arms['passive']/'summary.json').read_text())['rows']
    assert len(passive)==plan['passive_raw_episodes']
    assert all(r['takeover'] is None and r['all_executed_expert_actions']==0 for r in passive)
    assert all(r['seed']==task['evaluation_seed']+i//2 for i,r in enumerate(passive))
    labels=[not r['strict_success'] for r in passive]
    baseline_times=[next((q['step'] for q in r['queries'] if q['baseline_stop']),None) for r in passive]
    baseline_alarms=[t is not None for t in baseline_times]
    observations=[]
    for r in passive:
        with np.load(episode_path(arms['passive'],r)/'trace.npz') as d:
            observations.append(d['query_features'].copy())
    report={'scope':'no_training_no_post_SFT_SR','baseline':metrics(labels,baseline_alarms),
            'evaluation_seeds':[r['seed'] for r in passive], 'milestones':{},'actual_timing':{}}
    summaries={name:json.loads((arms[name]/'summary.json').read_text()) for name in plan['arms']}
    for name,s in summaries.items():
        p=json.loads((arms[name]/'provenance.json').read_text());cfg=p['feedback']
        cal=json.loads((Path(p['calibration'])/'calibration.json').read_text())
        z=np.load(Path(p['calibration'])/'gate_arrays.npz')
        report['milestones'][name]={}
        for milestone in plan['milestones']:
            seen=0;prefix=[]
            for r in s['rows']:
                prefix.append(r);seen+=int(r['accepted'])
                if seen>=milestone:break
            if seen<milestone:
                report['milestones'][name][str(milestone)]={'status':'NOT_REACHED','accepted':seen};continue
            gate=TimingFeedbackGate(cal['baseline_threshold'],z['center'],cal['scale'],cal['radius']*cfg['radius_multiplier'],
                regularization=cfg['lambda'],strength=cfg['beta'],min_support=cfg['minimum_interventions'],
                min_vote=cfg['minimum_absolute_vote'],block=cfg['execution_block'],enabled=name!='fixed',
                support_mode=cfg['support_mode'],max_wait_blocks=cfg.get('max_wait_blocks',1),
                remember_deferred_alarm=cfg.get('remember_deferred_alarm',False),use_later_duration=cfg.get('use_later_duration',False),
                max_feedback_events=cfg.get('max_feedback_events'))
            restore_memory(gate,prefix)
            times=[];changes=[]
            for i,(r,features) in enumerate(zip(passive,observations)):
                gate.begin_episode(1000000+i);first=None
                for q,f in zip(r['queries'],features):
                    decision=gate.query(f,q['score'],q['step'])
                    changes.append(decision['threshold']/cal['baseline_threshold']-1)
                    if decision['stop']:
                        first=q['step'];break
                times.append(first)
            alarms=[t is not None for t in times]
            result=metrics(labels,alarms)
            shifts=[b-a for a,b in zip(baseline_times,times) if a is not None and b is not None]
            # Resample complete reset-seed clusters, preserving paired ID/OOD.
            rng=np.random.default_rng(67031);y=np.asarray(labels);base=np.asarray(baseline_alarms);new=np.asarray(alarms)
            delta=[]
            if y.any() and (~y).any():
                seeds=sorted({r['seed'] for r in passive})
                clusters=[np.array([i for i,r in enumerate(passive) if r['seed']==seed]) for seed in seeds]
                for _ in range(2000):
                    ix=np.concatenate([clusters[j] for j in rng.integers(0,len(clusters),len(clusters))])
                    if y[ix].any() and (~y[ix]).any():
                        delta.append(metrics(y[ix],new[ix])['BA']-metrics(y[ix],base[ix])['BA'])
            result.update(status='COMPLETE',raw_attempts=len(prefix),accepted=seen,feedback_events=len(gate.memory),
                actual_expert_cost=sum(r['all_executed_expert_actions'] for r in prefix),
                passive_first_alarms=times,passive_shifts=shifts,
                shift_at_least10_count=sum(abs(x)>=10 for x in shifts),
                both_alarm_count=len(shifts),total_test_episodes=len(passive),
                lost_alarms=sum(a is not None and b is None for a,b in zip(baseline_times,times)),
                added_alarms=sum(a is None and b is not None for a,b in zip(baseline_times,times)),
                BA_delta_CI95=np.quantile(delta,[.025,.975]).tolist() if delta else None,
                threshold_relative_min=float(min(changes)),threshold_relative_max=float(max(changes)),
                interpretation='Passive counterfactual alarms; CI conditional on one learned memory, clustered by reset seed. No BA-based tuning.')
            report['milestones'][name][str(milestone)]=result
    for name in [a for a in summaries if a!='fixed']:
        pairs=[]
        for a,b in zip(summaries['fixed']['rows'],summaries[name]['rows']):
            assert (a['seed'],a['split'])==(b['seed'],b['split'])
            common=min(a['takeover'] if a['takeover'] is not None else a['total_retained_path_actions'],
                       b['takeover'] if b['takeover'] is not None else b['total_retained_path_actions'])
            with np.load(episode_path(arms['fixed'],a)/'trace.npz') as x,np.load(episode_path(arms[name],b)/'trace.npz') as y:
                diff=float(np.max(abs(x['actions'][:common]-y['actions'][:common]))) if common else 0.
                assert diff<1e-5,(name,a['episode'],diff)
            pairs.append(dict(episode=a['episode'],seed=a['seed'],split=a['split'],fixed=a['takeover'],feedback=b['takeover'],
                              shift=b['takeover']-a['takeover'] if a['takeover'] is not None and b['takeover'] is not None else None,
                              prefix_action_max_difference=diff,fixed_cost=a['all_executed_expert_actions'],feedback_cost=b['all_executed_expert_actions']))
        report['actual_timing'][name]={'common_raw_prefix_pairs':pairs,
            'unmatched_fixed_attempts':max(0,len(summaries['fixed']['rows'])-len(pairs)),
            'unmatched_feedback_attempts':max(0,len(summaries[name]['rows'])-len(pairs))}
    (root/output_name).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'status':'TIMING_BA_COMPLETE','baseline':report['baseline']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--task',required=True)
    p.add_argument('--output-name',default='timing_BA_report.json')
    a=p.parse_args();plan=json.loads(a.plan.read_text());main(a.root,plan,next(t for t in plan['task_settings'] if t['name']==a.task),a.output_name)
