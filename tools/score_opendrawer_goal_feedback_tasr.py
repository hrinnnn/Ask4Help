"""Fixed nominal-only Goal target calibration and later paired TASR scoring."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_open_drawer_target_ratio import compare_block,MIN_RADIUS
from collect_opendrawer_goal_tasr_reference import goal_blocks
from score_pi05_feedback_tasr import budget_selection
from pi05_feedback_artifacts import episode_path

PHASES=['transport','place','release']


def nominal_bank(root):
    summary=json.loads((root/'summary.json').read_text())
    assert len(summary['rows'])==30
    groups={p:[] for p in ['reference','calibration','checking']}
    for i,row in enumerate(summary['rows']):
        assert row['seed']==1786000+i
        data=np.load(root/f'episode_{i:04d}/trace.npz')
        n=len(data['actions']);assert n==row['expert_actions']<=400
        assert len(data['main'])==len(data['wrist'])==len(data['qpos'])==n+1
        assert np.allclose(data['object_p'][0],row['reset']['object_pose']['p'],atol=1e-6)
        assert np.allclose(data['target_p'][0],row['reset']['target_pose']['p'],atol=1e-6)
        assert bool(data['success'][-1])==row['accepted']
        if not row['accepted']:continue
        assert row['blocks']==goal_blocks(row['expert_report'],n)
        pose=dict(np.load(root/f'episode_{i:04d}/comparison_pose.npz'));contact=pose.pop('contact')
        assert np.allclose(pose['position'],data['tcp']-data['target_p'],atol=1e-6)
        assert np.allclose(pose['quaternion'],data['tcp_q'],atol=1e-6)
        assert np.allclose(pose['width'].ravel(),data['qpos'][:,-2:].sum(1),atol=1e-6)
        assert np.array_equal(contact,data['grasped'])
        groups[row['partition']].append({'seed':row['seed'],'blocks':row['blocks'],
              'comparison_pose':pose,'rows':[{'object_grasped':bool(c)} for c in contact]})
    assert len(groups['reference'])>=8 and len(groups['calibration'])>=3 and len(groups['checking'])>=3
    return groups


def nearest(query,bank):
    candidates=[]
    for ref in bank:
        assert query['seed']!=ref['seed']
        parts={p:compare_block(query,ref,p) for p in PHASES}
        costs=[np.mean(x['distance'])+1000*(1-np.mean(x['contact'])) for x in parts.values() if x['indices']]
        candidates.append((float(np.mean(costs)) if costs else 0.,ref['seed'],parts))
    return min(candidates,key=lambda x:x[0])


def counts(parts,limits):
    return {str(f):int(sum(sum(ok and distance<=limits[p]*f for distance,ok in zip(part['distance'],part['contact'])) for p,part in parts.items())) for f in [1,2,3]}


def calibrate(root):
    groups=nominal_bank(root);samples={p:[] for p in PHASES};pairs=[]
    for query in groups['calibration']:
        _,ref,parts=nearest(query,groups['reference']);pairs.append([query['seed'],ref])
        for p,part in parts.items():samples[p].extend(d for d,c in zip(part['distance'],part['contact']) if c)
    assert all(samples.values()),'A phase has no contact-compatible nominal calibration observations'
    limits={p:max(MIN_RADIUS,float(np.quantile(values,.925))) for p,values in samples.items()}
    checks=[]
    for query in groups['checking']:
        _,ref,parts=nearest(query,groups['reference'])
        checks.append({'seed':query['seed'],'reference_seed':ref,'target_points':sum(len(x['indices']) for x in parts.values()),
                       'compatible_points':counts(parts,limits)})
    result={'format':'Goal_OOD_nominal_TASR_calibration_v1','target_phases':PHASES,'coordinate_frame':'target_tray_relative',
            'units':{'position_m':.02,'rotation_deg':15,'gripper_m':.01},'quantile':.925,'minimum_radius':MIN_RADIUS,
            'limits':limits,'partitions':{p:[q['seed'] for q in group] for p,group in groups.items()},
            'calibration_pairs':pairs,'nominal_checks':checks,'used_by_online_gate':False,'collection_stream_results_used':False}
    (root/'metric_calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (root/'NOMINAL_REFERENCE_AUDITED.json').write_text(json.dumps({'status':'PASS','counts':{p:len(g) for p,g in groups.items()},'limits':limits},indent=2))
    print(json.dumps(result))


def score(root,bank_root):
    groups=nominal_bank(bank_root);cal=json.loads((bank_root/'metric_calibration.json').read_text());limits=cal['limits'];arms={}
    for arm in ['fixed','feedback']:
        summary=json.loads((root/arm/'summary.json').read_text());assert summary['task']=='open_drawer_goal_ood';rows=[]
        for row in summary['rows']:
            if not row['accepted']:continue
            n=row['expert_suffix_actions'];take=row['takeover'];i=row['episode']
            out={'arm':arm,'episode':i,'seed':row['seed'],'split':row['split'],'expert_points':n}
            if row['split']=='id':out.update(compatible_points={str(f):0 for f in [1,2,3]},coverage='ID_excluded_from_numerator')
            else:
                d=np.load(episode_path(root/arm,row)/'trace.npz');blocks=goal_blocks(row['expert_report'],n)
                blocks={p:[a+take,b+take] for p,(a,b) in blocks.items()}
                query={'seed':row['seed'],'blocks':blocks,'comparison_pose':{'position':d['tcp']-d['target_p'],
                    'quaternion':d['tcp_q'],'width':d['qpos'][:,-2:].sum(1)[:,None]},
                    'rows':[{'object_grasped':bool(g)} for g in d['grasped']]}
                _,ref,parts=nearest(query,groups['reference'])
                out.update(compatible_points=counts(parts,limits),coverage='COMPLETE',blocks=blocks,reference_seed=ref,parts=parts)
            rows.append(out)
        arms[arm]=rows
    budget,selected=budget_selection(arms);results={};all_results={}
    def aggregate(rows):
        denominator=sum(r['expert_points'] for r in rows)
        numerators={str(f):sum(r['compatible_points'][str(f)] for r in rows) for f in [1,2,3]}
        return {'expert_action_budget':denominator,'selected_episodes':[r['episode'] for r in rows],
                'OOD_expert_points':sum(r['expert_points'] for r in rows if r['split']=='ood'),
                'compatible_points':numerators,'TASR':{f:n/denominator if denominator else None for f,n in numerators.items()}}
    for arm,rows in arms.items():
        results[arm]=aggregate([r for r in rows if r['episode'] in selected[arm]])
        assert results[arm]['expert_action_budget']==budget
        all_results[arm]=aggregate(rows)
    out={'status':'PILOT_TASR_COMPLETE' if budget else 'NO_COMMON_POSITIVE_BUDGET','task':'open_drawer_goal_ood',
         'nominal_reference':str(bank_root),'limits':limits,'results':results,'all_accepted_unmatched':all_results,'rows':arms,
         'interpretation':'Expert-data evaluation before SFT, not post-training success or an online best-time label.'}
    (root/'pilot_TASR.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'status':out['status'],'results':results}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--root',type=Path)
    p.add_argument('--calibrate',action='store_true');args=p.parse_args()
    if args.calibrate:calibrate(args.bank)
    elif args.root is None:p.error('--root required for paired collection scoring')
    else:score(args.root,args.bank)
