"""Score fresh Grasp-OOD suffixes against the unchanged historical TASR bank."""
import argparse,io,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from analyze_open_drawer_target_ratio import relative_pose,stage_blocks,compare_block,PHASES
from score_pi05_feedback_tasr import budget_selection


def parts_against_reference(query,bank):
    candidates=[]
    for ref in bank:
        if ref['seed']==query['seed']:continue
        parts={p:compare_block(query,ref,p) for p in PHASES}
        costs=[np.mean(x['distance'])+1000*(1-np.mean(x['contact'])) for x in parts.values() if x['indices']]
        candidates.append((float(np.mean(costs)) if costs else 0.,ref['seed'],parts))
    return min(candidates,key=lambda x:x[0])


def run(args):
    with ZipFile(args.reference) as bundle:
        metadata=json.loads(bundle.read('manifest.json'));bank=[];checks=[]
        for r in metadata['references']:
            values=dict(np.load(io.BytesIO(bundle.read(r['array']))))
            contact=values.pop('contact')
            item={'seed':r['seed'],'blocks':r['blocks'],'comparison_pose':values,
                  'rows':[{'object_grasped':bool(x)} for x in contact]}
            if r['is_reference']:bank.append(item)
            else:checks.append((r,item))
    limits=metadata['protocol']['thresholds']
    for original,item in checks:
        _,_,parts=parts_against_reference(item,bank)
        for p,part in parts.items():
            count=sum(ok and distance<=limits[p] for distance,ok in zip(part['distance'],part['contact']))
            assert count==original['old_phase_stats'][p]['compatible'],(item['seed'],p,count)
    if args.reference_check_only:
        print(json.dumps({'status':'HISTORICAL_REFERENCE_REGRESSION_PASS','references':len(bank),'checks':len(checks)}))
        return
    arms={};gaps=[]
    for arm in ['fixed','feedback']:
        summary=json.loads((args.root/arm/'summary.json').read_text())
        assert summary['task']=='open_drawer_grasp_ood','Do not apply a grasp target metric to Goal-OOD'
        rows=[]
        for row in summary['rows']:
            if not row['accepted']:continue
            n=row['expert_suffix_actions'];take=row['takeover'];i=row['episode']
            result={'arm':arm,'episode':i,'seed':row['seed'],'split':row['split'],'expert_points':n}
            if row['split']=='id':
                result.update(compatible_points={str(f):0 for f in [1,2,3]},coverage='ID_excluded_from_numerator')
                rows.append(result);continue
            data=np.load(args.root/arm/f'episode_{i:04d}/trace.npz')
            # The endpoint may cut final release/settle actions at true success.
            # Preserve the earlier completed target-stage counts unchanged.
            meta={'actual_takeover_step':take,'expert_action_steps':n,'expert_result':row['expert_report'],'accepted':False}
            blocks,close_end=stage_blocks(meta)
            if blocks and close_end is not None:
                assert all(data['grasped'][close_end-2:close_end+1])
                assert not data['ever_lifted'][close_end]
            state_rows=[{'object_position':p,'object_quaternion':q,'object_grasped':bool(g)}
                        for p,q,g in zip(data['object_p'],data['object_q'],data['grasped'])]
            pose={'position':data['tcp'],'quaternion':data['tcp_q'],'width':data['qpos'][:,-2:].sum(1)[:,None]}
            query={'seed':row['seed'],'blocks':blocks,'comparison_pose':relative_pose(pose,state_rows),'rows':state_rows}
            _,ref_seed,parts=parts_against_reference(query,bank)
            counts={str(f):int(sum(sum(ok and distance<=limits[p]*f for distance,ok in zip(part['distance'],part['contact'])) for p,part in parts.items())) for f in [1,2,3]}
            result.update(compatible_points=counts,coverage='COMPLETE',blocks=blocks,reference_seed=ref_seed,parts=parts)
            rows.append(result)
        arms[arm]=rows
    budget,selected=budget_selection(arms);results={};all_results={}
    for arm,rows in arms.items():
        def aggregate(chosen):
            denominator=sum(r['expert_points'] for r in chosen)
            counts={str(f):sum(r['compatible_points'][str(f)] for r in chosen) for f in [1,2,3]}
            return {'expert_action_budget':denominator,'selected_episodes':[r['episode'] for r in chosen],
                    'OOD_expert_points':sum(r['expert_points'] for r in chosen if r['split']=='ood'),
                    'compatible_points':counts,'TASR':{f:n/denominator if denominator else None for f,n in counts.items()}}
        results[arm]=aggregate([r for r in rows if r['episode'] in selected[arm]])
        assert results[arm]['expert_action_budget']==budget
        all_results[arm]=aggregate(rows)
    out={'status':'PILOT_TASR_COMPLETE' if budget else 'NO_COMMON_POSITIVE_BUDGET',
         'task':'open_drawer_grasp_ood','metric_reference':str(args.reference),'frozen_limits':limits,
         'old_check_episode_count':len(checks),'results':results,'all_accepted_unmatched':all_results,'rows':arms,
         'interpretation':'Pretraining TASR with frozen historical reference. Unmatched corpus totals are descriptive; no post-SFT SR claim.'}
    (args.root/'pilot_TASR.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({'status':out['status'],'results':results}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path)
    p.add_argument('--reference-check-only',action='store_true')
    p.add_argument('--reference',type=Path,required=True);args=p.parse_args()
    if not args.reference_check_only and args.root is None:p.error('--root required for collection scoring')
    run(args)
