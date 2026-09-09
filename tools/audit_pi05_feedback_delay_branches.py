"""Describe actual delayed policy/expert branches and stage reachability, without retuning."""
import argparse,json
from pathlib import Path
import numpy as np
from pi05_feedback_artifacts import episode_path


def first_true(values):
    indices=np.flatnonzero(values)
    return int(indices[0]) if len(indices) else None


def run(a):
    result={}
    for task in ['grasp','goal']:
        sources={arm:json.loads((a.root/task/arm/'summary.json').read_text()) for arm in ['fixed','feedback']}
        costs={};funnel={}
        for arm,s in sources.items():
            costs[arm]={'successful_expert_actions':sum(r['all_executed_expert_actions'] for r in s['rows'] if r['accepted']),
                        'unsuccessful_expert_actions':sum(r['all_executed_expert_actions'] for r in s['rows'] if not r['accepted'])}
            rows=[]
            for r in s['rows']:
                if r['split']!='ood':continue
                with np.load(episode_path(a.root/task/arm,r)/'trace.npz') as d:
                    t=r['takeover'] if r['takeover'] is not None else len(d['actions'])
                    rows.append({'episode':r['episode'],'takeover':r['takeover'],
                        'opened_by_policy':bool(d['ever_drawer_opened'][:t+1].any()),
                        'grasped_by_policy':bool(d['grasped'][:t+1].any()),
                        'lifted_by_policy':bool(d['ever_lifted'][:t+1].any()),
                        'holding_at_boundary':bool(d['grasped'][t]),
                        'first_policy_open':first_true(d['ever_drawer_opened'][:t+1]),
                        'first_policy_grasp':first_true(d['grasped'][:t+1]),
                        'first_policy_lift':first_true(d['ever_lifted'][:t+1])})
            funnel[arm]={'OOD_episodes':len(rows),
                'opened_before_expert_or_end':sum(r['opened_by_policy'] for r in rows),
                'grasped_before_expert_or_end':sum(r['grasped_by_policy'] for r in rows),
                'lifted_before_expert_or_end':sum(r['lifted_by_policy'] for r in rows),'rows':rows}
        branches=[]
        for f,s in zip(sources['fixed']['rows'],sources['feedback']['rows']):
            if f['takeover'] is None or s['takeover'] is None:continue
            t=f['takeover'];delay=s['takeover']-t
            if delay<=0:continue
            with np.load(episode_path(a.root/task/'fixed',f)/'trace.npz') as x, np.load(episode_path(a.root/task/'feedback',s)/'trace.npz') as y:
                n=min(delay,f['expert_suffix_actions']);assert n==delay
                start_error=max(float(np.max(abs(x[k][t]-y[k][t]))) for k in ['qpos','tcp','object_p','object_q','drawer_qpos','target_p'])
                assert start_error<1e-5
                q1=x['tcp_q'][t+n].astype(float);q2=y['tcp_q'][t+n].astype(float)
                q1/=np.linalg.norm(q1);q2/=np.linalg.norm(q2)
                branches.append({'episode':f['episode'],'split':f['split'],'delay':delay,
                    'both_successful':f['strict_success'] and s['strict_success'],
                    'start_snapshot_max_difference':start_error,
                    'actual_action_MSE':float(np.mean((x['actions'][t:t+n].astype(float)-y['actions'][t:t+n])**2)),
                    'endpoint_TCP_difference_mm':float(np.linalg.norm(x['tcp'][t+n]-y['tcp'][t+n])*1000),
                    'endpoint_rotation_difference_deg':float(np.rad2deg(2*np.arccos(np.clip(abs(q1@q2),0,1)))),
                    'endpoint_width_difference_mm':float(abs(x['qpos'][t+n,-2:].sum()-y['qpos'][t+n,-2:].sum())*1000),
                    'fixed_expert_cost':f['all_executed_expert_actions'],'soft_expert_cost':s['all_executed_expert_actions']})
        result[task]={'cost_decomposition':costs,'policy_stage_funnel':funnel,'actual_delayed_branches':branches}
    output={'status':'EXPLORATORY_BRANCH_MEASUREMENT','results':result,
            'interpretation':'These are actual shared-prefix branch measurements. No threshold changed, no post-SFT utility inferred, and failed-segment cost savings are not successful-data savings.'}
    (a.root/'delay_branch_audit.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args())
