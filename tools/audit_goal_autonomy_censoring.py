"""Compare actual unassisted stage events with the old gate boundary, not post-SFT SR."""
import argparse,json
from pathlib import Path
import numpy as np
from pi05_feedback_artifacts import episode_path


def first(x):
    where=np.flatnonzero(x);return int(where[0]) if len(where) else None


def run(a):
    fixed=json.loads((a.gated_root/'goal/fixed/summary.json').read_text())['rows'];out={}
    for split in ['id','ood']:
        root=a.root/split;summary=json.loads((root/'summary.json').read_text())
        provenance=json.loads((root/'provenance.json').read_text())
        assert provenance['purpose']=='diagnostic_autonomous' and provenance['runtime']['inference_mode']=='eval'
        assert len(summary['rows'])==20 and summary['complete'] and (root/'EVALUATION_COMPLETE.json').exists()
        rows=[]
        for i,r in enumerate(summary['rows']):
            g=fixed[2*i+(split=='ood')];assert r['seed']==g['seed']==1785000+i
            with np.load(episode_path(root,r)/'trace.npz') as d, np.load(episode_path(a.gated_root/'goal/fixed',g)/'trace.npz') as old:
                n=len(d['actions']);assert n==r['actions']<=400 and len(d['qpos'])==n+1
                assert len(d['all_expert_actions'])==0 and bool(d['eval_strict_success'][-1])==r['strict_success']
                t=g['takeover'];common=t if t is not None else min(n,len(old['actions']))
                error=float(np.max(abs(d['actions'][:common]-old['actions'][:common]))) if common else 0.
                assert error<1e-5,('autonomous replay prefix mismatch',i,error)
                grasp=first(d['grasped']);lift=first(d['ever_lifted'])
                rows.append({'episode':i,'seed':r['seed'],'baseline_takeover':t,'first_autonomous_grasp':grasp,
                    'first_autonomous_lift':lift,'strict_success':r['strict_success'],'prefix_action_max_difference':error,
                    'grasp_after_gate':t is not None and grasp is not None and grasp>t,
                    'grasp_beyond_one_block_delay':t is not None and grasp is not None and grasp>t+5})
        out[split]={'episodes':20,'autonomous_grasped':sum(r['first_autonomous_grasp'] is not None for r in rows),
                    'autonomous_lifted':sum(r['first_autonomous_lift'] is not None for r in rows),
                    'strict_successes':sum(r['strict_success'] for r in rows),
                    'grasp_after_baseline_gate':sum(r['grasp_after_gate'] for r in rows),
                    'grasp_beyond_one_block_delay':sum(r['grasp_beyond_one_block_delay'] for r in rows),'rows':rows}
    result={'status':'DEVELOPMENT_AUTONOMY_CENSORING_AUDIT','results':out,
            'scope':'Existing development seeds and frozen base policy, not independent final post-SFT SR. Event timing alone does not establish the SR-optimal takeover window.'}
    (a.root/'autonomy_censoring_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gated-root',type=Path,required=True)
    run(p.parse_args())
