"""Final independent counts, source agreement, and branch-array reconciliation."""
import json
from pathlib import Path
import numpy as np
root=Path('artifacts/expert_feedback_overnight_20260907');counts={};queries=0
expected={'development':240,'directional':80,'validation':240,'bounded_development':120,'bounded_validation':180,'longstream':180}
for stage,n in expected.items():
 files=list((root/stage).glob('seed_*/*/episode_*/result.json'));assert len(files)==n,(stage,len(files));counts[stage]=len(files)
 assert json.loads((root/stage/'development_audit.json').read_text())['status']=='SCALAR_CHRONOLOGY_AUDIT_PASS'
 for p in files:
  r=json.loads(p.read_text());a=np.load(p.parent/'trace.npz')
  assert len(a['actions'])==r['total_actions']==r['policy_steps']+r['expert_actions']
  assert len(a['qpos'])==len(a['tcp'])==len(a['actions'])+1
  queries+=len(r['queries'])
  if r['feedback_config'].get('max_feedback_wait_blocks') is not None:
   suppressed=[q for q in r['queries'] if q['baseline_stop'] and not q['stop']]
   if suppressed and r['takeover'] is not None:assert r['takeover']<=suppressed[0]['step']+5
common={}
for stage in ['development','validation','bounded_development','bounded_validation','longstream']:
 d=json.loads((root/stage/'common_preference_windows.json').read_text());assert all(r['max_state_error']<1e-4 for r in d['rows']);common[stage]=len(d['rows'])
cf=root/'counterfactual_window_retry1';r=json.loads((cf/'summary.json').read_text())['rows'];assert len(r)==35
best={}
for seed in sorted({x['seed'] for x in r}):
 sub=[x for x in r if x['seed']==seed];assert sorted(x['takeover'] for x in sub)==[0,10,20,30,40,60,80]
 costs=[x['expert_actions'] for x in sub if x['success']];minimum=min(costs)
 best[seed]=dict(min_expert_actions=minimum,best_sampled_times=[x['takeover'] for x in sub if x['success'] and x['expert_actions']==minimum])
 for x in sub:
  a=np.load(cf/f'seed_{seed}'/f'takeover_{x["takeover"]}'/'expert.npz');assert len(a['actions'])==x['expert_actions'];assert len(a['qpos'])==len(a['actions'])+1
out=dict(status='COMPLETE_LOCAL_EVIDENCE_RECONCILIATION_PASS',new_collection_episodes=sum(counts.values()),stage_counts=counts,policy_queries=queries,common_feedback_segments=common,counterfactual_branches=35,branch_cost_minima=best,limitation='No new SFT. Preferred-proxy checks are not independently annotated learning-optimal timing. Existing audit files cover scalar chronology; final recount also checks saved arrays and bounded waiting.')
Path('outputs/expert_feedback_overnight_20260908/completion_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
