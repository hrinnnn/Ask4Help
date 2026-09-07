"""Describe real collection behavior without treating own feedback as truth."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
rows=[];pairs=[]
for stream in sorted(args.root.glob('seed_*')):
 base_path=stream/'fixed/summary.json'
 if not base_path.exists():continue
 fixed=json.loads(base_path.read_text())['rows']
 for summary in sorted(stream.glob('*/summary.json')):
  data=json.loads(summary.read_text())['rows'];name=summary.parent.name
  for split in ['id','stage2_ood']:
   sub=[r for r in data if r['split']==split];times=[r['takeover'] for r in sub if r['takeover'] is not None];costs=[r['expert_actions'] for r in sub];loss0=[];events=[];censored=0
   for r in sub:
    f=r['feedback']
    if f:
     e=f.get('event',f.get('first_observed_high'));events.append(e);censored+=e is None
    path=summary.parent/f'episode_{r["episode"]:03d}'/'trace.npz';a=np.load(path)
    if 'expert_feedback_loss' in a:
     loss=a['expert_feedback_loss'];loss0.append(float(loss[0]))
   matched=[(r,b) for r,b in zip(data,fixed) if r['split']==split]
   deltas=[r['expert_actions']-b['expert_actions'] for r,b in matched]
   rows.append(dict(stream=stream.name,variant=name,split=split,n=len(sub),assisted_complete=sum(r['success'] for r in sub),total_expert_actions=sum(costs),median_expert_actions=float(np.median(costs)),takeovers=len(times),reset_takeovers=sum(t==0 for t in times),median_takeover=float(np.median(times)) if times else None,
    median_initial_free_error=float(np.median(loss0)) if loss0 else None,own_boundary_at0=sum(e==0 for e in events),own_boundary_ge_delta=sum(e is not None and e>=5 for e in events),own_boundary_censored=censored,
    median_undo=float(np.median([r['motion_undo'] for r in sub if r['motion_undo'] is not None])) if any(r['motion_undo'] is not None for r in sub) else None,
    undo_flags=sum(r['motion_undo'] is not None and r['motion_undo']>1.9004588285680322 for r in sub),cost_less=sum(d<0 for d in deltas),cost_more=sum(d>0 for d in deltas),cost_same=sum(d==0 for d in deltas),feedback_reasons=dict(Counter(r['feedback']['reason'] for r in sub if r['feedback']))))
   for r,b in matched:
    if r['takeover']!=b['takeover']:pairs.append(dict(stream=stream.name,variant=name,seed=r['seed'],split=split,fixed_takeover=b['takeover'],new_takeover=r['takeover'],fixed_cost=b['expert_actions'],new_cost=r['expert_actions'],fixed_success=b['success'],new_success=r['success'],fixed_undo=b['motion_undo'],new_undo=r['motion_undo'],path=str(summary.parent/f'episode_{r["episode"]:03d}')))
report=dict(rows=rows,changed_pairs=pairs,note='Own error boundary depends on calibration and is not independent favorite-time truth; costs include unsuccessful expert episodes. ID takeovers are requests, not calibrated failure false positives.')
(args.root/'behavior_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2))
