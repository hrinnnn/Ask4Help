"""Independent scalar/chronology audit of flexible-feedback grid outputs."""
import argparse,json,math
from pathlib import Path
from collections import Counter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);args=p.parse_args();grid=json.loads(args.grid.read_text());rows=[]
for offset in grid['seed_offsets']:
 fixed_path=args.root.parent/'development'/f'seed_{offset}'/'fixed/summary.json'
 fixed=json.loads(fixed_path.read_text())['rows'] if fixed_path.exists() else None
 for variant in grid['variants']:
  root=args.root/f'seed_{offset}'/variant['name'];summary=json.loads((root/'summary.json').read_text());records=summary['rows'];assert len(records)==grid.get('episodes',20)
  cal=json.loads(Path(records[0]['feedback_config']['calibration_path']).read_text());radius=cal['radius']*variant.get('radius',1);center=np.array(cal['center']);scale=cal['scale'];memory=[];committed=0;neutral=set()
  for r in records:
   deadline=None
   a=np.load(root/f'episode_{r["episode"]:03d}/trace.npz');features=a['policy_query_features'];scores=a['policy_query_scores'];assert len(r['queries'])==len(scores)
   assert len(a['actions'])==r['total_actions']==r['expert_actions']+r['policy_steps'];assert len(a['qpos'])==len(a['actions'])+1
   for x,score,q in zip(features,scores,r['queries']):
    assert q['memory_episodes']==committed
    z=(x-center)/scale;near=[m for m in memory if np.linalg.norm(m[0]-z)<=radius];threshold=cal['tau0'];rule=variant.get('rule','original')
    if variant.get('arm')!='fixed' and near:
     if rule in ['soft','kernel']:
      by={}
      for mz,ms,my,me in near:by.setdefault(me,[]).append((math.exp(-.5*(np.linalg.norm(mz-z)/radius)**2),0 if me in neutral else (1 if my else -1)))
      n=len(by)
      if n>=variant.get('min_support',2):
       values=[sum(w*y for w,y in b)/sum(w for w,y in b) for b in by.values()];effective=n
       if rule=='kernel':
        weights=np.array([max(w for w,y in b) for b in by.values()]);vote=np.average(values,weights=weights);effective=float(weights.sum())
       else:vote=np.mean(values)
       if abs(vote)>=.25:threshold=cal['tau0']*math.exp(-variant.get('strength',.5)*(effective/(effective+2))*vote)
     else:
      lower=max([m[1] for m in near if not m[2]],default=-np.inf);upper=min([m[1] for m in near if m[2]],default=np.inf)
      if np.isfinite(upper):upper=np.nextafter(upper,-np.inf)
      if lower<=upper:threshold=float(np.clip(cal['tau0'],lower,upper))
    if variant.get('arm')!='fixed' and variant.get('max_wait_blocks') is not None:
     if deadline is not None and q['step']>=deadline:threshold=min(threshold,float(np.nextafter(score,-np.inf)))
     elif deadline is None and score>cal['tau0'] and not score>threshold:deadline=q['step']+5*variant['max_wait_blocks']
     assert q['wait_deadline']==deadline
    assert np.isclose(q['threshold'],threshold,rtol=1e-12,atol=1e-12),(variant['name'],q,threshold)
    assert q['stop']==bool(score>q['threshold'])
   f=r['feedback']
   if f and variant.get('arm')!='fixed':
    if f.get('credit_step') is not None:
     ix=next(i for i,q in enumerate(r['queries']) if q['step']==f['credit_step']);memory.append(((features[ix]-center)/scale,float(scores[ix]),f['request'],r['episode']))
     if variant.get('rule')=='kernel' and f['reason']=='current_supervision_gap_no_claim_of_lateness':neutral.add(r['episode'])
    committed+=1
  if variant.get('arm')=='fixed':fixed=records
  if fixed:
   assert [r['reset_metadata'] for r in records]==[r['reset_metadata'] for r in fixed]
  for split in ['id','stage2_ood']:
   sub=[r for r in records if r['split']==split];queries=[q for r in sub for q in r['queries']];times=[r['takeover'] for r in sub if r['takeover'] is not None]
   rows.append(dict(seed_offset=offset,variant=variant['name'],split=split,n=len(sub),completed=sum(r['success'] for r in sub),expert_actions=sum(r['expert_actions'] for r in sub),takeovers=len(times),median_takeover=float(np.median(times)) if times else None,mean_takeover=float(np.mean(times)) if times else None,supported=sum(q['reason']=='supported' for q in queries),queries=len(queries),flips=sum(q['stop']!=q['baseline_stop'] for q in queries),wait_flips=sum(not q['stop'] and q['baseline_stop'] for q in queries),request_flips=sum(q['stop'] and not q['baseline_stop'] for q in queries),feedback=dict(Counter(r['feedback']['reason'] for r in sub if r['feedback'])),undo_flags=sum(r['motion_undo'] is not None and r['motion_undo']>1.9004588285680322 for r in sub),changed_pairs=sum(r['takeover']!=b['takeover'] for r,b in zip(records,fixed or records) if r['split']==split)))
report=dict(status='SCALAR_CHRONOLOGY_AUDIT_PASS',rows=rows,limitations='Exploratory grid; assisted completion is not post-SFT SR. Replay audit checks saved feedback and local decisions, not validity of preferred-time proxy.')
(args.root/'development_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
