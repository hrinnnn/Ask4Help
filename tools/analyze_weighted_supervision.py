"""Weighted utility exploration on only the three user-specified pi0.5 cohorts."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'artifacts/weighted_supervision_20260910';O.mkdir(parents=True,exist_ok=True)
rows=[]
for task in ['stackcube','airplane','object']:
 d=json.loads((R/f'artifacts/ood_supervision_exploration_20260906/full_path_{task}.json').read_text())
 refs={r['seed']:r['length'] for r in d['rows'] if r['method']=='offline_oracle'}
 for method,s in d['summaries'].items():
  rr=[r for r in d['rows'] if r['method']==method and r['selected']];n=s['N'];op=sum(r['length'] for r in rr)
  severity=0.;linear=0.;extra=0.;desc=[]
  for r in rr:
   e=np.maximum(np.asarray(r['distance'])/r['threshold']-1.,0.)
   length_excess=max(r['length']-refs[r['reference_seed']],0)
   severity+=float(np.sum(e**2));linear+=float(e.sum());extra+=length_excess
   desc.append(dict(seed=r['seed'],length=r['length'],reference_length=refs[r['reference_seed']],excess_steps=length_excess,squared_severity_sum=float(np.sum(e**2))))
  assert op==s['OOD_points'] and op<=n
  rows.append(dict(task=task,method=method,SR=s['SR'],N=n,ID_points=n-op,OOD_points=op,P=op/n,L=extra/n,R=severity/n,R_linear=linear/n,episodes=desc))
rank=lambda x:np.array([1+sum(v<t for v in x)+(sum(v==t for v in x)-1)/2 for t in x])
trials=[]
for a in [0.,.25,1.]:
 for b in [.25,1.,4.]:
  values=[dict(task=r['task'],method=r['method'],score=r['P']-a*r['L']-b*r['R']) for r in rows]
  corr={}
  for task in ['stackcube','airplane','object']:
   ii=[i for i,r in enumerate(rows) if r['task']==task];corr[task]=float(np.corrcoef(rank([values[i]['score'] for i in ii]),rank([rows[i]['SR'] for i in ii]))[0,1])
  trials.append(dict(length_weight=a,recovery_weight=b,values=values,spearman=corr,mean_spearman=float(np.mean(list(corr.values())))))
out=dict(status='POSTHOC_WEIGHTED_SCORE_EXPLORATION_NOT_VALIDATED_METRIC',formula='S=P-alpha*L-beta*R; P=allOODpoints/N_allIDandOOD; L=sum positive(episode_length-reference_length)/N; R=sumOODpoints max(distance/original_calibrated_radius-1,0)^2/N',parameter_rule='fixed alpha in[0,.25,1],beta in[.25,1,4] before this computation; report all, noSR-fit coefficients',scope='Three user-specified pi0.5 cohorts only; no invalidStage2 orTimingSR',limitations=['AllOODpoints are credited optimistically; not proven capability-gap supervision.','Deviation is geometry-only and is not a verified recovery label.','Squaredseverity can be dominated by outliers; it encodes a proposed stronger penalty, not measured harm.','Length excess relative to complete successfulreference; does not identify mastered prefixes or all redundant suffix motions.','ID contributes zero positive credit but remains in all-costdenominator.','A signed value is an evaluation score, not directly a nonnegative trainingweight.','AllSRknown: illustrative weight sensitivity, not predictive validation.'],components=rows,trials=trials)
(O/'results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
for r in rows:
 print(r['task'],r['method'],'P,L,R',*[round(r[x],5) for x in ['P','L','R']],'SR',r['SR'])
for t in trials:print('weights',t['length_weight'],t['recovery_weight'],t['spearman'])
