"""Apply already-declared weighted scores to user-approved six-timing OpenDrawer."""
import json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'artifacts/opendrawer_weighted_validation_20260910';O.mkdir(parents=True,exist_ok=True)
load=lambda p:json.loads((R/p).read_text())
source='outputs/opendrawer_timing_figure_20260910/source/'
f=load(source+'final_report.json');a=load('artifacts/open_drawer_target_ratio_20260831/analysis.json');w=load('artifacts/ood_supervision_exploration_20260906/full_path_opendrawer.json')
assert f['budget_manifest']==a['budget_manifest']
refs={r['seed']:r['length'] for r in w['rows'] if r['method']=='t0'}
points=[];rows=[]
for anchor in [0,50,80,120,160,220]:
 selected=[r for r in a['rows'] if r['anchor']==anchor and r['accepted'] and r['accepted_index'] in a['budget_manifest']['selected_source_episode_indices'][f'anchor_{anchor}']]
 actual={r['seed']:r for r in selected};full=[r for r in w['rows'] if r['method']==f't{anchor}' and r['selected']];assert {r['seed'] for r in full}==set(actual)
 n=sum(r['length'] for r in full);assert n==2413
 severity=0.;length=0.;target=0;target3=0;prefix=0;target_excess=0
 for r in full:
  src=actual[r['seed']];assert r['length']==src['expert_anchors'];e=np.maximum(np.array(r['distance'])/r['threshold']-1.,0)
  length+=max(r['length']-refs[r['reference_seed']],0);severity+=float(np.sum(e**2))
  prefix+=src['blocks']['rotation_pregrasp'][0]-src['takeover']
  for phase,(lo,hi) in src['blocks'].items():
   x=np.array(src['distance'][lo-src['takeover']:hi-src['takeover']]);assert np.isfinite(x).all();z=x/src['thresholds'][phase];target+=len(x);target3+=int(np.sum(z<=3));target_excess+=float(np.sum(np.maximum(z-1.,0)**2))
  points.append(dict(anchor=anchor,seed=r['seed'],N=r['length'],length_excess=max(r['length']-refs[r['reference_seed']],0),squared_severity=float(np.sum(e**2))))
 ev=load(source+f'eval_anchor_{anchor}.json');assert ev['episodes']==len(ev['rows'])==100;successes=sum(int(r['success']) for r in ev['rows']);assert successes==ev['successes']==round(ev['success_rate']*100)
 row=dict(anchor=anchor,N=n,ID_points=0,episodes=len(selected),SR=successes/100,successes=successes,eval_count=100,eval_seed_start=ev['seed_start'],P=1.,L=length/n,R=severity/n,target_share=target/n,target_TASR3=target3/n,prefix_share=prefix/n,target_R=target_excess/n,full1=w['summaries'][f't{anchor}']['hard']['1'],full3=w['summaries'][f't{anchor}']['hard']['3'],full6=w['summaries'][f't{anchor}']['hard']['6'])
 row['weighted_main']=1-.25*row['L']-row['R'];rows.append(row)
rank=lambda x:np.array([1+sum(v<t for v in x)+(sum(v==t for v in x)-1)/2 for t in x])
def rho(x,y):return None if len(set(x))<2 else float(np.corrcoef(rank(x),rank(y))[0,1])
y=[r['SR'] for r in rows];metrics={k:dict(spearman=rho([r[k] for r in rows],y),scores=[r[k] for r in rows]) for k in ['target_TASR3','full1','full3','full6','weighted_main']}
trials=[]
for alpha in [0.,.25,1.]:
 for beta in [.25,1.,4.]:
  v=[r['P']-alpha*r['L']-beta*r['R'] for r in rows];trials.append(dict(alpha=alpha,beta=beta,score=v,spearman=rho(v,y)))
# Evaluation-only uncertainty: independent within-arm outcome resampling, no
# paired-seed claim; no training- or data-collection uncertainty represented.
rng=np.random.default_rng(310911);boot=rng.binomial(100,np.array(y),size=(5000,6))/100
for k in metrics:
 rs=[rho(metrics[k]['scores'],b.tolist()) for b in boot];rs=[v for v in rs if v is not None];metrics[k]['eval_only_bootstrap_95']=np.quantile(rs,[.025,.975]).tolist()
result=dict(status='FIXED_FORMULA_TRANSFER_CHECK_ON_USER_APPROVED_OPENDRAWER',protocol=f['protocol'],primary_formula='S=P-.25L-1R, exactly prior3task definition; no coefficient selected from DrawerSR',rows=rows,metrics=metrics,all_prior_weight_settings=trials,per_episode=points,source_final=source+'final_report.json',outcome_audit='600rawlocalOODrows recounted; identicalbudget_manifest and exactselectedseedmembership',limitations=['Previously known SR and inspected trajectories; retrospective check, not prospective validation.','One trainingseed,6different testseedblocks; uncertaintyinterval is evaluation samplingonly.','Fullpathgeometry proxy is not independently verified Recovery label.','Length excess compares to complete nominalreference; early normal prefix may escape this penalty.','Target_TASR3 recomputed from storedobjectrelativegeometry; source historical contactaudit saysselectedtargetcontactnotblocked; no newRGB/state extraction.','No SCStage2invalidSR or old20episodeDrawerSR used.'])
(O/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
print('ROWS')
for r in rows:print(r)
print('METRICS',metrics);print('WEIGHTS',trials)
