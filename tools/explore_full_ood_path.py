"""Whole original FK paths: exploratory geometry proxy, no task-phase clipping."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr,spearmanr
from compute_pi05_tasr import fast_map,MIN_RADIUS,UNITS
from analyze_open_drawer_suffix_recovery import PandaFK,pose_channels

ROOT=Path('artifacts/ood_supervision_exploration_20260906');BASE=Path('artifacts/pi05_table_tasr_20260903')

def main():
 fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));all_results={}
 original=json.loads((BASE/'pi05_tasr_table_values.json').read_text())['rows']
 for task in ['stackcube','airplane','object','opendrawer']:
  data=[];groups={}
  if task=='opendrawer':
   a=json.loads(Path('artifacts/open_drawer_tolerance_sweep_20260903/factor_1p00/analysis.json').read_text());srmap={0:.25,50:.75,80:.8,120:.35,160:.4,220:.65}
   for r in a['rows']:
    if not r['accepted']:continue
    selected=r['accepted_index'] in a['budget_manifest']['selected_source_episode_indices'][f'anchor_{r["anchor"]}']
    if not selected and r['anchor']!=0:continue
    q=np.load(Path(r['directory'])/'states.npy')[r['takeover']:r['takeover']+r['expert_anchors']]
    data.append(dict(method=f't{r["anchor"]}',seed=r['seed'],selected=selected,qpos=q,reference=r['anchor']==0))
   for step,sr in srmap.items():groups[f't{step}']=dict(N=2413,SR=sr)
  else:
   a=json.loads((BASE/f'tasr_{task}.json').read_text())
   for r in a['rows']:
    if r['split']!='ood':continue
    data.append(dict(method=r['method'],seed=r['seed'],selected=r['selected'],qpos=np.load(r['arrays'])['qpos'],reference=r['method']=='offline_oracle'))
   for m,s in a['summary'].items():groups[m]=dict(N=s['total_expert_points'],SR=next(r['SR_percent']/100 for r in original if r['task']==task and r['method']==m))
  for r in data:
   p,R=fk.pose(r.pop('qpos'));q=Rotation.from_matrix(R).as_quat()[:,[3,0,1,2]]
   # Retrieve width from original qpos separately, never from reconstructed tail.
   if task=='opendrawer':
    raw=next(x for x in a['rows'] if x['seed']==r['seed']);states=np.load(Path(raw['directory'])/'states.npy')[raw['takeover']:raw['takeover']+raw['expert_anchors']]
   else:
    raw=next(x for x in a['rows'] if x['seed']==r['seed'] and x['method']==r['method'] and x['split']=='ood');states=np.load(raw['arrays'])['qpos']
   r['pose']=dict(position=p,quaternion=q,width=states[:,-2:].sum(1)[:,None]);r['length']=len(p);r['key']=(r['method'],r['seed'])
  refs=[r for r in data if r['reference'] and r['seed']%5<3];cal=[r for r in data if r['reference'] and r['seed']%5==3];checks=[r for r in data if r['reference'] and r['seed']%5==4]
  assert len(refs)>=3 and len(cal)>=3,(task,len(refs),len(cal))
  cache={}
  def match(e,r):
   key=e['key'],r['key']
   if key not in cache:
    distance=np.linalg.norm(pose_channels(e['pose'],r['pose'])/UNITS,axis=2);mapping=fast_map(distance);d=distance[np.arange(len(mapping)),mapping]
    cache[key]=(float(d.mean()),d,mapping)
   return cache[key]
  def choose(e,bank):return min((r for r in bank if r['seed']!=e['seed']),key=lambda r:match(e,r)[0])
  limits={}
  def threshold(skip):
   if skip not in limits:
    bank=[r for r in refs if r['seed']!=skip];vals=[]
    for e in cal:
     if e['seed']==skip:continue
     vals.extend(match(e,choose(e,bank))[1])
    limits[skip]=max(MIN_RADIUS,float(np.quantile(vals,.925)))
   return limits[skip]
  reference_seeds={r['seed'] for r in data if r['reference']};rows=[]
  for i,e in enumerate(data):
   ref=choose(e,refs);mean,dist,mapping=match(e,ref);tau=threshold(e['seed'] if e['seed'] in reference_seeds else None)
   rows.append(dict(method=e['method'],seed=e['seed'],selected=e['selected'],reference_seed=ref['seed'],length=e['length'],threshold=tau,
    distance=dist.tolist(),mapping=mapping.tolist(),compatible={str(f):int(np.sum(dist<=f*tau)) for f in [1,3,6]},soft={str(f):float(np.sum(np.exp(-.5*(dist/(f*tau))**2))) for f in [1,3,6]}))
   if i%50==0:print(task,'WHOLE_PATH',i,'/',len(data),flush=True)
  summaries={}
  for method,g in groups.items():
   chosen=[r for r in rows if r['selected'] and r['method']==method]
   summaries[method]=dict(**g,OOD_points=sum(r['length'] for r in chosen),hard={str(f):sum(r['compatible'][str(f)] for r in chosen)/g['N'] for f in [1,3,6]},soft={str(f):sum(r['soft'][str(f)] for r in chosen)/g['N'] for f in [1,3,6]})
  correlations={k:{str(f):dict(pearson=float(pearsonr([v[k][str(f)] for v in summaries.values()],[v['SR'] for v in summaries.values()]).statistic),spearman=float(spearmanr([v[k][str(f)] for v in summaries.values()],[v['SR'] for v in summaries.values()]).statistic)) for f in [1,3,6]} for k in ['hard','soft']}
  report=dict(protocol='Whole original TCP FK position/orientation/width in robot-base coordinates, no object/grasp-axis normalization, no contact classification; single successful OOD reference, monotone DP, q=.925 full-path calibration, same reset excluded. This is a new geometry proxy, not a proven recovery label.',
   references=[r['seed'] for r in refs],calibration=[r['seed'] for r in cal],summaries=summaries,correlations=correlations,rows=rows)
  (ROOT/f'full_path_{task}.json').write_text(json.dumps(report,indent=2,allow_nan=False));all_results[task]={k:v for k,v in report.items() if k!='rows'};print(task,summaries,correlations,flush=True)
 (ROOT/'full_path_summary.json').write_text(json.dumps(all_results,indent=2,allow_nan=False))

if __name__=='__main__':main()
