"""Independent matrix-distance check of the full-path candidate artifact."""
import json
from pathlib import Path
import numpy as np
from analyze_open_drawer_suffix_recovery import PandaFK

root=Path('artifacts/ood_supervision_exploration_20260906');base=Path('artifacts/pi05_table_tasr_20260903');fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));audit={}
for task in ['stackcube','airplane','object','opendrawer']:
 report=json.loads((root/f'full_path_{task}.json').read_text());poses={}
 if task=='opendrawer':
  source=json.loads(Path('artifacts/open_drawer_tolerance_sweep_20260903/factor_1p00/analysis.json').read_text())
  lookup={(f"t{r['anchor']}",r['seed']):r for r in source['rows'] if r['accepted']};refmethod='t0'
 else:
  source=json.loads((base/f'tasr_{task}.json').read_text());lookup={(r['method'],r['seed']):r for r in source['rows'] if r['split']=='ood'};refmethod='offline_oracle'
 def pose(key):
  if key not in poses:
   r=lookup[key]
   q=np.load(Path(r['directory'])/'states.npy')[r['takeover']:r['takeover']+r['expert_anchors']] if task=='opendrawer' else np.load(r['arrays'])['qpos']
   p,R=fk.pose(q);poses[key]=(p,R,q[:,-2:].sum(1))
  return poses[key]
 worst=0;points=0
 for r in report['rows']:
  assert r['seed']!=r['reference_seed'];p,R,g=pose((r['method'],r['seed']));rp,rR,rg=pose((refmethod,r['reference_seed']));j=np.array(r['mapping'])
  assert len(j)==len(p) and np.all(np.diff(j)>=0) and np.all(np.diff(j)<=5)
  rel=np.swapaxes(R,1,2)@rR[j];ang=np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1))
  d=np.sqrt((np.linalg.norm(p-rp[j],axis=1)/.02)**2+(ang/np.deg2rad(15))**2+((g-rg[j])/.01)**2)
  err=float(np.max(abs(d-r['distance'])));assert err<1e-5;worst=max(worst,err);points+=len(j)
  for f in [1,3,6]:assert int(np.sum(d<=r['threshold']*f))==r['compatible'][str(f)]
 for m,s in report['summaries'].items():
  selected=[r for r in report['rows'] if r['method']==m and r['selected']];assert sum(r['length'] for r in selected)==s['OOD_points']
  for f in ['1','3','6']:assert abs(sum(r['compatible'][f] for r in selected)/s['N']-s['hard'][f])<1e-12
 audit[task]=dict(status='PASS',points=points,trajectories=len(report['rows']),maximum_distance_difference=worst)
(root/'full_path_independent_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit))
