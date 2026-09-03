#!/usr/bin/env python3
"""Stage-2 overlap: transport and release, not Stage-1 grasp supervision.

Phase boundaries follow the recorded action chunks and the existing oracle's
0.07 m lift transition. No downstream result is used by matching/calibration.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from analyze_open_drawer_suffix_recovery import PandaFK,pose_channels,suffix_map
from analyze_fixedgrid_target_ratio import MIN_RADIUS,UNITS

PHASES=['transport','release']

def load_episodes(root):
 fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));episodes=[];errors=[]
 for group in json.loads((root/'inputs/stage2/manifest.json').read_text()):
  for entry in json.loads(Path(group['episodes_file']).read_text()):
   a=dict(np.load(entry['arrays']));qpos=a['qpos'];start=int(entry['train']['expert_start_step']);n=len(qpos)
   ts=a['task_states'][start:start+n];actions=a['actions'][start:start+n];negative=np.flatnonzero(actions[:,-1]<0);assert len(negative)
   close=int(negative[0]);close_end=close+5;assert close_end<=n
   releases=np.flatnonzero(actions[close_end:,-1]>0);assert len(releases),(entry['method'],entry['meta']['seed'])
   release=close_end+int(releases[0])
   # Lift precedes transport in the oracle; examine its actual decision boundaries.
   candidates=[j for j in range(close_end,release+1,5) if ts[j,2]>=.07]
   align=candidates[0] if candidates else release
   blocks={'transport':(align,release),'release':(release,n)}
   p,R=fk.pose(qpos);p+=np.array([-.615,0,0]);error=np.linalg.norm(p-ts[:,6:9],axis=1);assert error.max()<1e-4;errors.extend(error)
   axes=R[close_end-1];rotation=Rotation.from_matrix(axes.T@R)
   pose=dict(position=(p-ts[:,3:6])@axes,quaternion=rotation.as_quat()[:,[3,0,1,2]],width=qpos[:,-2:].sum(axis=1)[:,None])
   key=(entry['cohort'],entry['method'],entry['meta']['seed'])
   episodes.append(dict(key=key,cohort=entry['cohort'],method=entry['method'],seed=int(entry['meta']['seed']),split=entry['meta']['split'],selected=entry['selected'],
    expert_start=start,expert_anchors=n,arrays=entry['arrays'],video=entry['meta']['video'],blocks=blocks,pose=pose,contact=ts[:,16]>.5,
    phase_evidence=dict(first_closing_offset=close,close_end=close_end,transport_start=align,release_start=release,
     target='after lift, transport toward green target and release; grasp/reinforcement/lift excluded'),budget=group['selection']['budget']))
 return episodes,max(errors)

def match(e,r,p):
 a,b=e['blocks'][p];c,d=r['blocks'][p]
 if a==b:return dict(indices=[],mapping=[],distance=[],contact=[])
 if c==d:return None
 residual=pose_channels({k:v[a:b] for k,v in e['pose'].items()},{k:v[c:d] for k,v in r['pose'].items()});cost=np.linalg.norm(residual/UNITS,axis=2)
 qc=e['contact'][a:b];rc=r['contact'][c:d];mapping=suffix_map(cost+1000*(qc[:,None]!=rc[None,:]),(0,len(rc)-1),False)
 return dict(indices=list(range(a,b)),mapping=(mapping+c).tolist(),distance=cost[np.arange(len(mapping)),mapping].tolist(),contact=(qc==rc[mapping]).tolist())

def main(root):
 episodes,maxerror=load_episodes(root);nominal=[e for e in episodes if e['method']=='immediate'];refs=[e for e in nominal if e['seed']%5<3];cal=[e for e in nominal if e['seed']%5==3]
 checks=[e for e in nominal if e['seed']%5==4];cache={};limits_cache={}
 def compare(e,r):
  key=e['key'],r['key']
  if key not in cache:cache[key]={p:match(e,r,p) for p in PHASES}
  return cache[key]
 def choose(e,bank):
  candidates=[]
  for r in bank:
   if e['seed']==r['seed']:continue
   m=compare(e,r)
   if any(v is None for v in m.values()):continue
   cost=np.mean([np.mean(v['distance'])+1000*(1-np.mean(v['contact'])) for v in m.values() if len(v['indices'])])
   candidates.append((cost,r))
  return min(candidates,key=lambda v:v[0])[1]
 def limits(skip):
  if skip in limits_cache:return limits_cache[skip]
  bank=[e for e in refs if e['seed']!=skip];values={p:[] for p in PHASES}
  for e in cal:
   if e['seed']==skip:continue
   r=choose(e,bank)
   for p,v in compare(e,r).items():values[p].extend(d for d,ok in zip(v['distance'],v['contact']) if ok)
  limits_cache[skip]={p:max(MIN_RADIUS,float(np.quantile(v,.925))) for p,v in values.items()}
  return limits_cache[skip]
 rows=[];nominalseeds={e['seed'] for e in nominal}
 for ei,e in enumerate(episodes):
  r=choose(e,refs);thresholds=limits(e['seed'] if e['seed'] in nominalseeds else None);parts=compare(e,r)
  distances=[None]*e['expert_anchors'];mapping=[-1]*e['expert_anchors'];contacts=[None]*e['expert_anchors']
  for p,m in parts.items():
   for i,j,d,ok in zip(m['indices'],m['mapping'],m['distance'],m['contact']):distances[i]=d;mapping[i]=j;contacts[i]=ok
  labels={}
  for f in [1,2,3]:
   ls=['non_target']*e['expert_anchors']
   for p,m in parts.items():
    for i,d,ok in zip(m['indices'],m['distance'],m['contact']):ls[i]='target_compatible' if ok and d<=thresholds[p]*f else 'target_mismatch'
   labels[str(f)]=ls
  row={k:v for k,v in e.items() if k not in ['pose','contact','key']}
  row.update(distance=distances,reference_mapping=mapping,contact_compatible=contacts,reference_seed=r['seed'],thresholds=thresholds,labels_by_factor=labels,
   compatible_points={f:v.count('target_compatible') for f,v in labels.items()},Q={f:v.count('target_compatible')/e['expert_anchors'] for f,v in labels.items()})
  rows.append(row)
  if ei%100==0:print('SCORED',ei,'of',len(episodes),flush=True)
 summaries={}
 for cohort in ['stack_stage2_timing','stack_stage2_gates']:
  summaries[cohort]={}
  for method in dict.fromkeys(e['method'] for e in episodes if e['cohort']==cohort):
   selected=[r for r in rows if r['selected'] and r['cohort']==cohort and r['method']==method];n=sum(r['expert_anchors'] for r in selected)
   assert n==selected[0]['budget']
   counts={f:sum(r['compatible_points'][f] for r in selected) for f in ['1','2','3']}
   summaries[cohort][method]=dict(episodes=len(selected),expert_points=n,compatible_points=counts,Q={f:c/n for f,c in counts.items()},
    split_episodes={s:sum(r['split']==s for r in selected) for s in ['id','stage2_ood']})
 checkrows=[r for r in rows if r['method']=='immediate' and r['seed'] in {e['seed'] for e in checks}]
 out=dict(status='EXPLORATORY_STAGE2_EXTENSION_NOT_PRE_REGISTERED',protocol=dict(target='transport toward green cube after lift, plus release; preceding grasp/lift excluded',
  frame='recorded green-target centre, stable-grasp TCP axes; no measured object rotation',alignment='one reference for both phases, monotone local DP with 0..5 reference advances',
  contact='recorded instantaneous is_cubeA_grasped, not ever-grasped',threshold='per-phase successful immediate expert calibration q=.925; same-reset left out',
  tolerance_factors=[1,2,3],reference_seeds=[e['seed'] for e in refs],calibration_seeds=[e['seed'] for e in cal],check_seeds=[e['seed'] for e in checks],
  reference_group='same Stage2 timing immediate successful expert pool; shared also with gate cohort, same task and base family',
  domain_rule='score every selected expert suffix against Stage2 reference; do not arbitrarily delete ID examples from denominator'),
  max_fk_error_m=float(maxerror),normal_target_retention={f:sum(r['compatible_points'][f] for r in checkrows)/sum(sum(b-a for a,b in r['blocks'].values()) for r in checkrows) for f in ['1','2','3']},summaries=summaries,rows=rows)
 (root/'stage2_overlap.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(summaries),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
