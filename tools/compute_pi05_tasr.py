#!/usr/bin/env python3
"""TASR on original pi05 suffixes, with explicit reconstruction coverage."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit
from analyze_open_drawer_suffix_recovery import PandaFK,pose_channels,suffix_map

UNITS=np.array([.02,np.deg2rad(15),.01])
MIN_RADIUS=float(np.linalg.norm(np.array([.001,np.deg2rad(1),.001])/UNITS))
PHASES=['approach_alignment','close']

@njit(cache=True)
def fast_map(cost):
 n,m=cost.shape;parent=np.zeros((n,m),dtype=np.int32);prev=cost[0].copy()
 for i in range(1,n):
  current=np.empty(m)
  for j in range(m):
   best=prev[j];arg=j
   for k in range(1,min(5,j)+1):
    if prev[j-k]<best:best=prev[j-k];arg=j-k
   current[j]=cost[i,j]+best;parent[i,j]=arg
  prev=current
 mapping=np.zeros(n,dtype=np.int64);mapping[n-1]=np.argmin(prev)
 for i in range(n-1,0,-1):mapping[i-1]=parent[i,mapping[i]]
 return mapping

def test_map():
 rng=np.random.default_rng(2903)
 for shape in [(1,1),(2,13),(21,3),(71,45)]:
  for costs in [rng.random(shape),np.zeros(shape)]:
   assert np.array_equal(fast_map(costs),suffix_map(costs,(0,shape[1]-1),False))

def target_blocks(task,actions,contact,start,n):
 negative=np.flatnonzero(actions[start:start+n,-1]<0)
 if not len(negative):raise ValueError('no closing command')
 close=int(negative[0]);stable=3 if task=='stackcube' else 4
 if task=='stackcube':
  end=close+5
  if end>n or not contact[start+end-stable+1:start+end+1].all():raise ValueError('first close is not stable')
 else:
  # End at the confirmed stable grasp. Extra closed holding, lift and transport
  # are not target supervision, including the old airplane 60-step close hold.
  candidates=[j for j in range(start+close+stable,start+n+1) if contact[j-stable+1:j+1].all()]
  if not candidates:raise ValueError('no stable grasp after closing')
  end=candidates[0]-start
 return {'approach_alignment':(0,close),'close':(close,end)}

def load_task(task,groups,replay,fk,retries=()):
 episodes=[];gaps=[]
 for group in groups:
  for row in json.loads(Path(group['episodes_file']).read_text()):
   short={k:v for k,v in row.items() if k!='metadata'}
   arr=np.load(row['arrays']);start=row['expert_start'];n=row['expert_points']
   accepted=False;gap=dict(row=short,reason='missing replay')
   for variant in [replay,*retries]:
    base=variant/task/row['method']/f"episode_{row['source_episode_index']:06d}"
    if not base.with_suffix('.json').exists():continue
    report=json.loads(base.with_suffix('.json').read_text());states=np.load(base.with_suffix('.npz'))
    try:blocks=target_blocks(task,arr['actions'],states['grasped'],start,n)
    except ValueError as exc:gap=dict(row=short,reason=str(exc),replay_report=report);continue
    end=blocks['close'][1]
    # Later state drift is preserved but is not an input to target membership.
    required_error=float(np.max(abs(states['qpos'][start:start+end+1]-arr['qpos'][:end+1]))) if end<n else float(np.max(abs(states['qpos'][start:start+end]-arr['qpos'][:end])))
    if required_error>=1e-4:gap=dict(row=short,reason='target or confirmation state replay drift',target_qpos_error=required_error,replay_report=report);continue
    accepted=True;break
   if not accepted:gaps.append(gap);continue
   row=dict(row,replay_array=str(base.with_suffix('.npz').resolve()))
   p,R=fk.pose(arr['qpos']);p+=np.array([-.615,0,0]);axes=R[end-1]
   pose=dict(position=(p-states['object_p'][start:start+n])@axes,quaternion=Rotation.from_matrix(axes.T@R).as_quat()[:,[3,0,1,2]],width=arr['qpos'][:,-2:].sum(axis=1)[:,None])
   episodes.append(dict(key=(row['method'],row['seed'],row['split']),row=row,pose=pose,contact=states['grasped'][start:start+n],blocks=blocks,
    required_replay_error=required_error,full_replay_status=report['status'],full_replay_error=report['max_qpos_error']))
 return episodes,gaps

def pair(e,r,p):
 a,b=e['blocks'][p];c,d=r['blocks'][p]
 if a==b:return dict(indices=[],mapping=[],distance=[],contact=[])
 if c==d:return None
 residual=pose_channels({k:v[a:b] for k,v in e['pose'].items()},{k:v[c:d] for k,v in r['pose'].items()});cost=np.linalg.norm(residual/UNITS,axis=2)
 qc=e['contact'][a:b];rc=r['contact'][c:d];mapping=fast_map(cost+1000*(qc[:,None]!=rc[None,:]))
 return dict(indices=list(range(a,b)),mapping=(mapping+c).tolist(),distance=cost[np.arange(len(mapping)),mapping].tolist(),contact=(qc==rc[mapping]).tolist())

def analyze_task(task,groups,replay,fk,retries=()):
 episodes,gaps=load_task(task,groups,replay,fk,retries)
 nominal=[e for e in episodes if e['row']['method']=='offline_oracle' and e['row']['split']=='ood']
 refs=[e for e in nominal if e['row']['seed']%5<3];cal=[e for e in nominal if e['row']['seed']%5==3];checks=[e for e in nominal if e['row']['seed']%5==4]
 assert len(refs)>=5 and len(cal)>=3 and len(checks)>=3,(task,len(refs),len(cal),len(checks))
 cache={};threshold_cache={}
 def compare(e,r):
  key=e['key'],r['key']
  if key not in cache:cache[key]={p:pair(e,r,p) for p in PHASES}
  return cache[key]
 def choose(e,bank):
  scores=[]
  for r in bank:
   if e['row']['seed']==r['row']['seed']:continue
   m=compare(e,r)
   if any(v is None for v in m.values()):continue
   scores.append((np.mean([np.mean(v['distance'])+1000*(1-np.mean(v['contact'])) for v in m.values() if v['indices']]),r))
  return min(scores,key=lambda x:x[0])[1]
 def thresholds(skip):
  if skip not in threshold_cache:
   values={p:[] for p in PHASES};bank=[e for e in refs if e['row']['seed']!=skip]
   for e in cal:
    if e['row']['seed']==skip:continue
    r=choose(e,bank)
    for p,m in compare(e,r).items():values[p].extend(d for d,ok in zip(m['distance'],m['contact']) if ok)
   threshold_cache[skip]={p:max(MIN_RADIUS,float(np.quantile(v,.925))) for p,v in values.items()}
  return threshold_cache[skip]
 rows=[];nominal_seeds={e['row']['seed'] for e in nominal}
 for i,e in enumerate(episodes):
  r=choose(e,refs);row=e['row'];limits=thresholds(row['seed'] if row['seed'] in nominal_seeds else None);parts=compare(e,r)
  n=row['expert_points'];labels={str(f):['non_target']*n for f in [1,2,3]};dist=[None]*n;mapping=[-1]*n;contact=[None]*n
  for p,m in parts.items():
   for j,k,d,ok in zip(m['indices'],m['mapping'],m['distance'],m['contact']):
    dist[j]=d;mapping[j]=k;contact[j]=ok
    for f in [1,2,3]:labels[str(f)][j]='target_compatible' if ok and d<=limits[p]*f else 'target_mismatch'
  counts={f:ls.count('target_compatible') for f,ls in labels.items()}
  rows.append(dict(**{k:v for k,v in row.items() if k!='metadata'},blocks=e['blocks'],thresholds=limits,distance=dist,reference_mapping=mapping,
   contact_compatible=contact,labels=labels,reference_seed=r['row']['seed'],compatible_points=counts,Q={f:c/n for f,c in counts.items()},
   full_replay_status=e['full_replay_status'],target_replay_max_error=e['required_replay_error']))
  if i%100==0:print(task,'SCORED',i,'of',len(episodes),flush=True)
 summary={}
 for group in groups:
  method=group['method'];selected=[r for r in rows if r['method']==method and r['selected']];known=sum(r['expert_points'] for r in selected);total=group['expert_points'];unknown=total-known
  count={f:sum(r['compatible_points'][f] for r in selected) for f in ['1','2','3']}
  summary[method]=dict(total_expert_points=total,scored_expert_points=known,missing_expert_points=unknown,scored_episodes=len(selected),total_selected_episodes=group['selected_episodes'],compatible_points=count,
   TASR={f:c/total if unknown==0 else None for f,c in count.items()},bounds_if_incomplete={f:[c/total,(c+unknown)/total] for f,c in count.items()} if unknown else None)
 checkrows=[r for r in rows if r['method']=='offline_oracle' and r['split']=='ood' and r['seed'] in {e['row']['seed'] for e in checks}]
 check={f:sum(r['compatible_points'][f] for r in checkrows)/sum(sum(b-a for a,b in r['blocks'].values()) for r in checkrows) for f in ['1','2','3']}
 return dict(protocol=dict(task=task,frame='current object centre and query stable-grasp TCP axes; original trained qpos used for FK',
  target='approach/alignment and closing through stable grasp; extra holding/lift/transport excluded',
  contact='reconstructed instantaneous grasp predicate; not ever-grasped',reference='one same-task successful OOD offline trajectory; same seed excluded across methods and calibration',
  calibration_q=.925,radius_factors=[1,2,3],units=dict(position_m=.02,rotation_deg=15,gripper_m=.01),minimum_radius=MIN_RADIUS,
  target_replay_tolerance=1e-4,unused_tail_rule='later replay drift retained in report, does not invalidate earlier verified metric inputs; full original suffix length retained',
  references=[e['row']['seed'] for e in refs],calibration=[e['row']['seed'] for e in cal],checks=[e['row']['seed'] for e in checks]),normal_target_retention=check,summary=summary,rows=rows,gaps=gaps)

def main(root,replay,task_filter=None,retries=()):
 test_map();manifest=json.loads((root/'training_assets/manifest.json').read_text());fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));result={}
 for task in ['object','stackcube','airplane']:
  if task_filter and task!=task_filter:continue
  result[task]=analyze_task(task,[g for g in manifest if g['task']==task],replay,fk,retries)
  (root/f'tasr_{task}.json').write_text(json.dumps(result[task],indent=2,allow_nan=False));print(task,'SUMMARY',json.dumps(result[task]['summary']),flush=True)
 (root/'tasr_scores.json').write_text(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--task',choices=['object','stackcube','airplane']);p.add_argument('--retry-root',action='append',type=Path,default=[]);a=p.parse_args();main(a.root,a.replay,a.task,a.retry_root)
