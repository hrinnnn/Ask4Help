#!/usr/bin/env python3
"""Independent geometry/count checks for TASR, without rerunning gate selection."""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_open_drawer_suffix_recovery import PandaFK

def main(root,replay,tasks):
 fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));checks={}
 for task in tasks:
  analysis=json.loads((root/f'tasr_{task}.json').read_text());rows=analysis['rows'];poses={};contacts={}
  for r in rows:
   q=np.load(r['arrays'])['qpos'];start=r['expert_start'];n=len(q);e=r['blocks']['close'][1]
   state=np.load(r.get('replay_array',replay/task/r['method']/f"episode_{r['source_episode_index']:06d}.npz"))
   position,rotation=fk.pose(q);position+=np.array([-.615,0,0]);axes=rotation[e-1]
   key=(r['method'],r['seed'],r['split']);poses[key]=((position-state['object_p'][start:start+n])@axes,axes.T@rotation,q[:,-2:].sum(axis=1));contacts[key]=state['grasped'][start:start+n]
  maxerror=0;points=0
  for r in rows:
   key=(r['method'],r['seed'],r['split']);rk=('offline_oracle',r['reference_seed'],'ood');p,R,w=poses[key];rp,rR,rw=poses[rk]
   assert r['seed']!=r['reference_seed']
   for phase,(a,b) in r['blocks'].items():
    if a==b:continue
    j=np.array(r['reference_mapping'][a:b]);assert np.all(np.diff(j)>=0) and np.all(np.diff(j)<=5)
    rel=np.swapaxes(R[a:b],1,2)@rR[j];angle=np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1))
    dist=np.sqrt((np.linalg.norm(p[a:b]-rp[j],axis=1)/.02)**2+(angle/np.deg2rad(15))**2+((w[a:b]-rw[j])/.01)**2)
    err=float(np.max(np.abs(dist-r['distance'][a:b])));assert err<1e-5;maxerror=max(maxerror,err);contact=contacts[key][a:b]==contacts[rk][j]
    assert contact.tolist()==r['contact_compatible'][a:b]
    for factor in ['1','2','3']:
     expected=contact&(dist<=r['thresholds'][phase]*int(factor));actual=[x=='target_compatible' for x in r['labels'][factor][a:b]]
     assert expected.tolist()==actual
    points+=b-a
  for method,summary in analysis['summary'].items():
   selected=[r for r in rows if r['method']==method and r['selected']]
   assert sum(r['expert_points'] for r in selected)==summary['scored_expert_points']
   for f in ['1','2','3']:
    count=sum(r['labels'][f].count('target_compatible') for r in selected);assert count==summary['compatible_points'][f]
    if summary['missing_expert_points']==0:assert abs(summary['TASR'][f]-count/summary['total_expert_points'])<1e-12
    else:assert summary['TASR'][f] is None
  checks[task]=dict(status='PASS',geometry_points=points,trajectories=len(rows),unscored=len(analysis['gaps']),maximum_distance_error=maxerror)
 path=root/'tasr_independent_audit.json';previous=json.loads(path.read_text()) if path.exists() else {};previous.update(checks);path.write_text(json.dumps(previous,indent=2));print(json.dumps(checks))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--tasks',nargs='+',default=['object','stackcube','airplane']);a=p.parse_args();main(a.root,a.replay,a.tasks)
