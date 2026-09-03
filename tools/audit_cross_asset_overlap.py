#!/usr/bin/env python3
"""Independent recount: geometry, contacts, fixed matching, budgets and utility."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from analyze_open_drawer_suffix_recovery import PandaFK

def main(root):
 baseline=json.loads(Path('artifacts/stackcube_airplane_target_ratio_20260831/analysis.json').read_text());checked=0
 for f in [1,2,3]:
  variant=json.loads((root/f'fixedgrid_x{f}/analysis.json').read_text())
  for task,group in baseline.items():
   for a,b in zip(group['rows'],variant[task]['rows']):
    for field in ['arrays','reference_seed','reference_mapping','distance','blocks','selected','expert_start','expert_anchors']:assert a[field]==b[field]
    q=np.load(a['arrays']);ref=next(r for r in group['rows'] if r['step']==0 and r['seed']==a['reference_seed']);r=np.load(ref['arrays'])
    labels=[]
    for i,d in enumerate(a['distance']):
     if d is None:labels.append('non_target');continue
     phase=next(p for p,(lo,hi) in a['blocks'].items() if lo<=i<hi)
     contact=(q['task_states'][a['expert_start']+i,-2]>.5)==(r['task_states'][ref['expert_start']+a['reference_mapping'][i],-2]>.5)
     labels.append('target_compatible' if contact and d<=a['thresholds'][phase]*f else 'target_mismatch')
    assert labels==b['labels'];assert b['compatible_anchors']==labels.count('target_compatible');checked+=1
 stage=json.loads((root/'stage2_overlap.json').read_text());fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));poses={}
 def pose(row):
  key=row['arrays']
  if key not in poses:
   q=np.load(key)['qpos'];p,R=fk.pose(q);p+=np.array([-.615,0,0]);ts=np.load(key)['task_states'][row['expert_start']:row['expert_start']+len(q)]
   axes=R[row['phase_evidence']['close_end']-1];poses[key]=((p-ts[:,3:6])@axes,axes.T@R,q[:,-2:].sum(axis=1),ts[:,16]>.5)
  return poses[key]
 lookup={r['seed']:r for r in stage['rows'] if r['method']=='immediate'};stage_points=0;largest_residual=0
 for row in stage['rows']:
  ref=lookup[row['reference_seed']];assert ref['seed']!=row['seed'];qp,qR,qw,qc=pose(row);rp,rR,rw,rc=pose(ref)
  for p,(lo,hi) in row['blocks'].items():
   js=np.array(row['reference_mapping'][lo:hi],dtype=int)
   if not len(js):continue
   rlo,rhi=ref['blocks'][p];assert np.all((js>=rlo)&(js<rhi)) and np.all((np.diff(js)>=0)&(np.diff(js)<=5))
   dp=np.linalg.norm(qp[lo:hi]-rp[js],axis=1);rel=np.swapaxes(qR[lo:hi],1,2)@rR[js]
   angle=np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1));gap=np.abs(qw[lo:hi]-rw[js])
   distance=np.sqrt((dp/.02)**2+(angle/np.deg2rad(15))**2+(gap/.01)**2)
   error=np.max(np.abs(distance-np.array(row['distance'][lo:hi])));assert error<1e-5;largest_residual=max(largest_residual,float(error))
   contact=qc[lo:hi]==rc[js];assert contact.tolist()==row['contact_compatible'][lo:hi]
   for f in [1,2,3]:
    green=(distance<=row['thresholds'][p]*f)&contact
    assert green.tolist()==[lab=='target_compatible' for lab in row['labels_by_factor'][str(f)][lo:hi]]
   stage_points+=len(js)
 for cohort,methods in stage['summaries'].items():
  for method,s in methods.items():
   rows=[r for r in stage['rows'] if r['selected'] and r['cohort']==cohort and r['method']==method]
   assert sum(r['expert_anchors'] for r in rows)==s['expert_points']
   for f in ['1','2','3']:assert sum(r['labels_by_factor'][f].count('target_compatible') for r in rows)==s['compatible_points'][f]
 utility=0
 for path in (root/'inputs').glob('utility_evidence_*.json'):
  for cohort,g in json.loads(path.read_text()).items():
   for item in g['files']:
    s=item['summary'];n=s.get('episodes');rows=s.get('rows',[])
    if not isinstance(n,int):continue
    assert len(rows)==n
    for summaryfield,rowfield in [('successes','success'),('ever_grasped_successes','ever_grasped'),('strict_successes','strict_success')]:
     if summaryfield in s:assert sum(bool(r[rowfield]) for r in rows)==s[summaryfield]
    utility+=1
 videos=0
 for filename in ['video_audit.json','stage2_video_audit.json']:
  for record in json.loads((root/filename).read_text()):
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',record['output']]))['streams'][0]
   assert int(probe['nb_frames'])==record['frames'] and record['source']['state_step_offset']==0;videos+=1
 report=dict(status='PASS_NUMERICAL_AND_ARTIFACT_AUDIT_NOT_SCIENTIFIC_VALIDATION',fixed_variant_episode_checks=checked,stage2_trajectories=len(stage['rows']),
  stage2_target_points_geometry_checked=stage_points,maximum_independent_distance_difference=largest_residual,utility_episode_summaries_recounted=utility,rendered_individual_videos_checked=videos,
  remaining='Legacy gate cohorts lack dynamic object/contact records; their overlap values are not computed. Stable neck-grasp utility is not reconstructed from ever_grasped.')
 (root/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
