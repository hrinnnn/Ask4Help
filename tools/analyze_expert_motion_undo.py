"""ID-calibrated expert undo of the actual preceding robot motion.

Uses only measured TCP translation and gripper width; no object state, OOD
reference, task phase or optimal timing labels enter the signal.
"""
import json
from pathlib import Path
import numpy as np
from analyze_open_drawer_suffix_recovery import PandaFK

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def undo(p,w,k,delta=5):
 if k<delta or k+delta>=len(p):return None
 prior=p[k]-p[k-delta];after=p[k+delta]-p[k];norm=float(np.linalg.norm(prior))
 back=max(0.,-float(np.dot(after,prior))/norm) if norm>1e-8 else 0.
 width_prior=float(w[k]-w[k-delta]);width_after=float(w[k+delta]-w[k]);width_back=max(0.,-np.sign(width_prior)*width_after) if abs(width_prior)>1e-8 else 0.
 return dict(value=float(np.hypot(back/.02,width_back/.01)),back_translation_m=back,back_gripper_m=width_back,policy_motion_m=norm)

def main():
 samples=json.loads((ROOT/'forward_v1/samples.json').read_text());q=np.load(ROOT/'forward_v1/proprio.npz')['qpos'];fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));p,_=fk.pose(q);w=q[:,-2:].sum(1)
 ep=np.array([r['episode_id'] for r in samples]);groups=np.array([r['group'] for r in samples]);cal=[]
 for e in dict.fromkeys(ep[groups=='id_calibration']):
  idx=np.flatnonzero(ep==e)
  for k in range(5,len(idx)-5,5):cal.append(undo(p[idx],w[idx],k))
 threshold=float(np.quantile([r['value'] for r in cal],.95,method='higher'));rows=[]
 manifest=json.loads(Path('artifacts/cross_asset_overlap_utility_20260903/inputs/stage2/manifest.json').read_text());spec=json.loads((ROOT/'sample_spec.json').read_text());want={e['episode_id'] for e in spec['episodes']}
 for g in manifest:
  for r in json.loads(Path(g['episodes_file']).read_text()):
   eid=f"{g['method']}:{r['meta']['seed']}:{r['train']['dataset_episode_index']}"
   if eid not in want:continue
   a=np.load(r['arrays']);ts=a['task_states'];k=r['train']['expert_start_step'];signal=undo(ts[:,6:9],ts[:,15],k)
   rows.append(dict(episode=eid,group=g['method'],actual_takeover=k,signal=signal,flag=signal is not None and signal['value']>threshold))
 check=[]
 for e in dict.fromkeys(ep[groups=='id_check']):
  idx=np.flatnonzero(ep==e)
  for k in range(5,len(idx)-5,5):check.append(undo(p[idx],w[idx],k))
 summary={g:dict(episodes=len([r for r in rows if r['group']==g]),observable=sum(r['signal'] is not None for r in rows if r['group']==g),flagged=sum(r['flag'] for r in rows if r['group']==g)) for g in dict.fromkeys(r['group'] for r in rows)}
 result=dict(status='OBSERVABLE_MOTION_UNDO_PROXY_NOT_RECOVERY_CLASSIFIER',definition='Opposite component of first Delta expert displacement against last Delta policy displacement, plus reversal of gripper width; normalize by 2cm and1cm.',
  delta=5,id_q95=threshold,id_calibration_windows=len(cal),id_check_windows=len(check),id_check_above=sum(r['value']>threshold for r in check),by_group=summary,rows=rows,
  limitations=['Cannot infer lateness at reset or after stationary policy motion.','Legitimate direction changes can look like undo; ID pseudo-takeovers calibrate but do not prove safety.','No autonomous feedback-gate outcome measured.'])
 (ROOT/'motion_undo_diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
