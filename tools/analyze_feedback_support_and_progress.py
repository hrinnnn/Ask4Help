"""Support-space ablations and ID-only progress-regression stress test."""
import json,collections
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from expert_feedback_pca_core import ExpertFeedbackPCA
from analyze_open_drawer_suffix_recovery import PandaFK,pose_channels

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def main():
 cache=np.load(ROOT/'forward_v1/head_probe_cache.npz');s=json.loads((ROOT/'forward_v1/samples.json').read_text());raw=np.load(ROOT/'forward_v1/proprio.npz')['qpos'];x=cache['bridge_features'];loss=cache['native_loss_mc'].mean(1);valid=cache['valid_mask'].sum(1)
 ep=np.array([r['episode_id'] for r in s]);group=np.array([r['group'] for r in s]);offset=np.array([r['offset'] for r in s]);cal=group=='id_calibration';check=group=='id_check'
 pca=np.load(ROOT/'original_pca_bridge.npz');meta=json.loads((ROOT/'original_pca_bridge.json').read_text());coords=(x-pca['mean'])@pca['eigenvectors'];cut=coords.shape[1]-meta['pca_dim'];scores=np.linalg.norm(coords[:,:cut],axis=1)
 descriptors={'full_bridge':x,'id_principal_coordinates':coords[:,cut:],'residual_coordinates':coords[:,:cut],'proprio_qpos':raw}
 summaries={}
 for name,features in descriptors.items():
  initial=ExpertFeedbackPCA.calibrate(features[cal],ep[cal],loss[cal],valid[cal],meta['threshold']);decisions=[];events=[];idchecks=[]
  for g in dict.fromkeys(group[~(cal|check)]):
   gate=ExpertFeedbackPCA(initial.tau0,initial.center,initial.scale,initial.radius,initial.q_loss)
   for e in dict.fromkeys(ep[group==g]):
    idx=np.flatnonzero(ep==e);qs=[gate.query(features[i],scores[i]) for i in idx[offset[idx]%5==0]];decisions.extend(qs)
    f=gate.add_episode(e,features[idx],scores[idx],loss[idx],valid[idx]);events.append(f)
   for e in dict.fromkeys(ep[check]):
    qs=[gate.query(features[i],scores[i]) for i in np.flatnonzero((ep==e)&(offset%5==0))]
    idchecks.append(dict(old=any(q['baseline_stop'] for q in qs),new=any(q['stop'] for q in qs)))
  summaries[name]=dict(queries=len(decisions),reason_counts=dict(collections.Counter(q['reason'] for q in decisions)),new_requests=sum(q['stop'] and not q['baseline_stop'] for q in decisions),
   new_wait=sum(not q['stop'] and q['baseline_stop'] for q in decisions),id_baseline_any=sum(q['old'] for q in idchecks),id_new_any=sum(q['new'] for q in idchecks),idcontexts=len(idchecks),radius=initial.radius)
 # Progress from a single nearest complete ID demonstration, with free nearest
 # position at each sample (monotone DP would suppress the very regression sought).
 fk=PandaFK(Path('artifacts/open_drawer_suffix_d_20260831/inputs/panda_v2.urdf'));p,R=fk.pose(raw);quats=Rotation.from_matrix(R).as_quat()[:,[3,0,1,2]];pose=dict(position=p,quaternion=quats,width=raw[:,-2:].sum(1)[:,None]);units=np.array([.02,np.deg2rad(15),.01])
 id_episodes=list(dict.fromkeys(ep[cal]));regression=[]
 for e in dict.fromkeys(ep):
  idx=np.flatnonzero(ep==e);q={k:v[idx] for k,v in pose.items()};candidates=[]
  for re in id_episodes:
   if re==e:continue
   ridx=np.flatnonzero(ep==re);r={k:v[ridx] for k,v in pose.items()};cost=np.linalg.norm(pose_channels(q,r)/units,axis=2)
   nearest=cost.argmin(1);d=cost[np.arange(len(idx)),nearest];candidates.append((float(d.mean()),re,nearest/(len(ridx)-1),d))
  _,re,phase,d=min(candidates,key=lambda t:t[0]);back=np.maximum(-np.diff(phase),0)
  regression.append(dict(episode=e,group=s[idx[0]]['group'],reference_id_episode=re,initial_regression_5steps=float(back[:5].sum()),total_regression=float(back.sum()),mean_distance=float(d.mean()),matched_phase=phase.tolist()))
 bound=float(np.quantile([r['initial_regression_5steps'] for r in regression if r['group']=='id_calibration'],.95,method='higher'))
 pseudo=[]
 for r in regression:
  if r['group']!='id_calibration':continue
  backwards=np.maximum(-np.diff(r['matched_phase']),0)
  for k in range(0,max(0,len(backwards)-4),5):pseudo.append(float(backwards[k:k+5].sum()))
 phase_consistent_bound=float(np.quantile(pseudo,.95,method='higher'))
 count={g:dict(episodes=len([r for r in regression if r['group']==g]),above=sum(r['initial_regression_5steps']>bound for r in regression if r['group']==g),median_initial=float(np.median([r['initial_regression_5steps'] for r in regression if r['group']==g]))) for g in dict.fromkeys(group)}
 robustcount={g:sum(r['initial_regression_5steps']>phase_consistent_bound for r in regression if r['group']==g) for g in count}
 distance_bound=float(np.quantile([r['mean_distance'] for r in regression if r['group']=='id_calibration'],.95,method='higher'))
 supported_count={g:sum(r['initial_regression_5steps']>phase_consistent_bound and r['mean_distance']<=distance_bound for r in regression if r['group']==g) for g in count}
 result=dict(status='EXPLORATORY_NO_SR_OR_RECOVERY_GROUND_TRUTH',support_ablation=summaries,progress=dict(definition='First5 expert-step backward nearest-ID phase travel; same ID episode excluded; no OOD demonstration reference',
  initial_ID_q95_diagnostic=bound,id_all_query_boundary_q95=phase_consistent_bound,id_pseudo_takeover_windows=len(pseudo),corrected_above_counts=robustcount,
  id_reference_distance_q95=distance_bound,regression_with_reference_support_counts=supported_count,by_group_initial_calibration=count,rows=regression),
  limitations=['Expert backward motion may be legitimate; nearest-ID phase is not calibrated task progress.','Full-bridge and projected-coordinate supports need independent rollout evaluation before use.','No OOD prior or manually labeled time used; retrospective selection of demonstrations remains a proxy.'])
 (ROOT/'support_progress_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({'support':summaries,'progress_initial_id_q95':bound,'progress_phase_consistent_id_q95':phase_consistent_bound,'progress_corrected_counts':robustcount,'progress_groups':count},indent=2))

if __name__=='__main__':main()
