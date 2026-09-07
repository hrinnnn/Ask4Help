"""Matched-observation free prediction versus expert-conditioned fitting loss."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from expert_feedback_pca_core import ExpertFeedbackPCA,first_boundary_feedback

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def main():
 p=ROOT/'free_action_v1';a=np.load(p/'head_probe_cache.npz');samples=json.loads((p/'samples.json').read_text());old=json.loads((ROOT/'forward_v1/samples.json').read_text());assert samples==old
 original=np.load(ROOT/'forward_v1/head_probe_cache.npz');assert np.allclose(a['bridge_features'],original['bridge_features'],atol=1e-6);assert np.allclose(a['native_loss_mc'],original['native_loss_mc'],rtol=1e-6,atol=1e-5)
 x=a['bridge_features'];target=a['targets'];mask=a['valid_mask'];valid=mask.sum(1);ep=np.array([s['episode_id'] for s in samples]);groups=np.array([s['group'] for s in samples]);offset=np.array([s['offset'] for s in samples]);cal=groups=='id_calibration';check=groups=='id_check';full=valid==10
 pred=np.clip(a['free_predictions'],-1,1);error=((pred-target[:,None,:,:])**2).mean(-1).mean(1)*100
 signals={'native_chunk':a['native_loss_mc'].mean(1),'free_first_action':error[:,0],'free_executed5':np.sum(error[:,:5]*mask[:,:5],1)/mask[:,:5].sum(1),'free_chunk10':np.sum(error*mask,1)/valid}
 pca=np.load(ROOT/'original_pca_bridge.npz');meta=json.loads((ROOT/'original_pca_bridge.json').read_text());score=np.linalg.norm((x-pca['mean'])@pca['eigenvectors'][:,:x.shape[1]-meta['pca_dim']],axis=1);result={}
 for name,loss in signals.items():
  gate0=ExpertFeedbackPCA.calibrate(x[cal],ep[cal],loss[cal],valid[cal],meta['threshold']);q=gate0.q_loss;episodes=[];ds=[];idrows=[]
  for g in dict.fromkeys(groups[~(cal|check)]):
   gate=ExpertFeedbackPCA(gate0.tau0,gate0.center,gate0.scale,gate0.radius,q)
   for e in dict.fromkeys(ep[groups==g]):
    idx=np.flatnonzero(ep==e)
    ds.extend(gate.query(x[i],score[i]) for i in idx[offset[idx]%5==0]);f=gate.add_episode(e,x[idx],score[idx],loss[idx],valid[idx]);episodes.append(dict(episode=e,group=g,**f))
  for e in dict.fromkeys(ep[check]):
   idx=np.flatnonzero(ep==e);f=first_boundary_feedback(loss[idx],valid[idx],q=q);idrows.append(f['event'])
  result[name]=dict(id_q95=q,expert_event0=sum(r['event']==0 for r in episodes),expert_event_later=sum(r['event'] is not None and r['event']>0 for r in episodes),expert_noevent=sum(r['event'] is None for r in episodes),
   id_check_any_loss_event=sum(v is not None for v in idrows),id_check_episodes=len(idrows),new_requests=sum(d['stop'] and not d['baseline_stop'] for d in ds),new_wait=sum(not d['stop'] and d['baseline_stop'] for d in ds),episodes=episodes)
 nativeq=result['native_chunk']['id_q95'];freeq=result['free_executed5']['id_q95'];native_low=signals['native_chunk']<=nativeq;free_high=signals['free_executed5']>freeq
 comparison={g:dict(full_windows=int(np.sum((groups==g)&full)),native_low=int(np.sum((groups==g)&full&native_low)),native_low_but_free_high=int(np.sum((groups==g)&full&native_low&free_high))) for g in dict.fromkeys(groups)}
 calibration=ExpertFeedbackPCA.calibrate(x[cal],ep[cal],signals['free_executed5'][cal],valid[cal],meta['threshold'])
 (p/'gate_calibration.json').write_text(json.dumps(dict(tau0=calibration.tau0,q_loss=calibration.q_loss,radius=calibration.radius,scale=calibration.scale,center=calibration.center.tolist(),H=10,delta=5,
  feedback='Mean squared error of clipped freely generated policy actions versus actual expert first5 actions, averaged across2 independent priors, x100',checkpoint='/root/xvla_stage2_inputs_priority/ckpt-7500'),indent=2))
 np.savez_compressed(p/'feedback_signals.npz',**signals)
 summary=dict(status='READ_ONLY_MODEL_COMPARISON_COMPLETE',matched_observations=len(samples),samples_each=2,denoising_steps=10,teacher_and_free_protocols_separate=True,results=result,disagreement_by_group=comparison,
  caveat='Agreement on an expert-state path does not prove the autonomous continuation follows that path; free-generated first block is more direct than expert-conditioned denoising but is still a proxy.')
 (p/'comparison.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:{f:v for f,v in r.items() if f!='episodes'} for k,r in result.items()},indent=2));print('DISAGREEMENT',comparison)

if __name__=='__main__':main()
