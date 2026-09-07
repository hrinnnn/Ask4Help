"""Independent counts and causality audit for the frozen-forward feedback probe."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('artifacts/expert_feedback_pca_20260907')

def main():
 a=np.load(ROOT/'forward_v1/head_probe_cache.npz');samples=json.loads((ROOT/'forward_v1/samples.json').read_text());result=json.loads((ROOT/'feedback_probe_results.json').read_text())
 n=len(samples);assert n==len(a['native_loss_mc'])==len(a['bridge_features']);assert all(np.isfinite(a[k]).all() for k in a.files)
 masks=a['valid_mask'];reconstructed=np.sum(a['native_loss_per_action_mc']*masks[:,None,:],axis=2)/masks.sum(1)[:,None]
 assert np.allclose(reconstructed,a['native_loss_mc'],rtol=1e-6,atol=1e-5)
 ep=np.array([s['episode_id'] for s in samples]);group=np.array([s['group'] for s in samples]);frames=np.array([s['offset'] for s in samples]);valid=masks.sum(1);loss=a['native_loss_mc'].mean(1)
 cal=group=='id_calibration';q=float(np.quantile(loss[cal&(valid==10)],.95,method='higher'));assert q==result['calibrations']['point']['q_loss']
 for e in dict.fromkeys(ep):
  ids=np.flatnonzero(ep==e);assert np.array_equal(frames[ids],np.arange(len(ids)));assert np.array_equal(valid[ids],np.minimum(10,len(ids)-np.arange(len(ids))))
 point=result['variants']['point_document'];lookup={e['episode']:e for e in point['episodes']};checked=0
 for eid,e in lookup.items():
  ids=np.flatnonzero(ep==eid);full=ids[valid[ids]==10];event=next((int(frames[i]) for i in full if loss[i]>q),None);assert event==e['event']
  queries=[d for d in point['decisions'] if d['episode']==eid];assert len(set(d['memory_episodes'] for d in queries))==1
  assert all(d['stop']==(d['score']>d['threshold']) for d in queries)
  assert all(d['baseline_stop']==(d['score']>result['original_pca']['threshold']) for d in queries)
  checked+=len(queries)
 for g in dict.fromkeys(e['group'] for e in lookup.values()):
  episodes=[e for e in point['episodes'] if e['group']==g]
  for index,e in enumerate(episodes):assert all(d['memory_episodes']==index for d in point['decisions'] if d['episode']==e['episode'])
 report=dict(status='PASS_ENGINEERING_AND_CHRONOLOGY_NOT_CLOSED_LOOP_VALIDATION',observations=n,id_calibration_episodes=len(set(ep[cal])),id_check_unique_episodes=len(set(ep[group=='id_check'])),
  expert_episodes=len(lookup),query_decisions_checked=checked,native_loss_max_reduction_difference=float(np.max(abs(reconstructed-a['native_loss_mc']))),
  no_ood_reference_calibration=True,within_episode_memory_frozen=True,limitation='Only archived expert-state query decisions; no autonomous adaptive rollouts or post-training result.')
 (ROOT/'independent_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
