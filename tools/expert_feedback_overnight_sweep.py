"""Development-only signal sweep; native and free generation remain separate."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def main():
 source=Path('artifacts/expert_feedback_pca_20260907/free_action_v1');out=Path('artifacts/expert_feedback_overnight_20260907');out.mkdir(exist_ok=True,parents=True)
 a=np.load(source/'head_probe_cache.npz');samples=json.loads((source/'samples.json').read_text());ep=np.array([s['episode_id'] for s in samples]);group=np.array([s['group'] for s in samples]);offset=np.array([s['offset'] for s in samples]);valid=a['valid_mask'].sum(1);full=valid==10
 cal=group=='id_calibration';check=group=='id_check';expert=~(cal|check)
 pred=np.clip(a['free_predictions'],-1,1);err=((pred-a['targets'][:,None])**2).mean(-1)*100
 signals={'native_denoising':a['native_loss_mc'].mean(1)}
 for h in [1,5,10]:
  block=err[:,:,:h].mean(-1)
  signals[f'free_mean_{h}']=block.mean(1);signals[f'free_best_of2_{h}']=block.min(1)
 rows=[];calibrations={};original=json.loads((source/'gate_calibration.json').read_text())
 for name,s in signals.items():
  for mode in ['point','episode_max']:
   values=s[cal&full] if mode=='point' else np.array([max(s[(ep==e)&full]) for e in dict.fromkeys(ep[cal])])
   for q in [.90,.95,.99]:
    threshold=float(np.quantile(values,q,method='higher'));r=dict(signal=name,mode=mode,quantile=q,threshold=threshold)
    for label,mask in [('id_check',check),('expert',expert)]:
     boundaries=[];wait_blocks=0
     for e in dict.fromkeys(ep[mask]):
      ix=np.flatnonzero((ep==e)&full);high=np.flatnonzero(s[ix]>threshold);boundary=int(offset[ix[high[0]]]) if len(high) else None
      boundaries.append(boundary)
      if boundary is not None:wait_blocks+=boundary//5
     r[label]=dict(n=len(boundaries),at_start=sum(v==0 for v in boundaries),later=sum(v is not None and v>0 for v in boundaries),at_least_one_block=sum(v is not None and v>=5 for v in boundaries),censored=sum(v is None for v in boundaries),low_error_blocks_before_boundary=wait_blocks)
    rows.append(r)
    if name=='free_mean_5':
     key=f'{mode}_{int(q*100)}';c=dict(original);c['q_loss']=threshold;c['calibration_mode']=mode;c['quantile']=q;calibrations[key]=c
 for key,c in calibrations.items():(out/f'calibration_{key}.json').write_text(json.dumps(c,indent=2))
 corr={name:float(spearmanr(signals['native_denoising'][full],s[full]).statistic) for name,s in signals.items() if name!='native_denoising'}
 result=dict(rows=rows,correlation_to_native=corr,n_observations=len(samples),note='Existing development observations; action discrepancy is not a learning-utility label; first observed expert boundary is not autonomous optimal timing.')
 (out/'signal_sweep.json').write_text(json.dumps(result,indent=2))
 print(json.dumps([r for r in rows if r['signal'] in ['native_denoising','free_mean_5'] and r['quantile']==.95],indent=2))

if __name__=='__main__':main()
