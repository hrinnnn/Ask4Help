"""ID-only calibration alternatives, without selecting against downstream SR."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('artifacts/expert_feedback_pca_20260907/free_action_v1')

def main():
    a=np.load(ROOT/'head_probe_cache.npz');samples=json.loads((ROOT/'samples.json').read_text())
    groups=np.array([s['group'] for s in samples]);episodes=np.array([s['episode_id'] for s in samples]);full=a['valid_mask'].sum(1)==10
    cal=groups=='id_calibration';check=groups=='id_check';expert=~(cal|check)
    pred=np.clip(a['free_predictions'][:,:,:5,:8],-1,1);target=a['targets'][:,:5,:8]
    expected_error=((pred-target[:,None])**2).mean((1,2,3))*100
    mean_error=((pred.mean(1)-target)**2).mean((1,2))*100
    stochastic_variance=((pred-pred.mean(1,keepdims=True))**2).mean((1,2,3))*100
    assert np.allclose(expected_error,mean_error+stochastic_variance,rtol=1e-5,atol=1e-5)
    rows=[]
    for name,loss in [('expected_free_error',expected_error),('mean_prediction_error',mean_error)]:
        for mode in ['point_q95','episode_max_q95']:
            values=loss[cal&full] if mode=='point_q95' else np.array([loss[(episodes==e)&full].max() for e in dict.fromkeys(episodes[cal])])
            threshold=float(np.quantile(values,.95,method='higher'))
            row=dict(signal=name,calibration=mode,threshold=threshold)
            for label,selection in [('ID_check',check),('expert',expert)]:
                events=[]
                for e in dict.fromkeys(episodes[selection]):
                    ix=np.flatnonzero((episodes==e)&full);cross=np.flatnonzero(loss[ix]>threshold)
                    events.append(int(cross[0]) if len(cross) else None)
                row[label]=dict(episodes=len(events),event0=sum(e==0 for e in events),later=sum(e is not None and e>0 for e in events),noevent=sum(e is None for e in events))
            rows.append(row)
    report=dict(status='ID_ONLY_ALTERNATIVE_CALIBRATION_DIAGNOSTIC',variants=rows,
                variance_fraction_median={label:float(np.median(stochastic_variance[selection&full]/np.maximum(expected_error[selection&full],1e-12))) for label,selection in [('ID_calibration',cal),('ID_check',check),('expert',expert)]},
                limitations=['Only2 independent generated chunks per observation; variance estimates are noisy.',
                             '20 original ID demonstration episodes calibrate feedback;8 check episodes were part of policy training.',
                             'A high expert-action discrepancy is not a failure label or proof of late takeover.',
                             'No alternative is selected using SR; live pilot keeps its originally frozen expected-error point threshold.'])
    (ROOT/'optimization_diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
