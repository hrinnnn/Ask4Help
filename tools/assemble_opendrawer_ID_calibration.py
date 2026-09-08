"""Assemble all32 dense ID demos and50 policy episodes, never tune on gate results."""
import argparse,io,json
from pathlib import Path
import numpy as np
import torch


def run(a):
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    groups={};provenance=None
    for component in ['demos','policies']:
        rows=[]
        for shard in [0,1]:
            root=a.components/f'{component}_{shard}'
            assert (root/'COMPONENT_COMPLETE.json').exists()
            p=json.loads((root/'provenance.json').read_text())
            assert p['runtime']['inference_mode']=='eval'
            if provenance is None:provenance=p
            assert p['runtime']['checkpoint']==provenance['runtime']['checkpoint']
            assert p['runtime']['norm']==provenance['runtime']['norm']
            assert p['reference_asset']==provenance['reference_asset']==str(a.reference)
            rows.extend(json.loads((root/'rows.json').read_text()))
        groups[component]=sorted(rows,key=lambda r:r['episode'])
    demos=groups['demos'];policies=groups['policies']
    assert len(demos)==32 and len(policies)==50
    assert [r['episode'] for r in demos]==np.linspace(0,127,32,dtype=int).tolist()
    assert [r['seed'] for r in policies]==list(range(1781000,1781050))
    successes=[r['maximum'] for r in policies if r['calibration_success']]
    (a.output/'ID_policy_progress.json').write_text(json.dumps(policies))
    reversals=[v for r in demos for v in r.pop('reversals')]
    (a.output/'demo_progress.json').write_text(json.dumps(demos))
    (a.output/'motion_calibration.json').write_text(json.dumps(reversals))
    assert len(successes)>=20,f'Only {len(successes)} successful eval-mode ID episodes; no gate threshold'
    asset=torch.load(a.reference,map_location='cpu',weights_only=False)
    features=asset['layers']['vlm_bridge_final_mean'].float().reshape(-1,2048)
    pca=asset['detectors']['vlm_bridge_final_mean__pca_residual']['statistics']
    dataset=Path(provenance['ID_dataset'])
    ep=[json.loads(x) for x in (dataset/'meta/episodes.jsonl').read_text().splitlines()]
    labels=np.concatenate([np.repeat(r['episode_index'],r['length']) for r in ep])
    assert len(features)==len(labels)==22973
    center=features.mean(0);scale=float(torch.sqrt(torch.mean(torch.sum((features-center)**2,dim=1))))
    z=(features-center)/scale;nearest=[]
    for start in range(0,len(z),128):
        distance=torch.cdist(z[start:start+128],z)
        distance[torch.as_tensor(labels[start:start+128,None]==labels[None,:])]=torch.inf
        nearest.extend(distance.min(1).values.numpy().tolist())
    radius=float(np.quantile(nearest,.95,method='higher'))
    buf=io.BytesIO();np.savez_compressed(buf,mean=pca['mean'].float().reshape(-1).numpy(),
                                       basis=pca['principal_components'].float().squeeze(0).numpy(),center=center.numpy())
    (a.output/'gate_arrays.npz').write_bytes(buf.getvalue())
    result={'task':'open_drawer_grasp_ood','baseline_threshold':float(np.quantile(successes,.95,method='higher')),
            'error_reference':float(np.quantile([r['max_error'] for r in demos],.95,method='higher')),
            'reversal_reference':float(np.quantile(reversals,.95,method='higher')),'scale':scale,'radius':radius,
            'principal_dim':int(pca['principal_dim']),'ID_demonstrations':32,'policy_episodes':50,
            'qualifying_ID_policy_episodes':len(successes),'calibration_success_rule':'strict',
            'provenance':provenance['runtime'],'reference_assets':{'pca':str(a.reference),'cache':str(a.reference)},
            'source_dataset':str(dataset),'array_file':'gate_arrays.npz','seed_start':1781000,
            'reused_calibration':None,'mode_repair':'Original OpenDrawer gated/fixed-timing/eval code uses eval; generic SDE50 retained as different-policy diagnostic.'}
    (a.output/'calibration.json').write_text(json.dumps(result,indent=2))
    (a.output/'CALIBRATION_COMPLETE.json').write_text(json.dumps({'status':'ID_ONLY_EVAL_MODE_CALIBRATION_COMPLETE'}))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['components','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        if a.output.exists():(a.output/'CALIBRATION_FAILED.json').write_text(json.dumps({'error':repr(e)}))
        raise
