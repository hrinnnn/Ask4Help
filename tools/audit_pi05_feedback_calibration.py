"""Read-only independent quantile/denominator audit before using a gate."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(root):
    cal=json.loads((root/'calibration.json').read_text())
    demos=json.loads((root/'demo_progress.json').read_text())
    policies=json.loads((root/'ID_policy_progress.json').read_text())
    assert len(demos)==cal['ID_demonstrations']==32
    assert len(policies)==cal['policy_episodes']==50
    assert len({d['episode'] for d in demos})==len(demos)
    assert len({p['seed'] for p in policies})==len(policies)
    for d in demos:
        assert len(d['errors'])==d['anchors']-4==d['observed_windows']
        assert np.isfinite(d['errors']).all() and min(d['errors'])>=0
        assert max(d['errors'])==d['max_error']
    for p in policies:
        assert p['maximum']==max(p['scores']) and len(p['scores'])>0
        assert p['calibration_success']==(p['strict_success'] if cal['calibration_success_rule']=='strict' else p['ever_grasped'])
    successful=[p['maximum'] for p in policies if p['calibration_success']]
    assert len(successful)==cal['qualifying_ID_policy_episodes'] and len(successful)>=20
    assert np.quantile(successful,.95,method='higher')==cal['baseline_threshold']
    assert np.quantile([d['max_error'] for d in demos],.95,method='higher')==cal['error_reference']
    data=np.load(root/'gate_arrays.npz')
    assert data['mean'].shape==data['center'].shape==(2048,)
    assert data['basis'].shape==(2048,cal['principal_dim'])
    gram=data['basis'].T@data['basis']
    assert np.max(abs(gram-np.eye(cal['principal_dim'])))<1e-4
    return {'status':'PASS','task':cal['task'],'ID_demo_episodes':len(demos),'ID_policy_episodes':len(policies),
            'qualifying_ID_successes':len(successful),'baseline_threshold':cal['baseline_threshold'],
            'error_reference':cal['error_reference'],'orthonormal_basis_error':float(np.max(abs(gram-np.eye(cal['principal_dim']))))}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    report=audit(args.root);(args.root/'INDEPENDENT_CALIBRATION_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
