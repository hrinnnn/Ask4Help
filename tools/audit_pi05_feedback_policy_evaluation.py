"""Independent reserved-seed/array/outcome audit before recording a final SR."""
import argparse,json
from pathlib import Path
import numpy as np
from pi05_feedback_artifacts import episode_path


def run(root):
    provenance=json.loads((root/'provenance.json').read_text())
    summary=json.loads((root/'summary.json').read_text());rows=summary['rows']
    assert (root/'EVALUATION_COMPLETE.json').exists() and summary['complete']
    assert len(rows)==summary['episodes']==len(provenance['seeds'])==100
    assert provenance['runtime']['inference_mode']=='eval' and not provenance['expert_or_gate_used']
    assert provenance['purpose']=='reserved_final_evaluation'
    assert summary['checkpoint']==provenance['runtime']['checkpoint']
    assert [r['seed'] for r in rows]==provenance['seeds'] and len(set(provenance['seeds']))==100
    horizon=100 if summary['task']=='stackcube_legacy_ood' else (400 if summary['task'].startswith('open_drawer_') else 250)
    for i,r in enumerate(rows):
        assert r['episode']==i and r['split']==summary['split']
        with np.load(episode_path(root,r)/'trace.npz') as data:
            n=len(data['actions']);assert n==r['actions']<=horizon
            assert len(data['qpos'])==len(data['main'])==len(data['wrist'])==n+1
            assert data['main'].shape[1:]==data['wrist'].shape[1:]==(384,384,3)
            assert np.isfinite(data['actions']).all() and data['actions'].shape[1:]==(8,)
            assert len(data['all_expert_actions'])==r['expert_actions']==0 and r['takeover'] is None
            assert bool(data['eval_strict_success'][-1])==r['strict_success']
            assert bool(data['grasped'].any())==r['ever_grasped']
            expected=r[summary['primary_endpoint']]
            assert r['success']==expected
        assert (episode_path(root,r)/'trajectory.mp4').is_file()
    assert summary['successes']==sum(r['success'] for r in rows)
    assert summary['strict_successes']==sum(r['strict_success'] for r in rows)
    assert summary['ever_grasped_successes']==sum(r['ever_grasped'] for r in rows)
    result={'status':'INDEPENDENT_UNASSISTED_EVALUATION_PASS','episodes':100,'successes':summary['successes'],
            'SR':summary['successes']/100,'strict_successes':summary['strict_successes'],
            'primary_endpoint':summary['primary_endpoint'],'checkpoint':summary['checkpoint']}
    (root/'INDEPENDENT_EVALUATION_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
