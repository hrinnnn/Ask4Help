#!/usr/bin/env python3
"""Separate audit of threshold-only variants; no matching/recalibration calls."""
import argparse,json,subprocess
from pathlib import Path


def main(root):
    report=json.loads((root/'sensitivity.json').read_text());base=json.loads((root/'baseline_analysis.json').read_text())
    assert json.loads(Path(report['source_analysis']).read_text())==base
    lookup={r['seed']:r for r in base['rows']};contact={}
    def states(seed):
        if seed not in contact:
            rows=json.loads((Path(lookup[seed]['directory'])/'task_state_timeline.json').read_text())['rows']
            contact[seed]=[r['object_grasped'] for r in rows]
        return contact[seed]
    previous={};results=[]
    for f in report['factors']:
        folder=root/f'factor_{f:.2f}'.replace('.','p');v=json.loads((folder/'analysis.json').read_text())
        assert len(v['rows'])==len(base['rows'])==262
        total_green=0
        for row in v['rows']:
            old=lookup[row['seed']]
            for key in ['reference_seed','reference_mapping','distance','blocks','expert_anchors','accepted_index']:
                assert row[key]==old[key]
            qc=states(row['seed']);rc=states(row['reference_seed']);count=0
            for i,label in enumerate(row['labels']):
                t=row['takeover']+i
                phase=next((p for p,(a,b) in old['blocks'].items() if a<=t<b),None)
                if phase is None:assert label=='non_target';continue
                j=old['reference_mapping'][i];radius=old['thresholds'][phase]*f
                expected=bool(qc[t]==rc[j] and old['distance'][i]<=radius)
                assert expected==(label=='target_compatible')
                count+=int(expected)
                if row['seed'] in previous and previous[row['seed']][i]=='target_compatible':assert expected
            assert count==row['compatible_anchors'];total_green+=count;previous[row['seed']]=row['labels']
            assert row['expert_anchors']==0 or abs(row['Q']-count/row['expert_anchors'])<1e-12
        for a in [0,50,80,120,160,220]:
            selected=[r for r in v['rows'] if r['anchor']==a and r['accepted'] and r['accepted_index'] in base['budget_manifest']['selected_source_episode_indices'][f'anchor_{a}']]
            n=sum(r['expert_anchors'] for r in selected);g=sum(r['compatible_anchors'] for r in selected)
            assert n==2413 and g==v['summary'][str(a)]['training_budget']['compatible_anchors']
        results.append(dict(factor=f,rows=262,total_green=total_green))
    videos=[]
    if (root/'video_audit.json').exists():
        for v in json.loads((root/'video_audit.json').read_text()):
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,width,height','-of','json',v['output']]))['streams'][0]
            assert int(info['nb_frames'])==v['frames']
            videos.append(dict(seed=v['seed'],frames=v['frames']))
    out=dict(status='PASS_SENSITIVITY_AUDIT',baseline_unchanged=True,variants=results,videos=videos,
             no_formal_threshold_selected=True,no_SR_predictive_validation=True)
    (root/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
