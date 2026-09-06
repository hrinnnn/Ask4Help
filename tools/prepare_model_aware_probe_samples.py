"""Fixed raw-observation samples for actual X-VLA gradient/learning probes."""
import json
from pathlib import Path
import numpy as np

OUT=Path('artifacts/model_aware_timing_20260906')

def main():
 OUT.mkdir(exist_ok=True);manifest=json.loads(Path('artifacts/cross_asset_overlap_utility_20260903/inputs/stage2/manifest.json').read_text());rng=np.random.default_rng(260906);specs=[];used=set();nominal=[]
 def add_samples(label,rows,pool,n):
  positions=[(r,i) for r in rows for i in range(r['train']['expert_action_steps'])]
  chosen=rng.choice(len(positions),size=min(n,len(positions)),replace=False)
  for k in chosen:
   r,i=positions[int(k)];specs.append(dict(group=label,seed=r['meta']['seed'],split=r['meta']['split'],episode=r['train']['dataset_episode_index'],offset=i,
    path=str(Path(pool)/f"data/chunk-000/episode_{r['train']['dataset_episode_index']:06d}.parquet"),episode_length=r['train']['expert_action_steps'],
    actual_takeover=r['train']['expert_start_step'],role='query' if label.startswith('query_') else label))
 for g in manifest:
  rows=json.loads(Path(g['episodes_file']).read_text());selected=[r for r in rows if r['selected']];used|={r['meta']['seed'] for r in selected}
  add_samples('query_'+g['method'],selected,g['selection']['source_pool'],128)
  if g['method']=='immediate':nominal=rows;nominalpool=g['selection']['source_pool']
 remaining=sorted([r for r in nominal if r['meta']['seed'] not in used],key=lambda r:r['meta']['seed']);assert len(remaining)>=40
 for label,rows in [('ood_reference',remaining[:20]),('ood_probe',remaining[20:40])]:add_samples(label,rows,nominalpool,96)
 summary=dict(samples=specs,ood_reference_seeds=[r['meta']['seed'] for r in remaining[:20]],ood_probe_seeds=[r['meta']['seed'] for r in remaining[20:40]],all_selected_training_seeds=sorted(used),
  id_samples_to_add=96,id_root='/mnt/data/ask4help/datasets/lerobot/local/stackcube_id_128_visual_v1_20260722',sampling_seed=260906)
 assert not(set(summary['ood_reference_seeds'])&set(summary['ood_probe_seeds']))
 assert not((set(summary['ood_reference_seeds'])|set(summary['ood_probe_seeds']))&used)
 (OUT/'sample_spec.json').write_text(json.dumps(summary,indent=2));print('SPECS',len(specs),'reference/probe disjoint',flush=True)

if __name__=='__main__':main()
