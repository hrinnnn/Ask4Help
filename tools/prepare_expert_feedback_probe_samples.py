"""Contiguous expert segments for within-round feedback; no precollected OOD reference."""
import json
from pathlib import Path

OUT=Path('artifacts/expert_feedback_pca_20260907')

def main():
 OUT.mkdir(exist_ok=True);manifest=json.loads(Path('artifacts/cross_asset_overlap_utility_20260903/inputs/stage2/manifest.json').read_text());samples=[];episodes=[]
 for g in manifest:
  rows=[r for r in json.loads(Path(g['episodes_file']).read_text()) if r['selected']]
  # Deterministic source order; no sampling by score, success rate or desired timing.
  chosen=sorted(rows,key=lambda r:r['train']['dataset_episode_index'])[:8]
  for r in chosen:
   ep=r['train']['dataset_episode_index'];n=r['train']['expert_action_steps'];seed=r['meta']['seed'];method=g['method'];eid=f'{method}:{seed}:{ep}'
   episodes.append(dict(episode_id=eid,method=method,seed=seed,length=n,split=r['meta']['split'],takeover=r['train']['expert_start_step']))
   for i in range(n):samples.append(dict(group='expert_'+method,seed=seed,split=r['meta']['split'],episode=ep,episode_id=eid,offset=i,path=str(Path(g['selection']['source_pool'])/f'data/chunk-000/episode_{ep:06d}.parquet'),episode_length=n,actual_takeover=r['train']['expert_start_step'],role='naturally_collected_expert'))
 spec=dict(samples=samples,id_samples_to_add=0,id_contiguous_episodes=list(range(100,128)),id_root='/mnt/data/ask4help/datasets/lerobot/local/stackcube_id_128_visual_v1_20260722',sampling_seed=270907,
  id_calibration_episodes=list(range(100,120)),id_check_episodes=list(range(120,128)),episodes=episodes,
  constraint='No OOD reference/calibration labels; earlier naturally collected expert episodes supply memory, later episodes are queried without self feedback. Expert paths provide a support/feedback diagnostic, not autonomous closed-loop timing evidence.')
 (OUT/'sample_spec.json').write_text(json.dumps(spec,indent=2));print('CONTIGUOUS_EXPERT_SAMPLES',len(samples),'episodes',len(episodes))

if __name__=='__main__':main()
