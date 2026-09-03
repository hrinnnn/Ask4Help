#!/usr/bin/env python3
"""Extract exact pi05 training suffixes, not similarly named earlier datasets."""
import argparse,gzip,json,subprocess
from pathlib import Path
import numpy as np

REMOTE=r'''
import gzip,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
result=[]
for item in items:
 collection=Path(item['collection']);dataset=Path(item['dataset']);indexroot=Path(item.get('index_root',item['collection']))
 if item['task']=='object':
  raw=[json.loads(s) for s in (collection/'attempts.jsonl').read_text().splitlines() if s.strip()];accepted=[r for r in raw if r['accepted']]
  budget=json.load(open(item['budget_manifest']));selected=set(budget['selected_source_episode_indices'][item['method']])
  train=[dict(dataset_episode_index=i,seed=r['seed'],split=r['split'],expert_start_step=r['expert_start_step'],raw_episode_index=r['attempt']) for i,r in enumerate(accepted)]
 else:
  raw=[json.loads(s) for s in (collection/'episodes.jsonl').read_text().splitlines() if s.strip()]
  train=[json.loads(s) for s in (indexroot/'training_episodes.jsonl').read_text().splitlines() if s.strip()];selected={r['dataset_episode_index'] for r in train}
 rows=[]
 for tr in train:
  i=tr['dataset_episode_index']
  if i not in selected and item['method']!='offline_oracle':continue
  meta=next(r for r in raw if r['seed']==tr['seed'] and r['split']==tr['split']);start=tr.get('expert_start_step',tr.get('start_step'))
  ri=tr.get('raw_attempt_index',tr.get('raw_episode_index'));seed=tr['seed']
  source=collection/'raw_archive/actions'/f'episode_{ri:06d}_seed_{seed:06d}.npy'
  table=pq.read_table(dataset/f'data/chunk-000/episode_{i:06d}.parquet',columns=['state','actions'])
  qpos=np.asarray(table['state'].to_pylist(),dtype=np.float32);target=np.asarray(table['actions'].to_pylist(),dtype=np.float32);actions=np.load(source)
  assert len(qpos)==len(target) and np.allclose(target,actions[start:start+len(target)],atol=1e-6),(item['task'],item['method'],i)
  rows.append(dict(task=item['task'],method=item['method'],source_episode_index=i,selected=i in selected,seed=seed,split=tr['split'],expert_start=start,
   expert_points=len(qpos),source_parquet=str(dataset/f'data/chunk-000/episode_{i:06d}.parquet'),source_actions=str(source),
   video=tr.get('video_path',meta.get('video',meta.get('video_path'))),metadata={k:v for k,v in meta.items() if k not in ['timeline','sources'] and 'sha256' not in k},
   qpos=qpos.tolist(),actions=actions.tolist()))
  if len(rows)%50==0:print(item['task'],item['method'],'read',len(rows),file=sys.stderr,flush=True)
 total=sum(r['expert_points'] for r in rows if r['selected'])
 if item['task']=='object':assert total==budget['common_expert_action_budget']
 result.append(dict(**item,expert_points=total,selected_episodes=sum(r['selected'] for r in rows),episodes=rows))
 print('COHORT_EXPORTED',item['task'],item['method'],total,file=sys.stderr,flush=True)
sys.stdout.buffer.write(gzip.compress(json.dumps(result).encode()))
'''

def main(output):
 output.mkdir(parents=True,exist_ok=True);h='/mnt/data/ask4help/results/';s=h+'stackcube_gated_dagger/';p=h+'pick_single_ycb_airplane/four_group_dagger_v1/formal_v3/';y='/data/zhaozhixuan/Ask4Help-airplane-5090/results/object_variation_pick_single_ycb_v1/'
 itemsH=[dict(task='stackcube',method='bridge_knn',collection=s+'full_v1/bridge_knn_successful_experts_100',index_root=s+'full_v2/bridge_knn_successful_experts_100_full_suffix_retry1',dataset=s+'full_v2/bridge_knn_successful_experts_100_full_suffix_retry1/dataset'),
  dict(task='stackcube',method='offline_oracle',collection=s+'pilot_v1/offline_oracle_100',index_root=s+'full_v2/two_gpu_lockstep_v1/offline_oracle_100_full_rebuilt',dataset=s+'full_v2/two_gpu_lockstep_v1/offline_oracle_100_full_rebuilt/dataset')]
 for m,rel in [('diffdagger','full_v5/diffdagger_successful_experts_100_from7000_retry2/collection'),('failure_recovery','full_v5/failure_recovery_successful_experts_100_from7000_retry1')]:itemsH.append(dict(task='stackcube',method=m,collection=s+rel,dataset=s+rel+'/dataset'))
 for m,folder in [('offline_oracle','offline_oracle_collection'),('bridge_pca','bridge_pca_collection'),('diffdagger','diffdagger_collection_gatecal_retry1'),('failure_recovery','failure_recovery_collection')]:itemsH.append(dict(task='airplane',method=m,collection=p+folder,dataset=p+folder+'/lerobot'))
 itemsY=[]
 for m,folder,ds in [('offline_oracle','collections_v1/offline_oracle','offline_oracle_v1'),('bridge_pca','collections_v1/bridge_pca_retry1','bridge_pca_v1_retry1'),('diffdagger','collections_diagnostic_v1/diffdagger_low_threshold_005_retry1','diffdagger_low_threshold_005_retry1_v1'),('failure_recovery','collections_v1/failure_recovery','failure_recovery_v1')]:
  itemsY.append(dict(task='object',method=m,collection=y+folder,dataset=y+'datasets/'+ds,budget_manifest=y+'diagnostic_low_threshold_005_v1_retry3/matched_expert_budget/budget_manifest.json'))
 manifest=[]
 for port,host,python,items in [('12001','zhaozhixuan@111.198.58.150','/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python',itemsY),('1012','root@39.101.70.188','/root/.venvs/xvla-h20/bin/python',itemsH)]:
  proc=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p',port,host,python+' -'],input=('items='+repr(items)+'\n'+REMOTE).encode(),stdout=subprocess.PIPE,check=True)
  for group in json.loads(gzip.decompress(proc.stdout)):
   rows=group.pop('episodes');folder=output/group['task']/group['method'];folder.mkdir(parents=True,exist_ok=True)
   for r in rows:
    arrays={k:np.asarray(r.pop(k),dtype=np.float32) for k in ['qpos','actions']};path=folder/f"episode_{r['source_episode_index']:06d}.npz";np.savez_compressed(path,**arrays);r['arrays']=str(path.resolve())
   f=folder/'episodes.json';f.write_text(json.dumps(rows,indent=2));group['episodes_file']=str(f.resolve());manifest.append(group)
  (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('PI05_EXACT_TRAINING_SUFFIX_EXPORT_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
