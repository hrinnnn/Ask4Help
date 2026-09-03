#!/usr/bin/env python3
"""Export exact selected Stage-2 expert suffixes and nominal reference pool."""
import argparse,gzip,json,subprocess
from pathlib import Path
import numpy as np

REMOTE=r'''
import gzip,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
b=Path('/mnt/data/ask4help/results/xvla_stackcube_v1/temporal_mask_v2/stackcube_target_ood_timing_v1_retry2')
g=Path('/root/ask4help_stage2_work/xvla_stackcube_diff_trigger_repair_v1_training_prep_matched_budget_434_retry1/selection/datasets_budget_434')
requests=[('stack_stage2_timing',m,b/'datasets_budget_1968'/m/'selection_manifest.json') for m in ['immediate','post_grasp','post_lift','failure_recovery']]
requests += [('stack_stage2_gates',m,g/m/'selection_manifest.json') for m in ['internal_pca','diffdagger']]
result=[]
for cohort,method,path in requests:
 selection=json.load(open(path));collection=Path(selection['source_collection']);dataset=Path(selection['source_pool']);chosen=set(selection['selected_source_episode_indices'])
 raw=[json.loads(x) for x in (collection/'episodes.jsonl').read_text().splitlines() if x.strip()]
 train=[json.loads(x) for x in (collection/'training_episodes.jsonl').read_text().splitlines() if x.strip()]
 rows=[]
 for tr in train:
  index=tr['dataset_episode_index']
  if index not in chosen and method!='immediate':continue
  meta=next(r for r in raw if r['seed']==tr['seed'] and r['accepted'])
  table=pq.read_table(dataset/f'data/chunk-000/episode_{index:06d}.parquet',columns=['state','actions'])
  qpos=np.asarray(table['state'].to_pylist(),dtype=np.float32);targets=np.asarray(table['actions'].to_pylist(),dtype=np.float32)
  states=np.load(meta['task_states']);action_path=collection/'raw_archive/actions'/Path(meta['task_states']).name;actions=np.load(action_path)
  start=int(tr['expert_start_step']);length=int(tr['expert_action_steps'])
  assert len(qpos)==length and len(states)==len(actions)+1
  assert np.allclose(targets,actions[start:start+length],atol=1e-6)
  rows.append(dict(cohort=cohort,method=method,selected=index in chosen,meta=meta,train=tr,qpos=qpos.tolist(),actions=actions.tolist(),task_states=states.tolist()))
  if len(rows)%50==0:print(cohort,method,'exported',len(rows),file=sys.stderr,flush=True)
 assert sum(r['train']['expert_action_steps'] for r in rows if r['selected'])==selection['budget']
 result.append(dict(cohort=cohort,method=method,selection=selection,selection_path=str(path),episodes=rows))
 print('EXPORTED',cohort,method,len(rows),file=sys.stderr,flush=True)
sys.stdout.buffer.write(gzip.compress(json.dumps(result).encode()))
'''

def main(output):
 output.mkdir(parents=True,exist_ok=True)
 proc=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p','1012','root@39.101.70.188','/root/.venvs/xvla-h20/bin/python -'],input=REMOTE.encode(),stdout=subprocess.PIPE,check=True)
 payload=json.loads(gzip.decompress(proc.stdout));manifest=[]
 for group in payload:
  folder=output/group['cohort']/group['method'];folder.mkdir(parents=True,exist_ok=True)
  rows=group.pop('episodes')
  for r in rows:
   arrays={k:np.asarray(r.pop(k),dtype=np.float32) for k in ['qpos','actions','task_states']}
   path=folder/f"seed_{r['meta']['seed']}.npz";np.savez_compressed(path,**arrays);r['arrays']=str(path.resolve())
  (folder/'episodes.json').write_text(json.dumps(rows,indent=2))
  group['episodes_file']=str((folder/'episodes.json').resolve());manifest.append(group)
 (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print('STAGE2_NUMERIC_EXPORT_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
