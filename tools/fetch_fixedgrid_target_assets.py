#!/usr/bin/env python3
"""Read-only remote extraction of numeric timing assets; no model/simulator."""
import argparse
import gzip
import json
import subprocess
from pathlib import Path
import numpy as np

REMOTE=r'''
import gzip,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
b=Path('/data/zhaozhixuan/Ask4Help-airplane-5090/results/xvla_fixedgrid_taskpolicy_knee_v1')
result={}
for task,name,budget in [('stackcube','formal_calibration_merged_v2',520),('airplane','airplane_calibration_merged_v2',2820)]:
 root=b/name;knee=json.load(open(root/'knee_summary_recoverable.json'));out={'root':str(root),'budget':budget,'anchors':knee['anchors'],'conditions':{}}
 for step in knee['anchors']:
  cal=root/f'calibration/step_{step}';dataset=root/f'datasets/step_{step}'
  raw=[json.loads(s) for s in (cal/'episodes.jsonl').read_text().splitlines() if s.strip()]
  train=[json.loads(s) for s in (dataset/'training_episodes.jsonl').read_text().splitlines() if s.strip()]
  selection=json.load(open(root/f'timing_datasets_budget_{budget}/step_{step}/selection_manifest.json'))
  chosen=set(selection['selected_source_episode_indices'])
  needed=[r for r in train if step==0 or r['dataset_episode_index'] in chosen]
  rows=[]
  for r in needed:
   meta=next(x for x in raw if x['seed']==r['seed'])
   index=r['dataset_episode_index'];table=pq.read_table(dataset/f'data/chunk-000/episode_{index:06d}.parquet',columns=['state','actions'])
   states=np.array(table['state'].to_pylist(),dtype=np.float32);targets=np.array(table['actions'].to_pylist(),dtype=np.float32)
   raw_states=np.load(meta['task_states']);raw_actions=np.load(meta['actions'])
   start=int(r['expert_start_step']);length=int(r['expert_action_steps'])
   assert len(states)==length and len(targets)==length
   assert len(raw_states)==len(raw_actions)+1
   assert np.allclose(targets,raw_actions[start:start+length],atol=1e-6)
   rows.append({'meta':meta,'train':r,'selected':index in chosen,'qpos':states.tolist(),'actions':raw_actions.tolist(),'task_states':raw_states.tolist()})
  assert sum(x['train']['expert_action_steps'] for x in rows if x['selected'])==budget
  out['conditions'][str(step)]={'selection':selection,'raw_count':len(raw),'raw_accepted':sum(bool(x.get('accepted')) for x in raw),'episodes':rows}
  print(task,step,'numeric episodes',len(rows),'selected',len(chosen),file=sys.stderr,flush=True)
 result[task]=out
sys.stdout.buffer.write(gzip.compress(json.dumps(result).encode()))
'''


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p','12001','zhaozhixuan@111.198.58.150',
        '/data/zhaozhixuan/Ask4Help-airplane-5090/RLinf/.venv/bin/python','-'],input=REMOTE.encode(),stdout=subprocess.PIPE,check=True)
    payload=json.loads(gzip.decompress(proc.stdout));manifest={}
    for task,data in payload.items():
        manifest[task]={k:v for k,v in data.items() if k!='conditions'};manifest[task]['conditions']={}
        for step,cond in data['conditions'].items():
            directory=output/task/f'step_{step}';directory.mkdir(parents=True,exist_ok=True)
            rows=[]
            for row in cond['episodes']:
                arrays={k:np.asarray(row.pop(k),dtype=np.float32) for k in ['qpos','actions','task_states']}
                seed=row['meta']['seed'];path=directory/f'seed_{seed}.npz';np.savez_compressed(path,**arrays)
                row['arrays']=str(path);rows.append(row)
            (directory/'episodes.json').write_text(json.dumps(rows,indent=2))
            manifest[task]['conditions'][step]={k:v for k,v in cond.items() if k!='episodes'}
            manifest[task]['conditions'][step]['episodes_file']=str(directory/'episodes.json')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('NUMERIC_ASSETS_EXPORTED',output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
