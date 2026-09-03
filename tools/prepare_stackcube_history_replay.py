#!/usr/bin/env python3
"""Restore the archived reset order, including unselected raw attempts."""
import argparse,gzip,json,subprocess
from pathlib import Path

REMOTE=r'''
from pathlib import Path
import gzip,json,sys,numpy as np
out={}
for group in groups:
 method=group['method'];collection=Path(group['collection']);indexroot=Path(group.get('index_root',group['collection']))
 if 'index_root' in group:
  rows=[json.loads(x) for x in (indexroot/'training_episodes.jsonl').read_text().splitlines() if x.strip()]
  out[method]=[dict(index=r['dataset_episode_index'],warmups=r.get('replay_attempts',1)-1) for r in rows]
 else:
  rows=[json.loads(x) for x in (collection/'episodes.jsonl').read_text().splitlines() if x.strip()];items=[]
  for r in rows:
   path=collection/'raw_archive/actions'/f'episode_{r["episode_index"]:06d}_seed_{r["seed"]:06d}.npy'
   items.append(dict(seed=r['seed'],split=r['split'],actions=np.load(path).tolist()))
  out[method]=items
 print(method,len(out[method]),file=sys.stderr,flush=True)
sys.stdout.buffer.write(gzip.compress(json.dumps(out).encode()))
'''

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
 groups=[g for g in json.loads((root/'training_assets/manifest.json').read_text()) if g['task']=='stackcube']
 result=subprocess.run(['ssh','-o','ConnectTimeout=12','-o','BatchMode=yes','-p','1012','root@39.101.70.188','/root/.venvs/xvla-h20/bin/python -'],
  input=('groups='+repr(groups)+'\n'+REMOTE).encode(),stdout=subprocess.PIPE,check=True)
 history=json.loads(gzip.decompress(result.stdout));(root/'stackcube_history_metadata.json').write_text(json.dumps(history,indent=2))
 payload=json.loads(gzip.decompress((root/'replay_bundle.json.gz').read_bytes()));original=[r for r in payload['rows'] if r['task']=='stackcube'];combined=[]
 for g in groups:
  method=g['method'];bank=[r for r in original if r['method']==method];byindex={r['source_episode_index']:r for r in bank};byseed={(r['seed'],r['split']):r for r in bank}
  for i,h in enumerate(history[method]):
   if 'index' in h:r=dict(byindex[h['index']],recorded_warmups=h['warmups'])
   elif (h['seed'],h['split']) in byseed:r=byseed[h['seed'],h['split']]
   else:r=dict(task='stackcube',method=method,source_episode_index=-i-1,seed=h['seed'],split=h['split'],actions=h['actions'],expert_start=0,history_only=True)
   combined.append(r)
 payload['rows']=combined;(root/'stackcube_history_bundle.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()))
 print('STACK_HISTORY_ROWS',len(combined),flush=True)
