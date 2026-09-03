#!/usr/bin/env python3
import argparse,gzip,json
from pathlib import Path

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tasks',nargs='+',required=True);p.add_argument('--name',required=True);a=p.parse_args()
 keys=set()
 for task in a.tasks:
  result=json.loads((a.root/f'tasr_{task}.json').read_text())
  keys.update((g['row']['task'],g['row']['method'],g['row']['source_episode_index']) for g in result['gaps'])
 payload=json.loads(gzip.decompress((a.root/'replay_bundle.json.gz').read_bytes()));payload['rows']=[r for r in payload['rows'] if (r['task'],r['method'],r['source_episode_index']) in keys]
 (a.root/f'{a.name}.json.gz').write_bytes(gzip.compress(json.dumps(payload).encode()));print('RETRY_ROWS',len(payload['rows']))
